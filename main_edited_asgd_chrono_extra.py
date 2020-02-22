import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn

import data
import model_chrono as model
import copy

from pathlib import Path
import sys
path = Path(__file__).parent.absolute()
sys.path.append(str(path)+'/cottoncandy')


import os
os.environ["LD_PRELOAD"]='/home/shivangi/anaconda3/lib/libstdc++.so.6.0.25' 
import matplotlib.pyplot as plt

from utils import batchify, get_batch, repackage_hidden


import cottoncandy as cc
access_key = 'SSE14CR7P0AEZLPC7X0R'
secret_key = 'K0MmeXiXotrGIiTeRwEKizkkhR4qFV8tr8cIXprI'
endpoint_url = 'http://c3-dtn02.corral.tacc.utexas.edu:9002/'
cci = cc.get_interface('lstm-timescales', verbose=False,ACCESS_KEY=access_key, SECRET_KEY=secret_key,endpoint_url=endpoint_url)

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')

##added by shivi
parser.add_argument('--Tmax', type=int,  default=20)
parser.add_argument('--Tmin', type=int,  default=1)

args = parser.parse_args()
args.tied = True

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    #with open(fn, 'wb') as f:
    #    torch.save([model, criterion, optimizer], f)
    cci.upload_pickle('shivangi/'+fn, [model, criterion, optimizer])

def model_copy_save(fn):
    #with open(fn, 'wb') as f:
    #    torch.save([model_copy, criterion, optimizer], f)
    cci.upload_pickle('shivangi/'+fn, [model_copy, criterion, optimizer])

def model_load(fn):
    global model, criterion2, optimizer
    try:
        with open(fn, 'rb') as f:
            model, criterion2, optimizer = torch.load(f)
    except:
        print('Downloading from pickle')
        model, criterion2, optimizer = cci.download_pickle('shivangi/'+fn)

import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)


#######                                                                                                                                    
#1.                                                                                                                                        
vocab_dict = corpus.dictionary.counter
sorted(vocab_dict.keys(),key=vocab_dict.get,reverse=True)
#print(vocab_dict)

total_count = sum(vocab_dict.values())
count=0
high_freq=set()
#print( total_count/2)
for key in vocab_dict.keys():
    if count < total_count/2:
       count+=vocab_dict[key]
       high_freq.add(key)

####### 
eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)


###############################################################################
# Build the model
###############################################################################

from splitcross_copy import SplitCrossEntropyLoss
criterion = None

ntokens = len(corpus.dictionary)
UNK_ind = corpus.dictionary.word2idx['<unk>']
Tmax = args.Tmax; Tmin= args.Tmin
model_copy = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied, args.Tmax, args.Tmin)
for l in range(args.nlayers):
    model_copy.rnns[l]._setweights()
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied, args.Tmax, args.Tmin)

###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop
        for rnn in model.rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
            elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
###
if args.cuda:
    model.cuda()
    model_copy.cuda()
    criterion.cuda()
###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    #with torch.no_grad():
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion2(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)

def evaluate_copy(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.                                                                                      
 
    model_copy.eval()

    if args.model == 'QRNN': model_copy.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model_copy.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model_copy(data, hidden)
        total_loss += len(data) * criterion(model_copy.decoder.weight, model_copy.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)

entropy_vec = [] #np.zeros(len(corpus.dictionary))
target_vec = []

def eval_story(data_matrix):
    # Turn on evaluation mode which disables dropout.                                                                                       
    model.eval()

    batch_size = len(data_matrix)
    hidden = model.init_hidden(batch_size)
    
    data_tensor = torch.tensor(np.transpose(data_matrix)).cuda()    
    output, hidden, raw_outputs, outputs = model(data_tensor, hidden, return_h=True)
    
    
    return [ np.swapaxes(x.cpu().detach().numpy(), 0,1) for x in outputs]

def eval_hig_low_fre(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.                                                                                      
    model.eval()
    #with torch.no_grad():                                                                                                                 
    target_vec, entropy_vec = [], []
    print('Hey')
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    
    for i in range(0,data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        x = len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        
        target_vec.extend(targets.cpu().numpy().tolist())
        entropy_vec.extend(x.squeeze(1).cpu().numpy().tolist())

        hidden = repackage_hidden(hidden)
    
    return entropy_vec , len(data_source) - 1, target_vec

def train():
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < 20: #train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()
        
        # added by shivi to keep reassigning the bias values
        #dict_param = dict(model.named_parameters())
        
        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len
    #added by shivi
    torch.cuda.empty_cache()

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000
val_loss_array = []
## Storing bias values along iterations
layer_fb = [ [ ] for l in range(args.nlayers)]
print('Should be a list of three empty list', layer_fb)
# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        torch.cuda.empty_cache()
        print('Epochs',epoch)
        print('used memory before',torch.cuda.memory_allocated()/1e9)
        print('cached memory beforec',torch.cuda.memory_cached()/1e9)
        
        if 't0' in optimizer.param_groups[0]:

            print("'"*89)
            
            l1 = dict(model.named_parameters())
            l2 = dict(model_copy.named_parameters())
                
            for prm_name in l1.keys():
                try: 
                    l2[prm_name].data = optimizer.state_dict()['state'][id(l1[prm_name])]['ax'] 
                except:
                    l2[prm_name].data = l1[prm_name].data
                
                
            val_loss2 = evaluate_copy(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                      epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            print('-' * 89)
            val_loss_array.append(val_loss2)
            if val_loss2 < stored_loss:
                model_copy_save(args.save)
                print('Saving Averaged!')
                stored_loss = val_loss2
                
            torch.cuda.empty_cache()
            print('used memory after',torch.cuda.memory_allocated()/1e9)
            print('cached memory after',torch.cuda.memory_cached()/1e9)

                        
        else:
            val_loss = evaluate(val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)
            val_loss_array.append(val_loss)
            if val_loss < stored_loss:
                model_save(args.save)
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching to ASGD')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                print('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    print('-=' * 89)
    print('Exiting from training early')

if False: 
    
    model_name = args.save.split('.')[0]
    plt.figure();
    plt.plot(val_loss_array);
    plt.axhline(y=stored_loss, color='r', linestyle='-');
    plt.savefig(model_name+'_val_loss_training.png')
    
    model_load(args.save)
        
    # Run on test data.                                                                                                                     
    test_loss = evaluate(test_data, test_batch_size) 
    #test_loss = eval_hig_low_fre(test_data, test_batch_size)
    """
    entropy_loss, l, target_vec = eval_hig_low_fre(test_data, test_batch_size)
    test_loss = sum(entropy_loss) / l
    
    """
    print('=' * 89)                                                                                                                         
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(                                             
    test_loss, math.exp(test_loss), test_loss / math.log(2)))                                                                          
    print('=' * 89)                                                                                                                         
    
    """
        
    high_freq_loss = 0
    for target_ind in range(len(target_vec)):
        if target_vec[target_ind] in high_freq:
            high_freq_loss += entropy_loss[target_ind] 
           
    """
    exit(0)
    
    #print forget gate bias values for ensurity

    dict_param = dict(model.named_parameters())                                                                                             
    """
    for l in range(args.nlayers):
        print(initial_bias_ih[l])
        print(initial_bias_hh[l])
        print('All expected to be 1',(dict_param['rnns.'+str(l)+'.module.bias_hh_l0'].data))
        print('All expected to be 0',(dict_param['rnns.'+str(l)+'.module.bias_ih_l0'].data))
    """
    
    hid_dim = [1150,1150,400]                                                                                                               
    
    
    print(dict_param['rnns.'+str(l)+'.module.bias_ih_l0'].data[hid_dim[l]*3:hid_dim[l]*4].cpu() + dict_param['rnns.'+str(l)+'.module.bias_hh_l0'].data[hid_dim[l]*3:hid_dim[l]*4].cpu())


    for l in range(3):                                                                                                                      
        x = (dict_param['rnns.'+str(l)+'.module.bias_ih_l0'].data[0:hid_dim[l]*2].cpu() + dict_param['rnns.'+str(l)+'.module.bias_hh_l0'].data[0:hid_dim[l]*2].cpu())  
        #-----------------------------------------------------------------------------------------------------------------
        #print('Forget gate initialization values for layer',l,'is', math.log( (Tmax/(2**(2-l)))-1))
        print('Input gates bias after training',x[0:hid_dim[l]])
        print('Forget gates bias after training',x[hid_dim[l]:hid_dim[l]+int(hid_dim[l]/2)], x[hid_dim[l]+int(hid_dim[l]/2):hid_dim[l]*2])
        #------------------------------------------------------------------------------------------------------------------
        plt.figure(); plt.plot(np.sort(x[hid_dim[l]:hid_dim[l]*2]));  
        plt.xlabel('Hidden units in sorted order'); plt.ylabel('Forget gate bias values of hidden unit')                               
        #v= math.log( (Tmax/(2**l))-1); plt.title('Input gates bias layer '+str(l+1)+' EV all to be '+str(-1*round(v,2)) )
        plt.savefig('Custom_fg_bias_modular_layer_'+str(l+1)+str(Tmax)+'.png')
        #------------------------------------------------------------------------------------------------------------------
        #plt.figure(); plt.hist(x[hid_dim[l]:hid_dim[l]*2]); plt.xlabel('Values of hidden units'); plt.ylabel('Counts')                     
        #plt.title('Forget_gates_bias_layer_'+str(l+1)+' EV all to be '+str(1*(round(v,2))) )                                               
        #plt.savefig('Fix_FG_bias_layer_f_'+str(Tmax)+'_L_'+str(l+1)+'.png') 
#exit(0)

#Perform model ablation, remove one word in one position, get the output hidden/cell state , compare with original, write in an array at position index
def model_abl_ppl(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.                                                                                       
    diff_arr = np.zeros(71); diff_arr_fl = np.zeros(71); diff_arr_sl = np.zeros(71); diff_arr_tl = np.zeros(71)
    cell_state_fl = []; cell_state_sl = []; cell_state_tl = []; hidden_store = []
    dist_arr = np.zeros(args.bptt); count = 0
    entropy_batch_vec =None;target_batch_vec = []
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    #hidden = model.init_hidden(batch_size)                                                                                                 
    hidden = model.init_hidden(args.bptt+1)
    entropy_2d = np.zeros((args.bptt,args.bptt+1))
    seq_len  = 70
    entropy_list = []
    for i in range(0, data_source.size(0) - 1, seq_len):
        data, targets = get_batch(data_source, i, args, seq_len=seq_len,evaluation=True)
        #print(data.shape, targets.shape)
        num_samples = data.shape[0]
        if data.shape[0] != seq_len:
            #print(data.shape[0])
            continue
        x = data.cpu().detach().numpy()
        y = np.repeat(x,num_samples+1,axis=1)
        z = np.arange(1,num_samples+1)
        data_batch_org = torch.tensor(y).cuda()

        t = targets.cpu().detach().numpy()
        a =  np.repeat(t,num_samples+1)

        targets_batch = torch.tensor(a).cuda()

        y[z-1,np.flip(z)] = UNK_ind
        
        data_batch = torch.tensor(y).cuda()
        output, h = model(data_batch, hidden)


        target_2d = a.reshape((num_samples,num_samples+1))
        
        #make it data batch not data batch org
        entropy_list.append(criterion(model.decoder.weight, model.decoder.bias, output, targets_batch).data.detach().cpu().numpy()) 
        #print(entropy_mat.shape)
        #cci.upload_npy_array('entropy_mat',entropy_mat.detach().cpu().numpy())
        
        x = len(data) * (criterion(model.decoder.weight, model.decoder.bias, output, targets_batch).data) / len(targets)
        print(x.shape)
        target_batch_vec.extend(t.tolist())
        b = x.squeeze(1).cpu().numpy().reshape((num_samples,num_samples+1))
        if i==0: 
            entropy_batch_vec = b
        else: 
            entropy_batch_vec = np.concatenate((entropy_batch_vec, b), axis=0)
        
        o, hidden = model(data_batch_org, hidden)
        hidden = repackage_hidden(hidden)
    cci.upload_npy_array('entropy_mat',np.array(entropy_list))
    exit(0)
    return [entropy_batch_vec/len(data_source),target_batch_vec] 

 
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
diff_arr = np.zeros(71)
diff_arr_fl = np.zeros(71)
diff_arr_sl = np.zeros(71)
diff_arr_tl = np.zeros(71)
l2norm = nn.PairwiseDistance(p=2)


def model_ablation(data_source, batch_size=10,modular=False,mod=0.5):
    # Turn on evaluation mode which disables dropout.                                                                                      
    seq_len = 200 #args.bptt  
    diff_arr = np.zeros(seq_len+1); diff_arr_fl = np.zeros(seq_len+1); diff_arr_sl = np.zeros(seq_len+1); diff_arr_tl = np.zeros(seq_len+1)
    cell_state_fl = []; cell_state_sl = []; cell_state_tl = []; hidden_store = []; 
    diff_arr_sl_first = np.zeros(seq_len+1); diff_arr_sl_second = np.zeros(seq_len+1);
    dist_arr = np.zeros(args.bptt); count = 0

    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    #hidden = model.init_hidden(batch_size)
    hidden = model.init_hidden(seq_len+1)   
    
    for i in range(0, data_source.size(0) - 1, seq_len): #args.bptt
        data, targets = get_batch(data_source, i, args, seq_len=seq_len,evaluation=True)
        
        if data.shape[0] != seq_len:
            print(data.shape[0])
            continue 

        x = data.cpu().detach().numpy()
        y = np.repeat(x,seq_len+1,axis=1)
        z = np.arange(1,seq_len+1)
        data_batch_org = torch.tensor(y).cuda()

        y[z-1,np.flip(z)] = UNK_ind
        data_batch = torch.tensor(y).cuda()

        output, h = model(data_batch, hidden)        
        o, hidden = model(data_batch_org, hidden)
        ## 0 is for hidden states and 1 is for cell states
        first_layer_cell =  torch.squeeze(h[0][1],dim=0)
        second_layer_cell=  torch.squeeze(h[1][1],dim=0)
        third_layer_cell =  torch.squeeze(h[2][1],dim=0)
        
        gt_cs_fl = first_layer_cell[0,:].unsqueeze_(0)
        gt_cs_sl = second_layer_cell[0,:].unsqueeze_(0)
        gt_cs_tl = third_layer_cell[0,:].unsqueeze_(0)
        
        diff_fl = l2norm(first_layer_cell,  gt_cs_fl) / torch.norm(gt_cs_fl).item()
        #print(first_layer_cell.shape, diff_fl.shape, gt_cs_fl.shape, torch.norm(gt_cs_fl).item())
        

        if modular:
            length = second_layer_cell.shape[1]
            diff_sl_first=l2norm(second_layer_cell[:,:int(mod*length)], gt_cs_sl[:,:int(mod*length)])/torch.norm(gt_cs_sl[:,:int(mod*length)]).item()
            diff_sl_second=l2norm(second_layer_cell[:,int(mod*length):], gt_cs_sl[:,int(mod*length):])/torch.norm(gt_cs_sl[:,int(mod*length):]).item()
            print('all should be 0',diff_sl_first[0],diff_sl_second[0])
        else:
            diff_sl = l2norm(second_layer_cell, gt_cs_sl) / torch.norm(gt_cs_sl).item()

        diff_tl = l2norm(third_layer_cell,  gt_cs_tl) / torch.norm(gt_cs_tl).item()
        
        diff_arr_fl += diff_fl.cpu().detach().numpy() 
        if modular:
            diff_arr_sl_second += diff_sl_second.cpu().detach().numpy()
            diff_arr_sl_first += diff_sl_first.cpu().detach().numpy()
        else:
            diff_arr_sl += diff_sl.cpu().detach().numpy()
        diff_arr_tl += diff_tl.cpu().detach().numpy()
        
        hidden = repackage_hidden(hidden)        
        count +=1

    diff_arr_fl /= float(count); diff_arr_sl /= float(count); diff_arr_tl /= float(count);
    diff_arr_sl_second /= float(count); diff_arr_sl_first /= float(count); 
    
    if modular: 
        return [diff_arr_fl[1:],[diff_arr_sl_first[1:],diff_arr_sl_second[1:]],diff_arr_tl[1:]]
    else:
        return [diff_arr_fl[1:],diff_arr_sl[1:],diff_arr_tl[1:]] #total_loss.item() / len(data_source)    


from scipy.optimize import curve_fit
def f(x, A, B): # this is your 'straight line' y=f(x)
    return A*x + B

def expf(x, a, b, c):
    return a * np.exp(-b * x) + c

import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

import random 
import statistics
import matplotlib.mlab as mlab
from scipy.stats import norm
def high_low_prefo():
    #model_name = ['PTB_1000_epochs.pt', 'PTB_custom_5.pt', 'PTB_custom_10.pt','PTB_custom_20.pt', 'PTB_custom_30.pt','PTB_custom_70.pt', 'PTB_custom_grad20_fix70.pt','PTB_modular_80_20.pt','PTB_l1_3_4_l2_pareto.pt','PTB_l1_3_4_l2_pareto_44.pt','PTB_l1_3_4_l2_pareto_64.pt']
    model_name = ['PTB_1000_epochs.pt','PTB_modular_80_20.pt','PTB_l1_3_4_l2_pareto.pt']
    k = {}
    for i in range(len(model_name)):
        try: cci.download_to_file('shivangi/'+ model_name[i], model_name[i])
        except: print('Not in file format', model_name[i])
        model_load(model_name[i])
        ### REMEMBER TO DIVIDE BY TARGETS COUNT
        
        entropy_loss_vec_name = 'entropy_loss_'+model_name[i].split('.')[0]
        target_vec_name = 'target_vec_'+model_name[i].split('.')[0]
        try:
            entropy_loss = cci.download_npy_array(entropy_loss_vec_name)
            target_vec   = cci.download_npy_array(target_vec_name)
            l = len(target_vec)
            print('Downloaded from CC')
        except:
            print('Running ablation and saving array')
            entropy_loss, l, target_vec = eval_hig_low_fre(test_data, test_batch_size)
            cci.upload_npy_array(entropy_loss_vec_name,entropy_loss)
            cci.upload_npy_array(target_vec_name,target_vec)
        
        #Calculating test loss
        test_loss = sum(entropy_loss) / l
        
        print('Model ',model_name[i])
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
            test_loss, math.exp(test_loss), test_loss / math.log(2)))
        print('=' * 89)
        
        #Create an array of 10000x824
        start_ind_arr = list(range(0,l,100))[:-1]
        bootstrap_mat_name = 'shivangi/'+'Bootstrap_samples'
        try:
            Boot_matrix = cci.download_npy_array(bootstrap_mat_name) 
        except:
            Boot_matrix = [random.choices(start_ind_arr,k=len(start_ind_arr)) for  _ in range(10000)]
            Boot_matrix.append(start_ind_arr)
            cci.upload_npy_array(bootstrap_mat_name,Boot_matrix)

        print('Shape of Bootstrap sample matrix', len(Boot_matrix),' x ',len(Boot_matrix[0]))    
        
        h = [ math.exp(sum([sum(entropy_loss[s:100+s]) for s in row]) / (100*(len(start_ind_arr))) ) for row in Boot_matrix ]
        print('Last ppl is ',h[-1])
        print('Mean ppl is ',round(statistics.mean(h),2),' with std dev of ',round(statistics.stdev(h),2))
        k[i] = h
        #plot hist of h
        plt.figure(); plt.hist(h,bins=150)
        plt.xlabel('PPL Bins')
        plt.ylabel('Sample counts')
        plt.savefig('Bootstrap_perfor_hist_'+model_name[i].split('.')[0])
                
            
        if False:
            #calculating loss for high and low freq words
            high_freq_loss = 0
            for target_ind in range(len(target_vec)):
                if target_vec[target_ind] in high_freq:
                    high_freq_loss += entropy_loss[target_ind]
        
            print('High freq word loss',math.exp( (high_freq_loss)/l) )
            print('Low freq word loss', math.exp( (sum(entropy_loss)- high_freq_loss)/l) )
    
            
    diff1 = np.array(k[1]) - np.array(k[0])
    diff2 = np.array(k[2]) - np.array(k[0])
    
    
    plt.figure()                                                                                                                            
    n1,bin1,p = plt.hist(diff1,color='k',bins=150,label='Mod8-Orig')                                                                      
           
    plt.legend(loc='upper right')
    plt.xlabel('PPL difference btw Mod8 and Orig model'); plt.ylabel('Sample count')
    plt.savefig('Bootstrap_comp_diff_histMod8')

    
    plt.figure()
    n2,bin2,p= plt.hist(diff2,color='m',bins=150,label='Pareto-Orig')                                                                      
           
    plt.legend(loc='upper right')                                                                                                           
    plt.xlabel('PPL difference btw Pareto and Orig model'); plt.ylabel('Sample count')                                                      
    plt.savefig('Bootstrap_comp_diff_histPareto')   
    
    (mu1, sigma1) = norm.fit(diff1)                                                                                                         
    (mu2, sigma2) = norm.fit(diff2) 
    
    y1 = mlab.normpdf( bin1, mu1, sigma1)                                                                                                  
    y2 = mlab.normpdf( bin2, mu2, sigma2)  
    
    plt.figure()                                                                                                                            
    plt.plot(bin1, y1, 'k--', linewidth=2,label='Mod80-Baseline')                                                                           
    plt.plot(bin2, y2, 'm--', linewidth=2,label='Pareto-Baseline')                                                                          
    plt.xlabel('PPL difference')
    plt.legend(loc='upper right')                                                                                                           
    plt.savefig('Bootstrap_comp_diff_norm_hist')                                                                                            
    

    
    (mu1, sigma1) = norm.fit(k[0])
    (mu2, sigma2) = norm.fit(k[1])
    (mu3, sigma3) = norm.fit(k[2])

    plt.figure(); 
    n1, bins1, patches = plt.hist(k[0],color='k',bins=150,label='Baseline')
    n2, bins2, patches = plt.hist(k[1],color='m',bins=150,label='Mod80')
    n3, bins3, patches = plt.hist(k[2],color='g',bins=150,label='Pareto')

    y1 = mlab.normpdf( bins1, mu1, sigma1)
    y2 = mlab.normpdf( bins2, mu2, sigma2)
    y3 = mlab.normpdf( bins3, mu3, sigma3)
        
        
    plt.xlabel('PPL Bins')
    plt.legend(loc='upper right')
    plt.ylabel('Sample counts')
    plt.savefig('Bootstrap_comp_hist')
    
    plt.figure()
    plt.plot(bins1, y1, 'k--', linewidth=2,label='Baseline')
    plt.plot(bins2, y2, 'm--', linewidth=2,label='Mod80')
    plt.plot(bins3, y3, 'g--', linewidth=2,label='Pareto')
    plt.xlabel('Perplexity')
    plt.legend(loc='upper right')
    plt.savefig('Bootstrap_comp_norm_hist')
    """
    plt.figure()
    plt.plot(k[0],'k--',label='Orig_1k')
    plt.plot(k[1],'m--',label='Mod8')
    plt.plot(k[2],'g--',label='Pareto')
        
    plt.legend(loc='upper right')
    plt.xlabel('Samples'); plt.ylabel('Perplexity')
    plt.savefig('Bootstrap_comp_data')
    """

high_low_prefo()
def abl_perfo():
    
    model_name = [ 'PTB_1000_epochs.pt']#, 'PTB_custom_5.pt', 'PTB_custom_10.pt','PTB_custom_20.pt', 'PTB_custom_30.pt','PTB_custom_70.pt', 'PTB_custom_grad20_fix70.pt','PTB_modular_80_20.pt'] 

    piclab     = ['Baseline','T5','T10','T20','T30','T70','Mod50','Mod80']  
    fig1, axs1 = plt.subplots(1, 1, figsize=(10, 4), sharey=True)
    fig2, axs2 = plt.subplots(1, 1, figsize=(10, 4), sharey=True)
    fig3, axs3 = plt.subplots(1, 1, figsize=(10, 4), sharey=True)

    for i in range(len(model_name)):
        try: cci.download_to_file('shivangi/'+ model_name[i], model_name[i])
        except: print('Not in file format', model_name[i])
        print(model_name[i])
        model_load(model_name[i])
        
        entropy_loss, target_vec= model_abl_ppl(test_data, test_batch_size)
        all_ppl  = np.exp(np.sum(entropy_loss,axis=0))
        
        
        all_ppl_1 = all_ppl - all_ppl[0]
        ax = axs1
        print('Overall ppl',all_ppl)
        ax.plot(all_ppl_1[1:], label = piclab[i]);
        ax.set_ylabel('PPL difference'); ax.set_xlabel('Ablated word pos')
        ax.set_title('Ablation')
        ax.legend(loc='upper right')
        
        high_freq_loss = np.zeros(71)
        for target_ind in range(len(target_vec)):
            if target_vec[target_ind] in high_freq:
                high_freq_loss += entropy_loss[target_ind,:]

        high_freq_ppl = np.exp(high_freq_loss)

        high_freq_ppl_1 = high_freq_ppl - high_freq_ppl[0]
        print('oth element hf',high_freq_ppl_1[0])
        ax = axs2
        ax.plot(high_freq_ppl_1[1:], label = piclab[i]);
        ax.set_ylabel('PPL difference for high freq words'); ax.set_xlabel('Ablated word pos')
        ax.set_title('Ablation')
        ax.legend(loc='upper right')
        
        print('High freq word loss',high_freq_ppl)
        
        #print('Low freq word loss', low_freq_ppl)
        low_freq_loss = np.sum(entropy_loss,axis=0) - high_freq_loss
        low_freq_ppl = np.exp(low_freq_loss)
        #print('Low freq word loss', low_freq_ppl)
        low_freq_ppl_1 = low_freq_ppl - low_freq_ppl[0]
        print('PPL element lf',low_freq_ppl)
        
        ax = axs3
        ax.plot(low_freq_ppl_1[1:], label = piclab[i]);
        ax.set_ylabel('PPL difference for low freq words'); ax.set_xlabel('Ablated word pos')
        ax.set_title('Ablation')
        ax.legend(loc='upper right')

    
    fig1.savefig('PPL_diff_plot.png')
    fig2.savefig('PPL_high_f_diff_plot.png')
    fig3.savefig('PPL_low_f_diff_plot.png')
        
#abl_perfo()
def plotting_model_ablation():
    ## Plotting cosine distance between hidden/cell state for fixed models
    #model_name =  ['PTB_asgd_orig.pt', 'PTB_fixed_tmax_20_nograd.pt', 'PTB_fixed_tmax_70_nograd.pt','PTB_fixed_tmax_200_nograd.pt']
    #piclab = ['orig','f20con','f70con','f200con']
    #T = [10,5,10,20,30,70]
    #T = [10,10,10,10]
    #model_name=['PTB_asgd_orig.pt','PTB_fixed_tmax_20_nograd.pt','PTB_fixed_tmax_20_rev_nograd.pt','PTB_fixed_tmax_20_midlong_nograd.pt']
    #piclab = ['orig','f20','f20rev','f20midlong']
    T = [10,5,10,20,30,70,30,30]
    #model_name = ['PTB_1000_epochs.pt', 'PTB_em_200_1k','PTB_em_800_1k','WT2.pt']
    model_name=['PTB_1000_epochs.pt','PTB_custom_5.pt', 'PTB_custom_10.pt','PTB_custom_20.pt','PTB_custom_30.pt','PTB_custom_70.pt','PTB_custom_grad20_fix70.pt','PTB_modular_80_20.pt']
    #model_name=['PTB_1000_epochs.pt','PTB_custom_grad20_fix70.pt','PTB_modular_80_20.pt']
    #piclab = ['400','200','800','Wiki2']
    piclab     = ['orig_1k','T5','T10','T20','T30','T70','mod5','mod8']
    #piclab     = ['orig_1k','mod5','mod8']

    #model_name =  ['PTB_asgd_orig.pt']#, 'PTB_grad_tmax_20_nograd.pt' , 'PTB_grad_tmax_70_nograd.pt' , 'PTB_grad_tmax_20_nograd_incon.pt']
    #piclab = ['orig','g20con','g70con','g20div_con']

    #model_name = ['PTB_asgd_orig.pt', 'PTB_asgd_grad_tmax_20.pt', 'PTB_asgd_grad_tmax_70.pt', 'PTB_asgd_grad_tmax_200.pt' ]     
    #piclab = ['orig','g20con','g70con','g20div_con']

    fig1, axs1 = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    fig2, axs2 = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    fig3, axs3 = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

    for i in range(len(model_name)):
        try: cci.download_to_file('shivangi/'+ model_name[i], model_name[i])
        except: print('Not in file format', model_name[i])
        print(model_name[i])
        model_load(model_name[i])
        if i == -1:
            modular=True;mod=0.5
        elif i == -2:
            modular=True;mod=0.8
        else:
            modular=False;mod=0.5

        diff_arr_all = model_ablation(test_data, test_batch_size,modular=modular,mod=mod)
       
        for j in range(len(diff_arr_all)):
            
            diff_arr = diff_arr_all[j]

            ax = axs1[j]
            if i in [-1,-2] and j==1: #real values [1,2]
                ax.plot(np.arange(1,len(diff_arr)+1), np.log(diff_arr[1]), label = piclab[i]+'_2');
                ax.plot(np.arange(1,len(diff_arr)+1), np.log(diff_arr[0]), label = piclab[i]+'_1');
            else:
                ax.plot(np.arange(1,len(diff_arr)+1), np.log(diff_arr), label = piclab[i]); 

            ax.set_ylabel('Log L2 norm btw cell states'); ax.set_xlabel('Ablated word pos')    
            ax.set_title('For LSTM layer '+str(j+1))
            ax.legend(loc='upper right')
            
            x = np.arange(20,70)
            y = np.log(diff_arr[x])
            popt, pcov = curve_fit(f, x, y)
            print('The Test of semi-log for 20-70',1/(1-np.exp(popt[0])))

            """
            x1 = np.arange(20)
            y1 = np.log(diff_arr[x1])
      
            
            ax = axs2[j];
            #ax.plot(np.log(diff_arr),  label = piclab[i]);
            #ax.plot(x, popt[0]*x + popt[1],'--')
            ax.plot(y1, label = piclab[i])
            sliding_linear = True; k = 0; window = 3; slide = 1;Test_1_20=[]
            if sliding_linear:
                x_1_20 = np.arange(20)
                y_1_20 = np.log(diff_arr[x_1_20])
                while k+window < 20:
                    x_win = x_1_20[k:k+window]
                    y_win = y_1_20[k:k+window]
                    print(x_win)
                    k+=slide
                    popt_win, pcov_win = curve_fit(f, x_win, y_win)
                    Test_1_20.append(1/(1-np.exp(popt_win[0])))                    
                    ax.plot(x_win, popt_win[0]*x_win + popt_win[1],'--')

                ax_3 = axs3[j]
                ax_3.plot(Test_1_20,'--', label = piclab[i])
                ax_3.plot([5,10,20],[5,10,20],'*')
                ax_3.set_title('For LSTM layer '+str(j+1)+' window '+str(window))
                ax_3.set_ylabel('Slope for each word position to next 4 words'); ax.set_xlabel('Ablated word pos')
                ax_3.legend(loc='upper right')
            else: 
                popt1, pcov1 = curve_fit(expf, x1, y1)
                ax.plot(x1,expf(x1, *popt1),'--')
                           
            ax.set_ylabel('Log L2 norm btw cell states'); ax.set_xlabel('Ablated word pos')
            ax.set_title('For LSTM layer '+str(j+1))
            ax.legend(loc='upper right')
            
            print('Model name is', piclab[i],'Layer number is', j+1)
            print('The bias value is', T[i]/(2**j))
            print('Slope should be', np.log(sigmoid(np.log( (T[i]/(2**j))-1))))
            print('Slope is',popt[0], 'and intercept is',popt[1])
            print('T estimated for 20-70',1/(1-np.exp(popt[0])))
            if sliding_linear:
                print('Test for 1-20',Test_1_20)
                
            else:
                print('exp function hyperparamter',popt1)
            """
        if (i==len(model_name)-1):
            for ax in axs1.flat:
                ax.label_outer()
            for ax in axs2.flat:
                ax.label_outer()
            for ax in axs3.flat:
                ax.label_outer()
            
            fig1.savefig('L2_cell_norm_1210_200.png')
            #fig2.savefig('L2_cell_log_emb_size.png')
            #fig3.savefig('Slope_1_20_'+str(window)+'.png')

#plotting_model_ablation()
## Plotting forget bias values initial  vs final for Chrono-grad
def plotting_grad_bias():
    model_name = ['PTB_asgd_grad_tmax_20.pt' , 'PTB_asgd_grad_tmax_70.pt',  'PTB_asgd_grad_tmax_200.pt'  ]
    piclab = ['g20','g70','g200']
    Tmin = 1; Tmax_l = [20, 70, 200]
    hid_dim = [1150,1150,400]
   
    for i in range(3):
        model_load(model_name[i])
        Tmax = Tmax_l[i]
        dict_param = dict(model.named_parameters())
     
        for l in range(3):
            diff = (Tmax - 1 - Tmin)
            forget_gate_bias_bfore_training = np.log(Tmin + (  (diff*np.arange(hid_dim[l])) / (hid_dim[l]-1) ) )
            forget_gate_bias_after_training = (dict_param['rnns.'+str(l)+'.module.bias_ih_l0'].data[hid_dim[l]:hid_dim[l]*2].cpu() + dict_param['rnns.'+str(l)+'.module.bias_hh_l0'].data[hid_dim[l]:hid_dim[l]*2].cpu())

            plt.figure(l+1); plt.plot(forget_gate_bias_bfore_training,forget_gate_bias_after_training,label='g'+str(Tmax))
            plt.axis([0, 6 , -5 ,5])
            plt.xlabel('Forget gate bias initialization'); plt.ylabel('Forget gate bias after training')
            plt.title('Bias values for layer '+str(l+1))

    for l in range(3):
        plt.figure(l+1);plt.legend(loc='upper left'); 
        plt.savefig('Grad_model_forget_bias_hidden_'+str(l+1)+'.png') 
    
#plotting_grad_bias()
cci_e2e = cc.get_interface('endtoend',verbose=False,ACCESS_KEY=access_key, SECRET_KEY=secret_key,endpoint_url=endpoint_url)
cci_data = cc.get_interface('neural-model-data',verbose=False,ACCESS_KEY=access_key, SECRET_KEY=secret_key,endpoint_url=endpoint_url)
cci_lm = cc.get_interface('lm-encoding',verbose=False)

def story_feat_extract():
    model_name = ['PTB_1000_epochs.pt','PTB_modular_80_20.pt','PTB_l1_3_4_l2_pareto.pt']
    allstories = [story.replace('story_arrays/', '') for story in cci_e2e.lsdir('story_arrays')]
    for m in range(len(model_name)):
        try: cci.download_to_file('shivangi/'+ model_name[m], model_name[m])
        except: print('Not in file format', model_name[m])
        model_load(model_name[m])
        
        for story in allstories:
            base = 'lstm/%s' %(model_name[m].split('.')[0])
            data_matrix = cci_data.download_npy_array('stimulus_seqs71_shivangi/%s'%story)
            feat = eval_story(data_matrix)
            for i in range(len(feat)):
                layer = i+1

                print('%s/vectors/layer%d/%s' % (base, layer, story))
                cci_lm.upload_raw_array('%s/vectors/layer%d/%s' % (base, layer, story), feat[i])
               
                print('Layer', layer, ': ', feat[i].shape)
                for cl in [2, 4, 8, 16, 32, 64]:
                    f2 = feat[i][:, 0] if not cl else np.vstack([feat[i][0, :cl], feat[i][:-cl, cl]])
                    print('Inside opath', '%s/hidden_state/layer%d/cl%d/%s' % (base, layer, cl, story))
                    cci_lm.upload_raw_array('%s/hidden_state/layer%d/cl%d/%s' % (base, layer, cl, story), f2)
            
#story_feat_extract()



def plotting_fix_bias():
    ## Plotting cosine distance between hidden/cell state for fixed models                                                                  
    #model_name =  ['PTB_asgd_orig.pt', 'PTB_fixed_tmax_20_nograd.pt', 'PTB_fixed_tmax_70_nograd.pt','PTB_fixed_tmax_200_nograd.pt']

    #model_name = ['PTB_asgd_orig.pt' ,'PTB_asgd_fixed_tmax_20.pt', 'PTB_asgd_fixed_tmax_70.pt', 'PTB_asgd_fixed_tmax_200.pt' ]
    #piclab = ['orig','f20','f70','f200']
    #piclab_in = ['orig_init','f20_init','f70_init','f200_init']
    
    #model_name = ['PTB_1000_epochs.pt','PTB_custom_20.pt', 'PTB_custom_70.pt','PTB_custom_grad20_fix70.pt']
    #piclab     = ['orig','T20','T70','modular']

    model_name = ['PTB_1000_epochs.pt','PTB_custom_5.pt', 'PTB_custom_10.pt','PTB_custom_20.pt','PTB_custom_30.pt','PTB_custom_70.pt']
    piclab     = ['orig_1k','T5','T10','T20','T30','T70']


    hid_dim = [1150,1150,400]
    #Tmax_l = [10, 20,70,200]

    for i in range(len(model_name)):
        model_load(model_name[i])
        #fixed_bias  = np.log([Tmax_l[i]-1,Tmax_l[i]/2-1,Tmax_l[i]/4-1])
        dict_param  = dict(model.named_parameters())
        
        for l in range(2,3):
            #forget_gate_bias_bfore_training =  fixed_bias[l]*np.ones(hid_dim[l]) 
            forget_gate_bias_after_training = (dict_param['rnns.'+str(l)+'.module.bias_ih_l0'].data[hid_dim[l]:hid_dim[l]*2].cpu() + dict_param['rnns.'+str(l)+'.module.bias_hh_l0'].data[hid_dim[l]:hid_dim[l]*2].cpu())
            #forget_gate_bias_sorted = np.cumsum(np.sort(forget_gate_bias_after_training))
        
            plt.figure(l+1); 
            plt.plot(np.sort(forget_gate_bias_after_training),label=piclab[i])
            #plt.plot(forget_gate_bias_bfore_training,label=piclab_in[i])
            plt.xlabel('Hidden units'); plt.ylabel('Forget gate bias (sorted)')
            plt.title('Forget gate bias after training for layer '+str(l+1))

    for l in range(2,3):
        plt.figure(l+1);plt.legend(loc='upper left');
        plt.savefig('Custom_fg_bias_fixed_'+str(l+1)+'.png')
        #plt.savefig('Fixed_model_nograd_bias_bf_af_'+str(l+1)+'.png')

#plotting_fix_bias()


#1. Distribution plot: hidden units vs. old/new value
#2. Cell state visualization as a heat map (for units vs. ablated positions) (seaboard - look at Shailee’s notebook)
#3. Distance vs. position plot: ylim same across layers - get code from Shailee for subplots!
