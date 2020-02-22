import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

from scipy import signal as signal
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
cci = cc.get_interface('lstm-timescales', ACCESS_KEY=access_key, SECRET_KEY=secret_key,endpoint_url=endpoint_url)

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
    global model, criterion, optimizer
    try:
        with open(fn, 'rb') as f:
            model, criterion, optimizer = torch.load(f)
    except:
        print('Downloading from pickle')
        model, criterion, optimizer = cci.download_pickle('shivangi/'+fn)

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

####Save word2idx dictionary 
if False:
    word2idx_dict_name = 'shivangi/'+'word2dict_'+ args.data
    word2idx_dict = corpus.dictionary.word2idx
    cci.upload_npy_array(word2idx_dict_name,list(word2idx_dict))

    
    #cross_check
    word2dict_penn = cci.download_npy_array(word2idx_dict_name)
    h = {word2dict_penn[i]:i for i in range(len(word2dict_penn))}
    for word in h:
        if word2idx_dict[word] != h[word]:
            print('Worng index')

    print(word2idx_dict)


#######                                                                                                                                    
#1.                                                                                                                                        
vocab_dict = corpus.dictionary.counter
#print(sorted(vocab_dict.keys(),key=vocab_dict.get,reverse=True))
#print(vocab_dict)
total_count = sum(vocab_dict.values())
count=0
high_freq=set()
print( total_count/2)
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

from splitcross import SplitCrossEntropyLoss
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
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
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

def eval_hig_low_fre(data_source, batch_size=10):
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
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)

def train():
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
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
    print('=' * 89)                                                                                                                         
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(                                             
    test_loss, math.exp(test_loss), test_loss / math.log(2)))                                                                          
    print('=' * 89)                                                                                                                         
    

    dict_param = dict(model.named_parameters())                                                                                             
    hid_dim = [1150,1150,400]                                                                                                               
    

    for l in range(3):                                                                                                                      
        x = (dict_param['rnns.'+str(l)+'.module.bias_ih_l0'].data[0:hid_dim[l]*2].cpu() + dict_param['rnns.'+str(l)+'.module.bias_hh_l0'].data[0:hid_dim[l]*2].cpu())  
        #-----------------------------------------------------------------------------------------------------------------
        #print('Forget gate initialization values for layer',l,'is', math.log( (Tmax/(2**(2-l)))-1))
        print('Expected random values')
        print('Input gates bias after training',x[0:hid_dim[l]])
        
        first_mod = int(0.8*hid_dim[l]); second_mod = int(0.1*hid_dim[l])+first_mod; third_mod = int(0.1*hid_dim[l])+second_mod;
        print('Expected values from ',np.log(1),'to',np.log(19))
        print('Forget gates bias after training',x[hid_dim[l]:hid_dim[l]+first_mod])

        print('Expected values',np.log(69))
        print('Forget gates bias after training',x[hid_dim[l]+first_mod:hid_dim[l]+second_mod])
        
        print('Expected values',10**8)
        print('Forget gates bias after training',x[hid_dim[l]+second_mod:hid_dim[l]+third_mod])
    
        #------------------------------------------------------------------------------------------------------------------

        plt.figure(); plt.plot(np.sort(x[hid_dim[l]:hid_dim[l]*2]));  
        plt.xlabel('Hidden units in sorted order'); plt.ylabel('Forget gate bias values of hidden unit')                               
        plt.savefig('Custom_fg_bias_modular_layer_'+str(l+1)+str(Tmax)+'.png')
        #------------------------------------------------------------------------------------------------------------------

#exit(0)

#Perform model ablation, remove one word in one position, get the output hidden/cell state , compare with original, write in an array at position index
 
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
diff_arr = np.zeros(71)
diff_arr_fl = np.zeros(71)
diff_arr_sl = np.zeros(71)
diff_arr_tl = np.zeros(71)
l2norm = nn.PairwiseDistance(p=2)

def model_ablation(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.                                                                                      
    diff_arr = np.zeros(71); diff_arr_fl = np.zeros(71); diff_arr_sl = np.zeros(71); diff_arr_tl = np.zeros(71)
    cell_state_fl = []; cell_state_sl = []; cell_state_tl = []; hidden_store = []
    dist_arr = np.zeros(args.bptt); count = 0

    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.bptt+1)   
    
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        if data.shape[0] != args.bptt:
            print(data.shape[0])
            continue 
        x = data.cpu().detach().numpy()
        y = np.repeat(x,args.bptt+1,axis=1)
        z = np.arange(1,71)
        data_batch_org = torch.tensor(y).cuda()
        
        y[z-1,np.flip(z)] = UNK_ind
        data_batch = torch.tensor(y).cuda()
        
        #print((data_batch))
        #exit(0)
        output, h = model(data_batch, hidden)
        o, hidden = model(data_batch_org, hidden)

        first_layer_cell =  torch.squeeze(h[0][1],dim=0)
        second_layer_cell=  torch.squeeze(h[1][1],dim=0)
        third_layer_cell =  torch.squeeze(h[2][1],dim=0)
        
        gt_cs_fl = first_layer_cell[0,:].unsqueeze_(0)
        gt_cs_sl = second_layer_cell[0,:].unsqueeze_(0)
        gt_cs_tl = third_layer_cell[0,:].unsqueeze_(0)
        
        diff_fl = l2norm(first_layer_cell,  gt_cs_fl) /torch.norm(gt_cs_fl).item()

        if True:
            diff_sl = l2norm(second_layer_cell, gt_cs_sl) / torch.norm(gt_cs_sl).item()
        else: 
            first_mod  = int(0.8*hid_dim) 
            diff_sl = l2norm(second_layer_cell[:first_mod], gt_cs_sl[:first_mod]) / l2norm(gt_cs_sl[:first_mod]) 
            diff_sl = l2norm(second_layer_cell[first_mod:], gt_cs_sl[first_mod:]) / l2norm( gt_cs_sl[first_mod:])

        #check whether normalized or not
        diff_tl = l2norm(third_layer_cell,  gt_cs_tl) / torch.norm(gt_cs_tl).item() 
        
        diff_arr_fl += diff_fl.cpu().detach().numpy() 
        diff_arr_sl += diff_sl.cpu().detach().numpy()
        diff_arr_tl += diff_tl.cpu().detach().numpy()
        
    
        hidden = repackage_hidden(hidden)        
        count +=1

    diff_arr_fl /= float(count); diff_arr_sl /= float(count); diff_arr_tl /= float(count);
    
    return [diff_arr_fl[1:],diff_arr_sl[1:],diff_arr_tl[1:] ] #total_loss.item() / len(data_source)                                       


from scipy.optimize import curve_fit
def linf(x, A, B, C): # this is your 'straight line' y=f(x) for straight line make B = 1 
    return A*(x**B) + C

def expf(x, a, b, c):
    return a * np.exp(-b * x) + c

def explinf(x, a, b, c,d):
    return a * np.exp(-b * x) + c*(x**(d)) 

from lmfit import Model
gmodel    =  Model(explinf)
expmodel  =  Model(expf)
linmodel  =  Model(linf)

import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
import statistics
from sklearn.metrics import mean_squared_error
def plotting_model_ablation():
    
    model_name = ['PTB_1000_epochs.pt', 'PTB_l1_3_4_l2_pareto.pt','PTB_modular_80_20.pt'] #'PTB_custom_grad20_fix70.pt'
    piclab     = ['Baseline','Pareto','Mod80']  
    
    fig3, axs3 = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
            
    for i in range(len(model_name)):
        ##load model
        try: cci.download_to_file('shivangi/'+ model_name[i], model_name[i])
        except: print('Not in file format', model_name[i])
        model_load(model_name[i])

        ###################################################################################
        #print model performance
        if False:
            print('Model is',model_name[i])
            test_loss = evaluate(test_data, test_batch_size)
            print('=' * 89)
            print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
                test_loss, math.exp(test_loss), test_loss / math.log(2)))
            print('=' * 89)
    
        #add MI plot on curves
        MI_path = ('/home/shivangi/lstm-timescales/mutual_information/MI_results_PTB_01.npz')
        mi_data = np.load(MI_path, allow_pickle=True)
        mi, var_mi = mi_data['all_MI']
        shuf_mi, var_shuf_mi = mi_data['shuff_MI']
        distance = np.arange(1, 101)
        MI = mi - shuf_mi
        ##################################################################################
        #load ablation arrays 
        fig1, axs1 = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
        fig2, axs2 = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

        arr_name = 'diff_arr_all_'+model_name[i].split('.')[0]
        try: 
            diff_arr_all = cci.download_npy_array(arr_name)
            print('Downloaded from CC')
        except: 
            print('Running ablation and saving array')  
            diff_arr_all = model_ablation(test_data, test_batch_size)
            cci.upload_npy_array(arr_name,diff_arr_all)     

        print('Model is',model_name[i])
        for j in range(len(diff_arr_all)):
            print('Layer is ',j )
            diff_arr = diff_arr_all[j]
            ax1 = axs1[j]; ax2 = axs2[j] ; ax3 = axs3[j] ;
        
            semi_log = True; log_log=True
            if semi_log: 
                ax3.plot((np.arange(1,len(diff_arr)+1)),(diff_arr), label = piclab[i]);
                ax3.set_ylabel('L2 norm btw cell states'); ax3.set_xlabel('Ablated word pos')
                ax3.set_title('For LSTM layer '+str(j+1))
                ax3.legend(loc='upper right')
                ax3.set_yscale('log');
            if log_log:
                ax1.plot((np.arange(1,len(diff_arr)+1)),(diff_arr), label = piclab[i]);
                ax1.plot((np.arange(1,len(diff_arr)+1)),(MI[:len(diff_arr)]), label = 'MI_PTB');
                ax1.set_ylabel('L2 norm btw cell states'); ax1.set_xlabel('Ablated word pos')
                #ax1.set_yscale('log'); ax1.set_xscale('log')
            

            #Plot slope of curve fit to log-log graph
            x,y = (np.arange(1,len(diff_arr)+1)), (diff_arr)                                                                                
            
            #coeff of exp and poly are init to constant
            a,c = 0.1,1
            # grid search values for b and d
            bd_list = [(b,d) for b in np.arange(0.1,1.5,0.1) for d in np.arange(-1,1,0.1)]
            h = defaultdict(dict)
            mse = []
            combined_model = True
            separate = False
            if combined_model:
                for count, (b,d) in enumerate(bd_list):
                    params = gmodel.make_params()
                    params.add('a',value=a, min=0)
                    params.add('c',value=c, min=0)
                    params.add('b',value=b)
                    params.add('d',value=d)

                    try:
                        result = gmodel.fit(y, params, x=x)                
                        [a,b,c,d] = [result.params['a'].value, result.params['b'].value, result.params['c'].value, result.params['d'].value]
                        if j not in h[i]: h[i][j] = {}
                        
                        mse.append(mean_squared_error(y, explinf(x, a,b,c,d)))
    
                        h[i][j][count]  = [(round(a,2),round(b,2),round(c,2),round(d,2))]
                        
                    except:
                        continue
                    
                #retreive opt a,b,c,d by taking arg min ove mse lsit
                (a_opt,b_opt,c_opt,d_opt) = h[i][j][np.argmin(np.array(mse))][0]
                ax1.plot(x,explinf(x, a_opt,b_opt,c_opt,d_opt),'m--',label = 'both')
                ax1.plot(x,explinf(x, a_opt,b_opt,0,0),'k--',label = 'expo only') 
                ax1.plot(x,explinf(x, 0, 0, c_opt,d_opt),'r--',label = 'poly only')
                ax1.set_ylim([0.01 , 1])
                ax2.plot(mse)
            
            elif separate: 
                paramse = expmodel.make_params( a=a, b=b, c=0.25)
                resulte = expmodel.fit(y, paramse,x=x)
                [a,b,c] = [resulte.params['a'].value, resulte.params['b'].value, resulte.params['c'].value]
                print('Coeffs of exp',a,b,c)
                ax1.plot(x,expf(x, a,b,c),'k--',label = 'expo')
                
                
                params = linmodel.make_params( A=c, B=d, C=0.25)
                result = linmodel.fit(y, params, x=x)
                [A,B,C] = [result.params['A'].value, result.params['B'].value, result.params['C'].value]
                print('Coeffs of lin',A,B,C)
                ax1.plot(x,linf(x,A,B,C),'r--',label = 'poly')


            ax1.set_title('For LSTM layer '+str(j+1))
            ax1.legend(loc='lower left')
            ax1.set_yscale('log'); ax1.set_xscale('log') 

        #if (j==len(model_name)-1):
        for ax in axs1.flat: 
            ax.label_outer()
        for ax in axs2.flat:
            ax.label_outer()
                
            
        fig1.savefig('Log_diff_log_x_0216_combined_'+model_name[i].split('.')[0]+'_'+str(round(b,2))+'_'+str(round(d,2))+'.png')
        fig2.savefig('MSE_0207_'+model_name[i].split('.')[0])
    
    for ax in axs3.flat:
        ax.label_outer()
    fig3.savefig('Log_diff_x_0216_top_3.png')

plotting_model_ablation()

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

def plotting_fix_bias():
    ## Plotting cosine distance between hidden/cell state for fixed models                                                                  
    #model_name =  ['PTB_asgd_orig.pt', 'PTB_fixed_tmax_20_nograd.pt', 'PTB_fixed_tmax_70_nograd.pt','PTB_fixed_tmax_200_nograd.pt']

    #model_name = ['PTB_asgd_orig.pt' ,'PTB_asgd_fixed_tmax_20.pt', 'PTB_asgd_fixed_tmax_70.pt', 'PTB_asgd_fixed_tmax_200.pt' ]
    #piclab = ['orig','f20','f70','f200']
    #piclab_in = ['orig_init','f20_init','f70_init','f200_init']
    
    #model_name = ['PTB_1000_epochs.pt','PTB_custom_20.pt', 'PTB_custom_70.pt','PTB_custom_grad20_fix70.pt']
    #piclab     = ['orig','T20','T70','modular']

    model_name = ['PTB_1000_epochs.pt','PTB_custom_5.pt', 'PTB_custom_10.pt','PTB_custom_20.pt','PTB_custom_30.pt','PTB_custom_70.pt','PTB_custom_grad20_fix70.pt','PTB_modular_80_20.pt']
    piclab     = ['orig_1k','T5','T10','T20','T30','T70','Mod5','Mod8']


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
