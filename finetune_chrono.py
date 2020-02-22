import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn

from scipy import signal as signal
import data
import model as model
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
cci = cc.get_interface('lstm-timescales',verbose=False, ACCESS_KEY=access_key, SECRET_KEY=secret_key,endpoint_url=endpoint_url)

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
parser.add_argument('--pretrain', type=str,  default='',
                    help='path of model to resume')
##added by shivi
parser.add_argument('--Tmax', type=int,  default=20)
parser.add_argument('--Tmin', type=int,  default=1)

args = parser.parse_args()
args.tied = True

# Set the random seed manually for reproducibility.
np.random.seed(args.seed) 
#np.random.seed()   
torch.manual_seed(args.seed) # 
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else: #
        torch.cuda.manual_seed(args.seed)#

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    cci.upload_pickle('shivangi/'+fn, [model, criterion, optimizer])

def model_copy_save(fn):
    cci.upload_pickle('shivangi/'+fn, [model_copy, criterion, optimizer])

def model_load(fn):
    global model, model_copy,criterion, optimizer
    try:
        with open(fn, 'rb') as f:
            model, criterion, optimizer = torch.load(f)
    
        with open(fn, 'rb') as f1:
            model_copy, c, o = torch.load(f1)
    
    except:
        print('Downloading from pickle')
        model, criterion, optimizer = cci.download_pickle('shivangi/'+fn)
        model_copy, criterion, optimizer = cci.download_pickle('shivangi/'+fn)

    
import os
import hashlib

cci_data = cc.get_interface('neural-model-data',verbose=False,ACCESS_KEY=access_key, SECRET_KEY=secret_key,endpoint_url=endpoint_url)
penn_stim_word2index = cci_data.download_json('stimulus_ptb_stim_shivangi/word2index')

stimuli= True
if stimuli:
    ntokens = len(penn_stim_word2index)
    train_stim_array = cci_data.download_npy_array('stimulus_ptb_stim_shivangi/stimulus_train_arrays')
    val_stim_array = cci_data.download_npy_array('stimulus_ptb_stim_shivangi/stimulus_val_arrays')
    test_stim_array = cci_data.download_npy_array('stimulus_ptb_stim_shivangi/stimulus_test_arrays')
    reddit_train_data = cci_data.download_npy_array('stimulus_ptb_stim_shivangi/reddit_story_arrays/good_docs')
    
    #train_tensor = torch.tensor(np.hstack( ( np.hstack( (reddit_train_data)) , np.hstack( (train_stim_array) ))))
    reddit_data = np.hstack( (reddit_train_data))
    stim_data   = np.hstack( ( train_stim_array) )
    train_tensor = torch.tensor(np.hstack( (stim_data , reddit_data[:len(stim_data)]) ))
    test_tensor  = torch.tensor(np.hstack(test_stim_array))
    val_tensor   = torch.tensor(np.hstack(val_stim_array))
    del reddit_data
    
else:    
    fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset...')
        corpus = data.Corpus(args.data)
        torch.save(corpus, fn)
    
    train_tensor = corpus.train
    test_tensor  = corpus.test
    val_tensor   = corpus.valid
####### 

eval_batch_size = 10
test_batch_size = 1

train_data = batchify(train_tensor, args.batch_size, args)
val_data   = batchify(val_tensor, eval_batch_size, args)
test_data  = batchify(test_tensor, test_batch_size, args)


print('After batchify')
print(train_data.shape)
print(val_data.shape)
print(test_data.shape)
print(type(test_data[0][0]), (test_data[0][0]))


###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss
criterion = None
###Initialize model but we donot need it in finetune part

ntokens = len(penn_stim_word2index)

model_copy = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)# args.Tmax, args.Tmin)

for l in range(args.nlayers):
    model_copy.rnns[l]._setweights()

model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)# args.Tmax, args.Tmin)

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
        for rnn in model_copy.rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
            elif rnn.zoneout > 0: rnn.zoneout = args.wdrop


if args.pretrain:
    try:
        with open(args.pretrain, 'rb') as f:
            model_pretrained, criterion, optimizerold = torch.load(f)
    except:
        print('Downloading from pickle')
        model_pretrained, criterion, optimizerold = cci.download_pickle('shivangi/'+args.pretrain)
    
    pretrained_dict = dict(model_pretrained.named_parameters())
    model_copy_dict = dict(model_copy.named_parameters())
    model_dict = dict(model.named_parameters())

    state_dict = model_pretrained.state_dict()

    for key in model_copy_dict.keys():
        model_copy_dict[key].data[:len(pretrained_dict[key].data)] = pretrained_dict[key].data
        
    for key in model_dict.keys():
        model_dict[key].data[:len(pretrained_dict[key].data)] = pretrained_dict[key].data
    
    

###
if not criterion:
    splits = []
    if ntokens > 500000:
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
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
            val_loss_array.append(math.exp(val_loss2))
            if val_loss2 < stored_loss:
                model_copy_save(args.save)
                print('Saving Averaged!')
                stored_loss = val_loss2

            if epoch%10==0:
                 model_name = args.save.split('.')[0]
                 plt.figure();
                 plt.plot(val_loss_array);
                 #plt.axhline(y=math.exp(stored_loss), color='r', linestyle='-');
                 plt.savefig(model_name+'_val_loss_training.png')
    
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
            val_loss_array.append(math.exp(val_loss))

            if epoch%10==0:
                 model_name = args.save.split('.')[0]
                 plt.figure();
                 plt.plot(val_loss_array);
                 plt.axhline(y=stored_loss, color='r', linestyle='-');
                 plt.savefig(model_name+'_val_loss_training.png')

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

if True: 
    model_name = args.save.split('.')[0]
    model_load(args.save)
        
    # Run on test data.                                                                                                                     
    test_loss = evaluate(test_data, test_batch_size)                                                                                        
    print('=' * 89)                                                                                                                         
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(                                             
    test_loss, math.exp(test_loss), test_loss / math.log(2)))                                                                          
    print('=' * 89)                                                                                                                         
    
    #print forget gate bias values for ensurity

    dict_param = dict(model.named_parameters())                                                                                             
    hid_dim = [1150,1150,400]                                                                                                               
    
    for l in range(3):                                                                                                                      
        x = (dict_param['rnns.'+str(l)+'.module.bias_ih_l0'].data[0:hid_dim[l]*2].cpu() + dict_param['rnns.'+str(l)+'.module.bias_hh_l0'].data[0:hid_dim[l]*2].cpu())  
        #-----------------------------------------------------------------------------------------------------------------
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

exit(0)



