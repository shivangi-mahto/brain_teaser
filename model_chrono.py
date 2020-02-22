import torch
import torch.nn as nn
import numpy as np
import math
from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop
import scipy.stats
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False, Tmax = 20, Tmin = 1):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]

            #added by Shivi

            ## STEP1: make a list of size of hidden layers - useful for init step
            hid_dim  = [nhid if l != nlayers -1 else ( ninp if tie_weights else nhid) for l in range(nlayers)]
            print('cross check hidden dim list', hid_dim)

            ## STEP2: Create bias values depending on type of init we want
            # constant: same bias values for all units in a layer
            # gradient: gradiently increasing bias values for units in a layer, from Tmin to Tmax
            # sampling: instead of constant or gradient, just randomly sample from uniform dist - just like chrono init paper

            constant = False; gradient = False; sampling= False; custom = True
            chrono_bias = [np.zeros(hid_dim[l]) for l in range(nlayers)]


            if constant: #Init fixed bias for each units in a layer
                fixed_bias  = np.log([Tmax-1,Tmax/2-1,Tmax/4-1]) #Different biases for different layers
                
                reverse = False; mid_longest = False; 
                if reverse: # Init first layer with smallest timescale and last with largest timescale.
                    fixed_bias  = np.log([Tmax/4-1,Tmax/2-1,Tmax-1]) 
                if mid_longest: # Init middle layer with longest timescale
                    fixed_bias  = np.log([Tmax/4-1,Tmax-1,Tmax/2-1])
                
                chrono_bias = [fixed_bias[l]*np.ones(hid_dim[l]) for l in range(nlayers)]  
            
            elif custom:
              fixed_bias  = np.log([5-1,1-1])#Tmax -1 orig
              chrono_bias = [fixed_bias[l]*np.ones(hid_dim[l]) for l in range(nlayers-1)]
              modules_in_middle_layer = False ; modules_in_first_layer = True
              if modules_in_middle_layer:
                  Tmin_l = 1; Tmax_l = 20; two_module = False; pareto =  True
                  if two_module:
                      first_mod   = int(0.5*hid_dim[1]) ;
                      second_mod  = int(0.5*hid_dim[1]) + first_mod;
                      third_mod   = int(0.0*hid_dim[1]) + second_mod ;
                      chrono_bias[1][0:first_mod] = np.log(Tmin_l + (((Tmax_l - 1 - Tmin_l)*np.arange(first_mod)) / (first_mod-1) ) )
                      chrono_bias[1][first_mod:second_mod] = np.log(70-1)*np.ones(second_mod-first_mod)
                  elif pareto:
                      chrono_bias[1] = np.log(scipy.stats.pareto.isf(np.linspace(0, 1, 1151), 0.54, 1)[1:] - 1)
                  else:
                      first_mod  = int(0.8*hid_dim[1]) ; 
                      second_mod = int(0.1*hid_dim[1]) + first_mod ; 
                      third_mod  = int(0.1*hid_dim[1]) + second_mod ;
                      chrono_bias[1][0:first_mod] = np.log(Tmin_l + (((Tmax_l - 1 - Tmin_l)*np.arange(first_mod)) / (first_mod-1) ) )
                      chrono_bias[1][first_mod:second_mod] = np.log(70-1)*np.ones(second_mod-first_mod)
                      chrono_bias[1][second_mod:third_mod] = (10**8)*np.ones(third_mod-second_mod) #(10**8)*

              if modules_in_first_layer:
                  Tmin_l = 1; Tmax_l = 20; two_module = False; pareto = True
                  if two_module:
                      first_mod   = int(0.5*hid_dim[1]) ;
                      second_mod  = int(0.5*hid_dim[1]) + first_mod;
                      third_mod   = int(0.0*hid_dim[1]) + second_mod ;
                      mixed_module = False;
                      if mixed_module:
                          #gradient 1->20 for first part and fixed 70 for second part  
                          chrono_bias[0][:first_mod] = np.log(Tmin_l + (((Tmax_l - 1 - Tmin_l)*np.arange(first_mod)) / (first_mod-1) ) )
                          chrono_bias[0][first_mod:second_mod] = np.log(70-1)*np.ones(second_mod-first_mod)
                      else:
                          #Fixed timescale of 3 and 4 in first and other 50%
                          chrono_bias[0][:first_mod] = np.log(3-1)*np.ones(first_mod)                                     
                          chrono_bias[0][first_mod:second_mod] = np.log(4-1)*np.ones(second_mod-first_mod)
                  elif pareto:
                      chrono_bias[0] = np.log(scipy.stats.pareto.isf(np.linspace(0, 1, 1151), 0.54, 1)[1:] - 1)
                  else:
                      first_mod  = int(0.8*hid_dim[1]) ;
                      second_mod = int(0.1*hid_dim[1]) + first_mod ;
                      third_mod  = int(0.1*hid_dim[1]) + second_mod ;
                      chrono_bias[0][:first_mod] = np.log(Tmin_l + (((Tmax_l - 1 - Tmin_l)*np.arange(first_mod)) / (first_mod-1) ) )
                      chrono_bias[0][first_mod:second_mod] = np.log(70-1)*np.ones(second_mod-first_mod)
                      chrono_bias[0][second_mod:third_mod] = (10**8)*np.ones(third_mod-second_mod) #(10**8)*                                

            
            elif sampling:
                print('Next step to do')

            elif gradient: # slope - (Tmax-1 - Tmin)/(hidden_dim) for assigning init values
                consistent = False
                if consistent: # Each layer has units init from 1 to log(T-1)
                    diff = (Tmax - 1 - Tmin)
                    chrono_bias = [  np.log(Tmin + (  (diff*np.arange(hid_dim[l])) / (hid_dim[l]-1) ) ) for l in range(nlayers)]
                else: #Different layer has different Tmin and Tmax unlike consistent
                    Tmax_l = [Tmax, Tmax/2, Tmax/4]
                    Tmin_l = [Tmax/2, Tmax/4, Tmin]
                    chrono_bias = [np.log(Tmin_l[l] + (  ((Tmax_l[l] - 1 - Tmin_l[l])*np.arange(hid_dim[l])) / (hid_dim[l]-1) ) ) for l in range(nlayers)]  
            
            ## assign bias to first 2 or all 3 layers depending on init
            if custom: limit = nlayers-1
            else: limit = nlayers
            ##assign bias values to the layers - first half is input gate bias, second half is forget gate for both i to h and h to h
            chrono_init = True; tie_input_forget = True; partial_tie_input_forget = False
            if chrono_init and tie_input_forget and not partial_tie_input_forget:
                for l in range(limit):
                    self.rnns[l].bias_ih_l0.data[0:hid_dim[l]*2] = torch.tensor(np.zeros(hid_dim[l]*2),dtype=torch.float)  
                    self.rnns[l].bias_hh_l0.data[0:hid_dim[l]*2] = torch.from_numpy(np.hstack((-1*chrono_bias[l], chrono_bias[l] )).astype(np.float32))   
            elif chrono_init and not tie_input_forget and not partial_tie_input_forget:
                for l in range(limit):
                    self.rnns[l].bias_ih_l0.data[hid_dim[l]:hid_dim[l]*2] = torch.tensor(np.zeros(hid_dim[l]*1) , dtype=torch.float)
                    self.rnns[l].bias_hh_l0.data[hid_dim[l]:hid_dim[l]*2] = torch.from_numpy(((chrono_bias[l])).astype(np.float32))

            elif chrono_init and partial_tie_input_forget:
                for l in range(limit):
                    self.rnns[l].bias_ih_l0.data[0:hid_dim[l]*2] = torch.tensor(np.zeros(hid_dim[l]*2),dtype=torch.float)
                    self.rnns[l].bias_hh_l0.data[0:hid_dim[l]*2] = torch.from_numpy(np.hstack((-1*chrono_bias[l], chrono_bias[l] )).astype(np.float32))
                    self.rnns[l].bias_hh_l0.data[second_mod:third_mod]=torch.tensor((-1)*np.ones(third_mod-second_mod),dtype=torch.float)
                    
            ## fix the bias - if we want to fix the bias instead of just init them 
            fixed_weights = True
            if fixed_weights:
                for l in range(limit):
                    self.rnns[l].bias_ih_l0.requires_grad = False 
                    self.rnns[l].bias_hh_l0.requires_grad = False
    
         
            ## just priting values of bias in the layer to cross check if they are as expected or not 
            """
            print('Init of the layers are')
            for l in range(nlayers):
                if gradient: 
                    print('Expected gradient from 0 to',-1*math.log(Tmax-1),' and then 0 to ', math.log(Tmax-1))
                    print(self.rnns[l].bias_hh_l0.data[0:1],self.rnns[l].bias_hh_l0.data[hid_dim[l]-1:hid_dim[l]+1], self.rnns[l].bias_hh_l0.data[2*hid_dim[l]-1:2*hid_dim[l]])
                    print('Expected all 0')
                    print(self.rnns[l].bias_ih_l0.data[0:hid_dim[l]*2])
                if constant: 
                    print('Expected values to be constant from', -1*math.log(Tmax/(2**(2-l)) -1 ),' to ', math.log(Tmax/(2**(2-l)) -1 )  )
                    print(self.rnns[l].bias_hh_l0.data[0:hid_dim[l]*2])
                if custom: 
                    half_hiddim = int(0.8*hid_dim[1])
                    if l==-1:
                        print('layer0 has forget bias values from ', -1*np.log(4),' to ', np.log(4))
                        print(self.rnns[l].bias_hh_l0.data[hid_dim[l]:hid_dim[l]*2])
                        print('layer0 has input bias 0 values')
                        print(self.rnns[l].bias_ih_l0.data[hid_dim[l]:hid_dim[l]*2])
                    if l==0 or l==1:
                        print('layer '+str(l)+' has values from', np.log(1),' to ',np.log(19))
                        print(self.rnns[l].bias_hh_l0.data[hid_dim[l]:hid_dim[l]+first_mod])
                        print(self.rnns[l].bias_hh_l0.data[:first_mod])
                        
                        print('layer '+str(l)+' has 10 percent values', np.log(69))
                        print(self.rnns[l].bias_hh_l0.data[hid_dim[l]+first_mod:hid_dim[l]+second_mod])
                        print(self.rnns[l].bias_hh_l0.data[first_mod:second_mod])

                        print('layer '+str(l)+' has last 10 percent values', 10**8)
                        print(self.rnns[l].bias_hh_l0.data[hid_dim[l]+second_mod:hid_dim[l]+third_mod])
                        print('layer'+str(l)+' input has last 10 percent values', -1)
                        print(self.rnns[l].bias_hh_l0.data[second_mod:third_mod])
                    
                        print('Expected 0 values')
                        print(self.rnns[l].bias_ih_l0.data[hid_dim[l]:hid_dim[l]*2])
                    if l==2: 
                        print('Expected random values')
                        #print(self.rnns[l].bias_hh_l0.data[0:hid_dim[l]*2])
                        #print(self.rnns[l].bias_ih_l0.data[0:hid_dim[l]*2])
            """
            ##end edit by shivi
            ###
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            if False:
                if l == 2:
                    print('Guess who is damn stupid ?', current_input.shape)
                    hiddim = current_input.shape[-1]
                    half_hiddim = int(0.5*hiddim)
                    #current_input[:,:,:half_hiddim] = torch.tensor(np.zeros(half_hiddim) , dtype=torch.float)
                    current_input[:,:,half_hiddim:hiddim] = torch.tensor(np.zeros(hiddim-half_hiddim) , dtype=torch.float)
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs #check if raw and outputs are same
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]
