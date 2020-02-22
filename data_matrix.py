import numpy as np
from pathlib import Path
import sys
import os
path = Path(__file__).parent.absolute()
sys.path.append(str(path)+'/cottoncandy')

import matplotlib.pyplot as plt
import cottoncandy as cc

access_key = 'SSE14CR7P0AEZLPC7X0R'
secret_key = 'K0MmeXiXotrGIiTeRwEKizkkhR4qFV8tr8cIXprI'
endpoint_url = 'http://c3-dtn02.corral.tacc.utexas.edu:9002/'

cci_e2e = cc.get_interface('endtoend',verbose=False,ACCESS_KEY=access_key, SECRET_KEY=secret_key,endpoint_url=endpoint_url)
cci_data = cc.get_interface('neural-model-data',verbose=False,ACCESS_KEY=access_key, SECRET_KEY=secret_key,endpoint_url=endpoint_url)
cci = cc.get_interface('lstm-timescales',verbose=False, ACCESS_KEY=access_key, SECRET_KEY=secret_key,endpoint_url=endpoint_url)

if False:
    vocab_file = 'vocab_70stim+top10k'
    vocab = np.array(cci_data.download_raw_array(vocab_file), str)
    nvocab = len(vocab)
    int2word = {i: vocab[i] for i in range(nvocab)}

    vocab_penn = cci.download_npy_array('shivangi/'+'word2dict_'+ 'data/penn')
    word2index = {vocab_penn[i]:i for i in range(len(vocab_penn))}
    allstories = cci_e2e.lsdir('story_arrays')

def download_reddit():

    #cci_data.download_to_file('reddit_docs_250.npz','/data2/shivangi_data/reddit_data/')

    word2index = cci_data.download_json('stimulus_ptb_stim_shivangi/word2index')
    documents = np.load('/data2/shivangi_data/reddit_data/reddit_docs_250_1.npz', allow_pickle=True)['docs']
    joint_documents = [' '.join([' '.join(x) for x in doc]) for doc in documents]
        
    good_docs = []
    t = 0
    for doc in joint_documents:
        words = doc.split(' ')
        l = 0
        for word in words:
            if word not in word2index:
                l += 1
        if l / len(words) <= 0.05:
            good_docs.append(doc)
        t += 1
        if t % 100 == 0:
            print(t, len(good_docs))


    good_docs = [doc.split(' ') for doc in good_docs] 
    good_docs = [[word2index.get(word, word2index['<unk>']) for word in doc] for doc in good_docs]

    cci_data.upload_npy_array('stimulus_ptb_stim_shivangi/reddit_story_arrays/good_docs' ,np.array(good_docs))

    #doc_seq   = get_doc_sequence_arrays(good_docs, 70)
    #cci_data.upload_npy_array('stimulus_seqs71_shivangi/%s'%('reddit'), doc_seq)
    #cci_data.upload_json('stimulus_ptb_stim_shivangi/ptb_stim_red_word2index', word2index)
        
    return 

def get_reddit_sequence_arrays(docs, seq_len):
    in_seq = []
    for doc in docs:
        l = len(doc)
        for j in range(int(l/seq_len)):
            start = j * seq_len
            end = (j+1) * seq_len
            if end < l:
                in_seq.append(np.array(doc[start:end]))
    in_seq = np.array(in_seq, dtype=np.int32)
    print(in_seq.shape)
    return in_seq


download_reddit()
def align_dicts(vocab_penn, vocab):
    align, total = 0,0
    for word in vocab:
        total += 1
        if word in vocab_penn: 
            align += 1
    return align,total

def make_arrays(story_array, seq_len, unk):
    l = story_array.shape[0]
    seq = unk * np.ones([story_array.shape[0], seq_len] , dtype=np.int64)
    for j in range(seq_len):
        if j > 0:
            seq[:-j, j] = story_array[j:]
        else:
            seq[:, j] = story_array[j:]
    return seq
    
count = set(); count_all = 0

def extract_seq_test():
    allstories = cci_e2e.lsdir('story_arrays')
    for story in allstories:
        orig_seq = cci_e2e.download_raw_array('%s'%story)
    
        story_words = list(map(int2word.get, orig_seq))
        story_token = np.array([word2index.get(word, word2index['<unk>']) for word in story_words])
    
        seq = make_arrays(story_token, 70, word2index['<unk>'])
        cci_data.upload_npy_array('stimulus_seqs71_shivangi/%s'%(story.replace('story_arrays/', '')), seq)    
    return 

def make_arrays(story_array, seq_len, unk):
    l = story_array.shape[0]
    seq = unk * np.ones([story_array.shape[0], seq_len] , dtype=np.int64)
    for j in range(seq_len):
        if j > 0:
            seq[:-j, j] = story_array[j:]
        else:
            seq[:, j] = story_array[j:]
    return seq

def extract_vectors():
    allstories = cci_e2e.lsdir('story_arrays')
    word2index = cci_data.download_json('stimulus_ptb_stim_shivangi/word2index')
    for story in stories:
        story_array = cci_data.download_raw_array('stimulus_ptb_stim_shivangi/story_arrays/%s' % story)                                                                                      
        seq = make_arrays(story_array, 70, word2index['<unk>'])
        print(story, seq.shape)
        cci_data.upload_npy_array('stimulus_ptb_stim_shivangi/seq_arrays/%s'%(story.replace('story_arrays/', '')), seq)
    return

def extend_vocab():
    vocab_penn = cci.download_npy_array('shivangi/'+'word2dict_'+ 'data/penn')
    word2index = {vocab_penn[i]:i for i in range(len(vocab_penn))}
    nvocab = len(word2index)
    stimuli_words = cci_data.download_json('stimulidb_words')
    print('Previous length of PTB',nvocab)
    for story in list(stimuli_words.keys()):
        if story in ['wheretheressmoke', 'fromboyhoodtofatherhood', 'onapproachtopluto']:
            story_array = [word2index.get(word, word2index['<unk>']) for word in stimuli_words[story].split(' ')+ ['<eos>']]
        else:
            story_words = stimuli_words[story].split(' ')+ ['<eos>']
            story_array = []
            
            for word in story_words:
                if word not in word2index:
                    word2index[word] = nvocab
                    nvocab += 1
                story_array.append(word2index[word])
        
        cci_data.upload_raw_array('stimulus_ptb_stim_shivangi/story_arrays/%s' % story, np.array(story_array))
    cci_data.upload_json('stimulus_ptb_stim_shivangi/word2index', word2index)
    print('length of ptb_stim vocab', len(word2index))
    return

def create_dataset():
    
    allstories = [story.replace('story_arrays/', '') for story in cci_e2e.lsdir('story_arrays')]
    
    story_arrays,test_story_arrays,val_arrays = [], [],[]
    base = 'stimulus_ptb_stim_shivangi/story_arrays/'
    for f in cci_data.lsdir(base):
        story = f.replace(base, '')
        if story in ['wheretheressmoke', 'fromboyhoodtofatherhood', 'onapproachtopluto']: 
            test_story_arrays.append(cci_data.download_raw_array(f))
        elif story in allstories and story not in ['wheretheressmoke', 'fromboyhoodtofatherhood', 'onapproachtopluto']:
            val_arrays.append(cci_data.download_raw_array(f))
        else:
            story_arrays.append(cci_data.download_raw_array(f))
    
    
    #order = list(range(len(story_arrays)))
    train_arrays = story_arrays

    cci_data.upload_npy_array('stimulus_ptb_stim_shivangi/stimulus_train_val_order', np.array(order))
    cci_data.upload_npy_array('stimulus_ptb_stim_shivangi/stimulus_train_arrays', np.array(train_arrays))
    cci_data.upload_npy_array('stimulus_ptb_stim_shivangi/stimulus_val_arrays', np.array(val_arrays))
    cci_data.upload_npy_array('stimulus_ptb_stim_shivangi/stimulus_test_arrays', np.array(test_story_arrays))

    return 

def plot_encoding_model_output(subject):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    model_list = ['PTB_1000_epochs', 'PTB_modular_80_20', 'PTB_l1_3_4_l2_pareto']
    for i, model in enumerate(model_list):
        axs = axes[i]
        for layer in range(1, 4):
            r2 = []
            for cl in [0,2,4,8,16,32,64]:
                #os.system('python3 lstm_encoding.py --subject AA --model %s --layer %d --cl 0 --nboots 10' % (model, layer))
                save_location = '%s/%s/layer%s/cl%s' % (subject, model, layer, cl)
                corrs = cci.download_raw_array(os.path.join(save_location, 'corrs'))
                
                r2.append(sum(corrs * np.abs(corrs)))
            axs.plot([0,2,4,8,16,32,64], r2, label='Layer '+str(layer))
            axs.legend(loc='upper right')

            axs.xlabel('Context length'); 
            axs.ylabel('Encoding model performance')
    

    plt.savefig('Encoding_model_'+subject)

#plot_encoding_model_output('AA')
#extend_vocab()
#create_dataset()
