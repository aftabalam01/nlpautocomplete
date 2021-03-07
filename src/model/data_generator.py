import os
import torch
import pickle
import numpy as np
import random
from pathlib import Path
from model.tokenize_data import Vocabulary

DATA_PATH = os.path.join(Path(__file__).parent, '..', 'data')

def ngram(n, tokens):
    """
    input list of tokens
    n - ngram 

    output: n-gram tokens
    """
    N = len(tokens)
    ngrams = []
    for t in range(N-n+1):
        ngrams = [*ngrams,tokens[t:t+n]]
    return ngrams

class AutoCompleteDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, sequence_length, batch_size):
        super(AutoCompleteDataset, self).__init__()
        self.vocab = Vocabulary().get()
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.datas =[] 
        self.labels =[]

        with open(data_file, 'rb') as data_pkl:
            dataset = pickle.load(data_pkl)
        data = dataset['tokens']
        inputs_list=[]
        targets_list = []
        i=0
        print(f"Starting to build {self.sequence_length} character lengths seq")
        for line in data:
            i +=1
            if i%10000==0:
              print(f'{i} sentences are processes')
            ngram_tokens = ngram(n=self.sequence_length,tokens=line)
       
            inputs_list.extend(ngram_tokens[:-1])
            targets_list.extend(ngram_tokens[1:])
        len_tokens = int((len(inputs_list)//self.batch_size) * self.batch_size)
        self.datas = inputs_list[0:len_tokens]
        self.labels = targets_list[0:len_tokens]

    def array_to_sentence(self, arr):
        return ''.join([self.vocab['ind2voc'][int(ind)] for ind in arr])

    # Returns a torch tensor representing each token in words.
    def sentence_to_array(self, sentence):
        return torch.LongTensor([self.vocab['voc2ind'].get(char,self.vocab['voc2ind'][self.UNK]) for char in sentence])

    def __len__(self):
        # TODO return the number of unique sequences you have, not the number of characters.
         return len(self.datas)
        
    def __getitem__(self, idx):
        # Return the data and label for a character sequence for idx batch number
        try:
            data = torch.LongTensor([data for data in self.datas[idx]])
            label = torch.LongTensor([label for label in self.labels[idx]])
        except:
            print(self.labels[idx])
            raise
        return data, label

    def vocab_size(self):
        return len(self.vocab['voc2ind'])
if __name__=='__main__':
    BATCH_SIZE =  5
    SEQUENCE_LENGTH = 10
    dataset = AutoCompleteDataset(data_file=f'{DATA_PATH}/valid_freq_sample.pkl',sequence_length=SEQUENCE_LENGTH,batch_size=BATCH_SIZE)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    iter = enumerate(loader)
    for batch_num in range(1):
        print(f"Retrieving batch {batch_num}...")
        id, (data,label) = next(iter)
        print(f"Got batch: {id} type({data}) \n {label}")
        print(f'input : {dataset.array_to_sentence(data[-1])} \n and label :\n {dataset.array_to_sentence(label[-1])}')
