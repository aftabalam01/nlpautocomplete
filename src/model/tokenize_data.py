# data processing classes and method
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import re
from model import torch_utils as utils
from pathlib import Path
import string

DATA_PATH = os.path.join(Path(__file__).parent, '..', 'data')

class Vocabulary(object):
    def __init__(self,train_file=None):
        self.UNK = '__unk__'
        self.START = '__<s>__'
        self.PAD = '__pad__'
        self.STOP = '\n'
        self.train_file = train_file
        if self.train_file:
            self._build_vocab()
    
    def _build_vocab(self):
        with open(self.train_file, 'r') as f:
            data = f.read()
        data = nlp_text_preprocessing(data) # removing newlines as I am going newlines as stop symbol
        char_list = [self.PAD]+[self.UNK]+ list(set(data)) + [self.STOP]
        #char_list = list(set(char_list))
        print(len(char_list), char_list)
        voc2ind = {}
        ind2voc = {}
        voc2ind.update({char : char_list.index(char) for char in char_list})
        ind2voc.update({char_list.index(char): char  for char in char_list})
        self.vocab = {'voc2ind':voc2ind,'ind2voc': ind2voc}
        # Returns a string representation of the tokens.
        print("saving vocab as pickle file")
        pickle.dump(self.vocab, open(f'{DATA_PATH}/vocabulary.pkl', 'wb'))

    def get(self):
        with open(f'{DATA_PATH}/vocabulary.pkl', 'rb') as vocab_file:
            self.vocab = pickle.load(vocab_file)
        return self.vocab

    def array_to_sentence(self, arr):
        return ''.join([self.vocab['ind2voc'][int(ind)] for ind in arr])

    # Returns a torch tensor representing each token in words.
    def sentence_to_array(self, sentence):
        return torch.LongTensor([self.vocab['voc2ind'].get(char,self.vocab['voc2ind'][self.UNK]) for char in sentence])

    # Returns the size of the vocabulary.
    def __len__(self):
        return len(self.vocab['voc2ind'])
    

def nlp_text_preprocessing(text):
  """
  Do all common preprocessing steps
  https://towardsdatascience.com/nlp-text-preprocessing-a-practical-guide-and-template-d80874676e79
  """
  # remove leading/trailing space
  text = text.strip()

  # replace whitespace characters with a single one
  #text = " ".join(text.split())
  text = re.sub('\s+', ' ', text)
  text = re.sub('\n+', '\n', text)
  text = re.sub('\t+', ' ', text)
  text = re.sub(f'[^{re.escape(string.printable)}]', '', text)
  return text

def prepare_data(data_file, vocab=None,data_type='train',is_sample=False):
    """
    converts data files into tokens

    """
    if not vocab:
        vocab = Vocabulary()
        vocab.get()
    with open(data_file) as f:
        # This reads all the data from the file, but does not do any processing on it.
        lines = f.readlines()
    if is_sample:
        lines = lines[:1000]
    line_tokens=[]
    for line in lines:
        data = nlp_text_preprocessing(line) 
        line_token = vocab.sentence_to_array(data).tolist() + [vocab.vocab['voc2ind'][vocab.STOP]]
        line_tokens = [*line_tokens,line_token]

    pickle.dump({'tokens': line_tokens}, open(f'{data_file}.pkl', 'wb'))

def tokenize_prepare_data():
    print("Creating vocab using train data")
    vocab = Vocabulary(f'{DATA_PATH}/train')
    print(len(vocab) , vocab.get()['voc2ind'] )
    print("Coverting train sentences to tokens")
    prepare_data(data_file=f'{DATA_PATH}/train')
    print("Coverting test sentences to tokens")
    prepare_data(vocab=vocab,data_file=f'{DATA_PATH}/test_freq')
    print("Coverting validation sentences to tokens")
    prepare_data(vocab=vocab,data_file=f'{DATA_PATH}/valid_freq')


if __name__=='__main__':
    tokenize_prepare_data()