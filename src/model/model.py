import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import sys
import pickle
import re
from model import torch_utils
# model definition
class AutoCompleteNet(nn.Module):
    def __init__(self, vocab_size, feature_size):
        super(AutoCompleteNet, self).__init__()
        self.vocab_size = vocab_size
        self.feature_size = feature_size
        self.encoder = nn.Embedding(self.vocab_size, self.feature_size)
        self.gru = nn.GRU(self.feature_size, self.feature_size, batch_first=True,num_layers=5, dropout=.20,bidirectional=False)
        self.decoder = nn.Linear(self.feature_size, self.vocab_size)
        # This shares the encoder and decoder weights as described in lecture.
        self.decoder.weight = self.encoder.weight
        self.decoder.bias.data.zero_()
        
        self.best_accuracy = -1
    
    def forward(self, x, hidden_state=None):
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        # TODO finish defining the forward pass.
        x = self.encoder(x)
        #print(f'encode output shape {x.shape}')
        x, hidden_state = self.gru(x, hidden_state)
        #print(f'gru output shape {x.shape}')
        x = self.decoder(x)

        return x, hidden_state

    # This defines the function that gives a probability distribution and implements the temperature computation.
    def inference(self, x, hidden_state=None, temperature=1):
        x = x.view(-1, 1)
        x, hidden_state = self.forward(x, hidden_state)
        x = x.view(1, -1)
        x = x / max(temperature, 1e-20)
        x = F.softmax(x, dim=1)
        return x, hidden_state

    # Predefined loss function
    def loss(self, prediction, label, reduction='mean'):
        try:
            loss_val = F.cross_entropy(prediction.view(-1, self.vocab_size), label.view(-1), reduction=reduction)
        except:
            print(prediction.view(-1, self.vocab_size).size(), label.view(-1).size())
            print(prediction.view(-1, self.vocab_size))
            print(label.view(-1))
            raise
        return loss_val

    # Saves the current model
    def save_model(self, file_path, num_to_keep=3):
        torch_utils.save(self, file_path, num_to_keep)

    # Saves the best model so far
    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        if accuracy > self.best_accuracy:
            self.save_model(file_path, num_to_keep)
            self.best_accuracy = accuracy

    def load_model(self, file_path):
        torch_utils.restore(self, file_path)

    def load_last_model(self, dir_path):
        return torch_utils.restore_latest(self, dir_path)