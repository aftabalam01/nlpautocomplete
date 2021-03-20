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
        num_rrn_layers = 8
        self.gru = nn.GRU(self.feature_size, self.feature_size, batch_first=True, num_layers=num_rrn_layers, dropout=0,
                          bidirectional=False)
        layered_hidden_dim = self.feature_size * num_rrn_layers
        self.output = nn.Linear(layered_hidden_dim, self.vocab_size)
        self.best_accuracy = -1

    def forward(self, x):
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        a = x.shape
        x = self.encoder(x)
        x, hidden_state = self.gru(x)
        hidden_transformed = hidden_state.transpose(0, 1).reshape(hidden_state.shape[1], -1)
        return self.output(hidden_transformed)

    # Predefined loss function
    def loss(self, prediction, label, reduction='mean'):
        try:
            loss_val = F.cross_entropy(prediction, label)
        except:
            raise
        return loss_val

    # Saves the current model
    def save_model(self, file_path, num_to_keep=3):
        torch_utils.save(self, file_path, num_to_keep)

    # Saves the best model so far
    def save_best_model(self, accuracy, file_path, num_to_keep=5):
        if accuracy > self.best_accuracy:
            self.save_model(file_path, num_to_keep)
            self.best_accuracy = accuracy

    def load_model(self, file_path, device='cpu'):
        torch_utils.restore(self, file_path, device)

    def load_last_model(self, dir_path):
        return torch_utils.restore_latest(self, dir_path)
