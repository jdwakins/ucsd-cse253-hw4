from __future__ import print_function
import torch.autograd as autograd
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import pdb
import numpy as np
# for feature evaluation - pick one neuron

# use_gpu = torch.cuda.is_available()
class LSTM_Dual(nn.Module):
# class LSTM_Mod2():
    def __init__(self, hidden_dim, vocab_size, bs, seq_len, meta_dim, is_gpu = False):
        super(LSTM_Mod2, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(1, hidden_dim, 1)
        # The linear layer maps from hidden state space to target space
        # target space = vocab size, or number of unique characters in daa
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.linear2 = nn.Linear(meta_dim, vocab_size)
        self.bs = bs
        self.seq_len = seq_len
        self.is_gpu = is_gpu
        self.hidden = self.init_hidden()


    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.is_gpu:
            self.hidden = (Variable(torch.zeros(1, self.bs, self.hidden_dim).cuda()),
                    Variable(torch.zeros(1, self.bs, self.hidden_dim).cuda()))
        else:
            self.hidden = (Variable(torch.zeros(1, self.bs, self.hidden_dim)),
                    Variable(torch.zeros(1, self.bs, self.hidden_dim)))

    def forward(self, primary, secondary):
        # outputs=[]
        # input primary is shape: sequence x batch x 1
        output, self.hidden = self.lstm(primary.float().view(-1, self.bs, 1), self.hidden)
        meta = self.linear2(secondary)
        outputs = self.linear(output)
        outputs = torch.cat([meta, outputs], 1)
        # for character in primary:
        #     output, self.hidden = self.lstm(character.float().view(1,self.bs,-1), self.hidden)
        #     output = self.linear(output)
        #     outputs += [output]
        return outputs
