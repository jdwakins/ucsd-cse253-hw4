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
class LSTM_Mod2(nn.Module):
# class LSTM_Mod2():
    def __init__(self, hidden_dim, vocab_size, bs, seq_len, is_gpu = False):
        super(LSTM_Mod2, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(1, hidden_dim, 1)
        # The linear layer maps from hidden state space to target space
        # target space = vocab size, or number of unique characters in daa
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax()
        self.bs = bs
        self.seq_len = seq_len
        self.is_gpu = is_gpu
        self.hidden = self.init_hidden()
        self.cell = self.init_hidden()


    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.is_gpu:
            return (Variable(torch.zeros(1, self.bs, self.hidden_dim).cuda()),
                    Variable(torch.zeros(1, self.bs, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(1, self.bs, self.hidden_dim)),
                    Variable(torch.zeros(1, self.bs, self.hidden_dim)))

    def forward(self, sentence):
        outputs=[]
        # input sentence is shape: sequence x batch x 1
        output, self.hidden = self.lstm(sentence.float().view(self.seq_len,self.bs,-1), self.hidden)
        outputs = self.linear(output)
        # for character in sentence:
        #     output, self.hidden = self.lstm(character.float().view(1,self.bs,-1), self.hidden)
        #     output = self.linear(output)
        #     outputs += [output]
        return outputs
