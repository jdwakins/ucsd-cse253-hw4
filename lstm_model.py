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

# need non empty commit

class LSTM_Mod(nn.Module):

    def __init__(self, hidden_dim, vocab_size):
        super(LSTM_Mod, self).__init__()
        self.hidden_dim = hidden_dim  # make 100
        # vocab size is number of characters
        # embedding dim is batch size..?
        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(1, hidden_dim, 1)
        # self.lstm2 = nn.LSTM(hidden_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        # tag space is number of characters in data
        self.linear = nn.Linear(hidden_dim, 1)
        self.hidden = self.init_hidden()
        self.cell = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        outputs = []
        # lstm_out, (self.hidden, self.cell) = self.lstm(sentence.float().view(1,1,-1),(self.hidden, self.cell))
        # outputs = self.linear(self.hidden)
        for character in sentence:
            pdb.set_trace()
            lstm_out, (self.hidden, self.cell) = self.lstm(character.float().view(1,1,-1),(self.hidden, self.cell))
            output = self.linear(self.hidden)
            outputs += [output]
        return outputs

# input data
file = open('input.txt', 'r')
data = file.read()

# function maps each word to an index
def get_idx(char_data):
    word_to_ix = {}
    for word in char_data:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    return word_to_ix

# get 30 character random slices of dataset
# slice data into trianing and testing
slice_ind = int(round(len(data)*.8))
training_data = data[:slice_ind]
val_data = data[slice_ind:]

training_idx = get_idx(training_data)
val_idx = get_idx(val_data)

np.random.seed(0)

model = LSTM_Mod(100, len(training_data))
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.00001)

for epoch in range(2):
    #get random slice
    a = range(len(training_data) - 30)

    # after going through all of a , will have gone through all possible 30
    # character slices
    while len(a) >0:
        idx = random.choice(a)
        a.remove(idx)

        # turn data and targets into input and target indices for model
        rand_slice = training_data[idx : idx + 30]
        rand_idx = [training_idx[char] for char in rand_slice]
        rand_idx = Variable(torch.LongTensor(rand_idx))

        targets = training_data[idx + 1:idx+31]
        target_idx = [training_idx[char] for char in targets]
        target_idx = Variable(torch.LongTensor(target_idx))

        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        # I (jen) commented this out because idk why the tutorial does it..??
        # model.hidden = model.init_hidden()

        # Step 3. Run our forward pass.
        tar_scores = model(rand_idx)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tar_scores, target_idx)
        loss.backward()
        optimizer.step()
        pdb.set_trace()
