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

use_gpu = torch.cuda.is_available()
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
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.hidden = self.init_hidden()
        self.cell = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)

        # must wrap these in cuda as well for GPU version
        if use_gpu:
            return (Variable(torch.zeros(1, 1, self.hidden_dim).cuda()),
                    Variable(torch.zeros(1, 1, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                    Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        outputs = []
        # lstm_out, (self.hidden, self.cell) = self.lstm(sentence.float().view(1,1,-1),(self.hidden, self.cell))
        # outputs = self.linear(self.hidden)
        for character in sentence:
            # pdb.set_trace()
            lstm_out, self.hidden = self.lstm(character.float().view(1,1,-1),self.hidden)
            output = self.linear(self.hidden[0])
            outputs += [output]
        return outputs

# input data
with open('input.txt', 'r') as f:
    data = f.read()

# file = open('input.txt', 'r')
# data = file.read()

# function maps each word to an index
def get_idx(char_data):
    word_to_ix = {}
    for word in char_data:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    return word_to_ix


def prepare_data(data_nums, is_gpu):
    if is_gpu:
        return Variable(torch.LongTensor(data_nums).cuda())
    else:
        return Variable(torch.LongTensor(data_nums))

# get 30 character random slices of dataset
# slice data into trianing and testing
slice_ind = int(round(len(data)*.8))
vocab_idx = get_idx(data)
vocab_size = len(vocab_idx)

training_data = data[:slice_ind]
val_data = data[slice_ind:]

training_nums = [vocab_idx[char] for char in training_data]
val_nums = [vocab_idx[char] for char in val_data]
val_inputs = prepare_data(val_nums[:-1], use_gpu)
val_targets = prepare_data(val_nums[1:], use_gpu)

np.random.seed(0)

model = LSTM_Mod(100, vocab_size)
if use_gpu:
    model.cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.00001)

for epoch in range(2):
    #get random slice
    a = range(len(training_data) - 30)

    # after going through all of a , will have gone through all possible 30
    # character slices
    total = 0
    correct = 0
    iterate = 0
    while len(a) > 0:
        idx = random.choice(a)
        a.remove(idx)

        # turn data and targets into input and target indices for model
        # wrap rand_slice and targets in cuda for GPU version
        rand_slice = training_nums[idx : idx + 30]
        rand_slice = prepare_data(rand_slice, use_gpu)

        targets = training_nums[idx + 1:idx+31]
        targets = prepare_data(targets, use_gpu)

        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.

        # I don't understand why the tutorial does this but it doesn't work if
        # the hidden layer isn't re-initailized
        model.hidden = model.init_hidden()

        # Step 3. Run our forward pass.
        outputs = model(rand_slice)

        outputs = torch.cat(outputs)
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(outputs.squeeze(1), targets)
        loss.backward()
        optimizer.step()

        correct, total, running_accuracy = get_accuracy(outputs.squeeze(1), targets, correct, total)
        if iterate % 2000 == 1999:
            print('Accuracy: ' + str(running_accuracy))
            print('Loss' + str(loss.data[0]))
            outputs_val = model(val_inputs)
            outputs_val = torch.cat(outputs_val)
            _, _, val_accuracy = get_accuracy(outputs_val.squeeze(1), val_targets, 0, 0)
            print('Validataion Accuracy' + str(val_accuracy))
        iterate += 1
    print('Completed Epoch ' + str(epoch))
