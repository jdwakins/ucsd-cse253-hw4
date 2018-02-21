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
class LSTM_Mod2(nn.Module):
# class LSTM_Mod2():
    def __init__(self, hidden_dim, vocab_size, bs):
        super(LSTM_Mod2, self).__init__()

        self.hidden_dim = hidden_dim  # make 100
        # vocab size is number of characters
        # embedding dim is batch size..?
        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(1, hidden_dim, 1)
        # input is 10x30

        # nn.LSTM might be faster and can process batches at a time

        # self.lstm2 = nn.LSTM(hidden_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        # tag space is number of characters in data
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.hidden = self.init_hidden()
        self.cell = self.init_hidden()
        self.bs = bs

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)

        # must wrap these in cuda as well for GPU version
        if use_gpu:
            return (Variable(torch.zeros(1, self.bs, self.hidden_dim).cuda()),
                    Variable(torch.zeros(1, self.bs, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(1, self.bs, self.hidden_dim)),
                    Variable(torch.zeros(1, self.bs, self.hidden_dim)))

    def forward(self, sentence):
        outputs=[]
        # lstm_out, (self.hidden, self.cell) = self.lstm(sentence.float().view(1,1,-1),(self.hidden, self.cell))
        # outputs = self.linear(self.hidden)
        for character in sentence:
            output, self.hidden = self.lstm(character.float().view(1,self.bs,-1), self.hidden)
            output = self.linear(output)
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

def get_accuracy(outputs, labels, correct, total):
    _, predicted = torch.max(outputs.data, 1)
    correct += (predicted == labels.data).sum()
    total += labels.size(0)
    running_acc = ((correct/float(total)) *100.0)
    return correct, total, running_acc

def prepare_data(data_nums, is_gpu):
    tensor = torch.LongTensor(data_nums)
    if isinstance(data_nums, list):
        tensor.unsqueeze_(0)
    tensor.unsqueeze_(2)
    if is_gpu:
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

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
batch_size = 10

model = LSTM_Mod2(100, vocab_size, batch_size)
if use_gpu:
    model.cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(2):
    #get random slice
    a = range(len(training_data) - 30)

    # after going through all of a , will have gone through all possible 30
    # character slices
    total = 0
    correct = 0
    iterate = 0
    while len(a) >0:
        idxs = random.sample(a,batch_size)
        rand_slice = [training_nums[idx : idx + 30] for idx in idxs]
        rand_slice = np.array(rand_slice).T
        targets = [training_nums[idx + 1:idx+31] for idx in idxs]
        targets = np.array(targets).T

        for idx in idxs:
            a.remove(idx)

        # turn data and targets into input and target indices for model
        # wrap rand_slice and targets in cuda for GPU version
        rand_slice = prepare_data(rand_slice, use_gpu)

        targets = prepare_data(targets, use_gpu)
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        # From TA:
        # another option is to feed sequences sequentially and let hidden state continue
        # could feed whole sequence, and then would kill hidden state

        # Step 3. Run our forward pass.
        outputs = model(rand_slice)

        outputs = torch.cat(outputs)
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss=0
        for bat in range(batch_size):
            loss += loss_function(outputs[:,bat,:], targets[:,bat,:].squeeze(1))
        loss.backward()
        optimizer.step()

        # correct, total, running_accuracy = get_accuracy(outputs.squeeze(1), targets, correct, total)
        if iterate % 2000 == 1999:
            print('Loss ' + str(loss.data[0]))
            outputs_val = model(val_inputs)
            # outputs_val = torch.cat(outputs_val)
            val_loss = loss_function(outputs_val, val_targets)
            print('Validataion Loss ' + str(val_loss))
        iterate += 1
    print('Completed Epoch ' + str(epoch))
