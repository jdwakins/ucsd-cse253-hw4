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

# input data
with open('input.txt', 'r') as f:
    data = f.read()

# check for GPU
use_gpu = torch.cuda.is_available()

# function maps each word to an index
def get_idx(char_data):
    word_to_ix = {}
    for word in char_data:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    return word_to_ix

def prepare_data(data_nums, is_gpu):
    tensor = torch.LongTensor(data_nums)
    if isinstance(data_nums, list):
        tensor.unsqueeze_(0)
    tensor.unsqueeze_(2)
    if use_gpu:
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

vocab_idx = get_idx(data)
vocab_size = len(vocab_idx)

# slice data into trianing and testing (could do this much better)
slice_ind = int(round(len(data)*.8))
training_data = data[:slice_ind]
val_data = data[slice_ind:]

# turn training and validation data from characters to numbers
training_nums = [vocab_idx[char] for char in training_data]
val_nums = [vocab_idx[char] for char in val_data]

val_inputs = prepare_data(val_nums[:-1], use_gpu)
val_targets = prepare_data(val_nums[1:], use_gpu)

np.random.seed(0)

# set batch size & sequence length
seq_len = 30
batch_size = 10
# call model
model = LSTM_Mod2(100, vocab_size, batch_size, seq_len, is_gpu=use_gpu)
if use_gpu:
    model.cuda()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# train model
for epoch in range(2):
    #get random slice
    a = range(len(training_data) - (seq_len+1))
    # after going through all of a , will have gone through all possible 30
    # character slices
    total = 0
    correct = 0
    iterate = 0
    while len(a) >0:
        idxs = random.sample(a,batch_size)
        # get random slice, and the targets that correspond to that slice
        rand_slice = [training_nums[idx : idx + seq_len] for idx in idxs]
        rand_slice = np.array(rand_slice).T
        targets = [training_nums[idx + 1:idx+(seq_len+1)] for idx in idxs]
        targets = np.array(targets).T

        for idx in idxs:
            a.remove(idx)

        # prepare data and targets for model
        rand_slice = prepare_data(rand_slice, use_gpu)
        targets = prepare_data(targets, use_gpu)
        # Pytorch accumulates gradients. We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        # From TA:
        # another option is to feed sequences sequentially and let hidden state continue
        # could feed whole sequence, and then would kill hidden state

        # Run our forward pass.
        outputs = model(rand_slice)
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
            val_loss = loss_function(outputs_val, val_targets)
            print('Validataion Loss ' + str(val_loss))
        iterate += 1
    print('Completed Epoch ' + str(epoch))
