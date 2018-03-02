
#Plot hidden states

from __future__ import print_function
import torch.autograd as autograd
from torch.autograd import Variable
import torch
import csv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import pdb
import numpy as np

from helper import *
from train_model import *
from LSTM import *
from generate_music import *

import seaborn as sns


# input data
start_char = '$'  # Every music sample <start> will be marked with $
end_char = '%'    # Every music sample <end> will be marked with %
pad_char = '`'
data = clean_up_data('input.txt', start_char, end_char) #Last <end> does not get replaced by %
len_ABC_file = avg_len_music_file(data, start_char) #Avg length of music file in ABC format

# import IPython; IPython.embed()

vocab = get_idx(data + start_char + end_char + pad_char) #!!! WHY + $,%,'??
# check for GPU
use_gpu = torch.cuda.is_available()
seq_len = 30
# Increment sequence length by 10% each epoch.
hidden_layer_size = 100
seq_incr_perc = 0.15
seq_incr_freq = 1
lr = 0.001

batch_size = 30
num_epochs = 10

predict_length = 100
primer = start_char + '\nX:'
temperature = 1

model = LSTM_Mod2(hidden_layer_size, vocab, batch_size, seq_len, data, end_char,
                  start_char, pad_char, is_gpu=use_gpu)

train_loss, val_loss = model.train(vocab, seq_len, batch_size,
                                   num_epochs, lr, seq_incr_perc,
                                   seq_incr_freq=seq_incr_freq)


# Load model trained on GPU into CPU
model.load_state_dict(torch.load('model3.pt', map_location=lambda storage, loc: storage))

####

words = model.daydream(primer, temperature, predict_len=1000)
print(words)



#Forward pass

words_encoded = [vocab[c] for c in words]
words_encoded = np.array(words_encoded).T

words_encoded = torch.LongTensor(words_encoded)
words_encoded = words_encoded.unsqueeze_(1)
words_encoded = words_encoded.unsqueeze_(1)
words_encoded = Variable(words_encoded)

init_hidden(model) #Restart
model.batch_size = 1
# out, hidden = lstm(i.view(1, 1, -1), hidden)

# Outputs: output, (h_n, c_n)
# output (seq_len, batch, hidden_size * num_directions): tensor containing the output features (h_t) from the last layer of the RNN, for each t.
# If a torch.nn.utils.rnn.PackedSequence has been given as the input, the output will also be a packed sequence.
# h_n (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t=seq_len
# c_n (num_layers * num_directions, batch, hidden_size): tensor containing the cell state for t=seq_len
output, (h_n,c_n) = model.lstm(words_encoded.float().view(-1, model.batch_size, 1), model.hidden)
output = output.data.numpy() #Convert torch tensor to numpy

# output.shape() = [time, 1, units]
hidden = output[:,0,7]
labels = weights_to_2d(np.array(list(words)))
pixmap = weights_to_2d(hidden).astype(float)

plt.figure()
# fig, ax = plt.subplots()
sns.heatmap(pixmap, annot=labels, fmt = '', cmap="coolwarm", xticklabels =False, yticklabels=False)
plt.show(block=False)


def weights_to_2d(weights):
    dim1 = int(np.ceil(np.sqrt(len(weights))))
    zero_pad = dim1*dim1 - len(weights) #Add zeros at end of vector if necesary to make len squared
    weights = np.pad(weights, (0,zero_pad), 'constant')
    return np.reshape(weights, (dim1, dim1))
