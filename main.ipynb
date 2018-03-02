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

# input data
start_char = '$'
end_char = '%'
pad_char = '`'
data = clean_up_data('input.txt', start_char, end_char)

vocab = get_idx(data + start_char + end_char + pad_char)
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

torch.save(model.state_dict(), 'model.pt')

with open('log.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(train_loss)
    writer.writerow(val_loss)
# plt.plot(range(len(val_loss)), val_loss)
# plt.plot(range(len(train_loss)), train_loss)
#
# plt.show()
# words = model.daydream(primer, temperature, predict_len=1000)
# print(words)
