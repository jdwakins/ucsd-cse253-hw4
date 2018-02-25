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
seq_incr_perc = None
batch_size = 10
num_epochs = 2

predict_length = 100
primer = start_char * 5
temperature = 1

model = LSTM_Mod2(100, vocab, batch_size, seq_len, data, end_char,
                  start_char, pad_char, is_gpu=use_gpu)
train_loss, val_loss = model.train(vocab, seq_len, batch_size,
                                   num_epochs, use_gpu, seq_incr_perc, center_examples=True)
words = model.daydream(temperature, use_gpu, primer)
print(words)
