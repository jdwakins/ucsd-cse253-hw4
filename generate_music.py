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
<<<<<<< HEAD
from LSTM.py import *
from helper import *

model = LSTM_Mod2(100, vocab_size, 1, seq_len, is_gpu=use_gpu)

def generate(model, data, primer, predict_len=100, T, cuda):
    hidden = model.init_hidden()
    data_nums = get_idx(data)
    primer_input = [data_nums[char] for char in primer]
    primer_input = prepare_data(primer_input, cuda)

    for p in primer_input:
        _ = model(p)
=======
from LSTM import *
from helper import *

def generate(model, vocab, primer, predict_len, T, use_gpu):
    vocab_size = len(vocab)

    model.bs = 1

    model.init_hidden()
    primer_input = [vocab[char] for char in primer]

    model.seq_len = len(primer_input)
    # build hidden layer
    _ = model(prepare_data(primer_input[:-1], use_gpu))

    inp = prepare_data([primer_input[-1]], use_gpu)

    model.seq_len = 1
    predicted=list(primer_input)
    for p in range(predict_len):
        output = model(inp)
        soft_out = custom_softmax(output.data.squeeze(), T)
        predicted.append(flip_coin(soft_out, use_gpu))
        inp = prepare_data([predicted[-1]], use_gpu)
    strlist = [vocab.keys()[vocab.values().index(pred)] for pred in predicted]
    return ''.join(strlist)
>>>>>>> master
