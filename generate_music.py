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
from LSTM import *
from helper import *

def generate(model, vocab, primer, predict_len, T, use_gpu):
    vocab_size = len(vocab)

    model.bs = 1

    hidden = model.init_hidden()
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
        predicted.append(flip_coin(soft_out))
        inp = prepare_data([predicted[-1]], use_gpu)
    strlist = [vocab.keys()[vocab.values().index(pred)] for pred in predicted]
    return ''.join(strlist)
