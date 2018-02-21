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
