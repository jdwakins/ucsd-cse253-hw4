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

from generate_music import *
from LSTM import *
from helper import *

def train_model(model, data, vocab_idx, seq_len, batch_size, epochs, use_gpu):
    vocab_size = len(vocab_idx)

    # slice data into trianing and testing (could do this much better)
    slice_ind = int(round(len(data)*.8))
    training_data = data[:slice_ind]
    val_data = data[slice_ind:]

    training_music, training_meta = get_music_meta(training_data)
    val_music, val_meta = get_music_meta(val_data)


    # turn training and validation data from characters to numbers
    training_nums_music_music = [[vocab_idx[char] for char in ex] for ex in training_music]
    training_nums_music_meta = [[vocab_idx[char] for char in ex] for ex in training_meta]
    val_nums_music = [[vocab_idx[char] for char in ex] for ex in val_music]
    val_nums_meta = [[vocab_idx[char] for char in ex] for ex in val_meta]

    np.random.seed(0)

    if use_gpu:
        model.cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # train model
    train_loss_vec = []
    val_loss_vec=[]
    for epoch in range(epochs):
        #get random slice
        sample = np.randint(len(training_music))
        music_slice = range(len(training_data) - (seq_len+1))
        # after going through all of a , will have gone through all possible 30
        # character slices
        total = 0
        correct = 0
        iterate = 0
        while len(a) > 30:
            model.bs = batch_size
            idxs = random.sample(a, batch_size)
            # get random slice, and the targets that correspond to that slice
            rand_slice_music = [training_nums_music[idx: idx + seq_len] for idx in idxs]
            rand_slice_music = np.array(rand_slice_music).T

            rand_slice_meta = [training_nums_meta[idx: idx + seq_len] for idx in idxs]
            rand_slice_meta = np.array(rand_slice_meta).T
            targets = [training_nums_music[idx + 1:idx+(seq_len+1)] for idx in idxs]
            targets = np.array(targets).T

            for idx in idxs:
                a.remove(idx)

            # prepare data and targets for model
            rand_slice_music = prepare_data(rand_slice_music, use_gpu)
            targets = prepare_data(targets, use_gpu)
            # Pytorch accumulates gradients. We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.init_hidden()
            # From TA:
            # another option is to feed sequences sequentially and let hidden state continue
            # could feed whole sequence, and then would kill hidden state

            # Run our forward pass.
            outputs = model(rand_slice_music)
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss=0
            for bat in range(batch_size):
                loss += loss_function(outputs[:,bat,:], targets[:,bat,:].squeeze(1))
            loss.backward()
            optimizer.step()

            # correct, total, running_accuracy = get_accuracy(outputs.squeeze(1), targets, correct, total)
            if iterate % 2000:
                print('Loss ' + str(loss.data[0]/batch_size))
                train_loss_vec.append(loss.data[0]/batch_size)

                idxs_val = random.sample(range(len(val_nums)-(seq_len+1)),batch_size)

                val_inputs = [val_nums[idx_v:idx_v + seq_len] for idx_v in idxs_val]
                val_inputs = np.array(val_inputs).T
                val_targets = [val_nums[idx_v+1: idx_v + seq_len+1] for idx_v in idxs_val]
                val_targets = np.array(val_targets).T

                val_inputs = prepare_data(val_inputs, use_gpu)
                val_targets = prepare_data(val_targets, use_gpu)
                model.init_hidden()
                outputs_val = model(val_inputs)
                val_loss=0
                for bat in range(batch_size):
                    val_loss += loss_function(outputs_val[:,1,:], val_targets[:,1,:].squeeze(1))
                val_loss_vec.append(val_loss.data[0]/batch_size)

                print('Validataion Loss ' + str(val_loss.data[0]/batch_size))
            iterate += 1
        print('Completed Epoch ' + str(epoch))
        print(generate(model, vocab_idx, '<start>', 100, 1, use_gpu))
    return train_loss_vec, val_loss_vec
