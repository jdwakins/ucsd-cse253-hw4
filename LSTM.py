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
# for feature evaluation - pick one neuron

# use_gpu = torch.cuda.is_available()
class LSTM_Mod2(nn.Module):
# class LSTM_Mod2():
    def __init__(self, hidden_dim, vocab_size, bs,
                 seq_len, data, end_char, start_char, pad_char,
                 is_gpu=False):
        super(LSTM_Mod2, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(1, hidden_dim, 1)
        # The linear layer maps from hidden state space to target space
        # target space = vocab size, or number of unique characters in daa
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.batch_size = bs
        self.seq_len = seq_len
        self.is_gpu = is_gpu
        self.hidden = self.__init_hidden()


        self.data = data
        self.end_char = end_char
        self.start_char = start_char
        self.pad_char = pad_char
        self.__generate_discrete_examples()

    # Rather than having the data as one long string it is an array of strings.
    def __generate_discrete_examples(self):
        self.examples = []
        split = self.data.split(self.end_char)
        for ex in split:
            self.examples.append(ex + self.end_char)


    def __get_new_sequence_length(self, old, incrementer):
        return int(old + old * incrementer)

    def __pad_sequence(self, examples, sequence_length):

        for i, example in enumerate(examples):
            if len(example) < sequence_length:
                examples[i] = example + self.pad_char * (sequence_length - len(example))
        return examples

    def __init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.is_gpu:
            self.hidden = (Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()))
        else:
            self.hidden =  (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def __forward(self, sentence):
        outputs=[]
        # input sentence is shape: sequence x batch x 1
        output, self.hidden = self.lstm(sentence.float().view(-1, self.batch_size, 1), self.hidden)
        outputs = self.linear(output)
        return outputs

    def train(self, vocab_idx, seq_len, batch_size, epochs,
              use_gpu, seq_incr_perc):
        vocab_size = len(vocab_idx)

        # slice data into trianing and testing (could do this much better)
        val_split = 0.9
        slice_ind = int((len(self.examples) * val_split))
        training_data = self.examples[:slice_ind]
        val_data = self.examples[slice_ind:]

        # turn training and validation data from characters to numbers
        # training_nums = [vocab_idx[char] for char in training_data]
        # val_nums = [vocab_idx[char] for char in val_data]

        np.random.seed(1)

        if use_gpu:
            self.cuda()

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.01)

        # train self
        train_loss_vec = []
        val_loss_vec=[]
        for epoch in range(epochs):
            #get random slice
            possible_example_indices = range(len(self.examples))
            # after going through all of a , will have gone through all possible 30
            # character slices
            total = 0
            correct = 0
            iterate = 0

            while len(possible_example_indices) > 0:
                self.batch_size = batch_size

                example_indices = random.sample(possible_example_indices, batch_size)
                print(example_indices)
                rand_starts = [np.random.randint(len(self.examples[i])) for i in example_indices]

                rand_slices = [self.examples[index][rand_starts[i]: rand_starts[i] + seq_len] for i, index in enumerate(example_indices)]
                targets = [self.examples[index][rand_starts[i] + 1: rand_starts[i] + seq_len + 1] for i, index in enumerate(example_indices)]
                self.__pad_sequence(rand_slices, seq_len)
                print(rand_slices)

                rand_slices = [[vocab_idx[c] for c in ex] for ex in rand_slices]
                targets = [[vocab_idx[c] for c in ex] for ex in targets]
                print(rand_slices)
                # print(targets)




                # get random slice, and the targets that correspond to that slice
                rand_slice = [training_nums[idx : idx + seq_len] for idx in idxs]
                rand_slice = np.array(rand_slice).T
                targets = [training_nums[idx + 1:idx+(seq_len+1)] for idx in idxs]
                targets = np.array(targets).T

                # for idx in idxs:
                [possible_example_indices.remove(idx) for idx in idxs]

                # prepare data and targets for self
                rand_slice = add_cuda_to_variable(rand_slice, use_gpu)
                targets = add_cuda_to_variable(targets, use_gpu)
                # Pytorch accumulates gradients. We need to clear them out before each instance
                self.zero_grad()

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                self.__init_hidden()
                # From TA:
                # another option is to feed sequences sequentially and let hidden state continue
                # could feed whole sequence, and then would kill hidden state

                # Run our __forward pass.
                outputs = self.__forward(rand_slice)
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

                    val_inputs = add_cuda_to_variable(val_inputs, use_gpu)
                    val_targets = add_cuda_to_variable(val_targets, use_gpu)
                    self.__init_hidden()
                    outputs_val = self.__forward(val_inputs)
                    val_loss = 0
                    for bat in range(batch_size):
                        val_loss += loss_function(outputs_val[:,1,:], val_targets[:,1,:].squeeze(1))
                    val_loss_vec.append(val_loss.data[0]/batch_size)
                    print('Validataion Loss ' + str(val_loss.data[0]/batch_size))
                iterate += 1
            print('Completed Epoch ' + str(epoch))
        return train_loss_vec, val_loss_vec

    def daydream(self, vocab, primer, predict_len, T, use_gpu):
        vocab_size = len(vocab)

        self.batch_size = 1

        self.__init_hidden()
        primer_input = [vocab[char] for char in primer]

        self.seq_len = len(primer_input)
        # build hidden layer
        _ = self(add_cuda_to_variable(primer_input[:-1], use_gpu))

        inp = add_cuda_to_variable([primer_input[-1]], use_gpu)

        self.seq_len = 1
        predicted=list(primer_input)
        for p in range(predict_len):
            output = self(inp)
            soft_out = custom_softmax(output.data.squeeze(), T)
            predicted.append(flip_coin(soft_out, use_gpu))
            inp = add_cuda_to_variable([predicted[-1]], use_gpu)
        strlist = [vocab.keys()[vocab.values().index(pred)] for pred in predicted]
        return ''.join(strlist)
