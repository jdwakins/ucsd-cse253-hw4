import numpy as np
import random
from torch.autograd import Variable
import torch
import pdb

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
        tensor = tensor.unsqueeze_(0)
    tensor = tensor.unsqueeze_(2)
    if is_gpu:
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

# returns prediction based on probabilites
def flip_coin(probabilities):
    stacked_probs = np.cumsum(probabilities)
    # stacked_probs = stacked_probs - min(stacked_probs)
    # stacked_probs = stacked_probs/max(stacked_probs)
    rand_int = random.random()
    dist = abs(stacked_probs.numpy() - rand_int)
    return np.argmin(dist)

def custom_softmax(output, T):
    return torch.exp(torch.div(output, T)) / torch.sum(torch.exp(torch.div(output, T)))
