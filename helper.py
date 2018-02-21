import numpy as np
import random

# function maps each word to an index
def get_idx(char_data):
    word_to_ix = {}
    for word in char_data:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    return word_to_ix


def prepare_data(data_nums, is_gpu):
    if is_gpu:
        return Variable(torch.LongTensor(data_nums).cuda())
    else:
        return Variable(torch.LongTensor(data_nums))

# returns prediction based on probabilites
def flip_coin(probabilities):
    stacked_probs = np.cumsum(probabilities)
    stacked_probs = stacked_probs - min(stacked_probs)
    stacked_probs = stacked_probs/max(stacked_probs)
    rand_int = random.random()
    dist = abs(stacked_probs - rand_int)
    return np.argmin(dist)
