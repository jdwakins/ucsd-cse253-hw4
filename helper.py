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
def flip_coin(probabilities, is_gpu):
    stacked_probs = np.cumsum(probabilities)
    rand_int = random.random()
    if is_gpu:
        sp = stacked_probs[0].cpu().numpy()
    else:
        sp = stacked_probs.numpy()
    dist = abs(sp - rand_int)
    return np.argmin(dist)
# def flip_coin(probabilities):
#     stacked_probs = np.cumsum(probabilities)
#     # stacked_probs = stacked_probs - min(stacked_probs)
#     # stacked_probs = stacked_probs/max(stacked_probs)
#     rand_int = random.random()
#     dist = abs(stacked_probs.numpy() - rand_int)
#     pdb.set_trace()
#     return np.argmin(dist)

'''
This extracts the metadata and the music data and returns a list of tuples.
'''
def get_music_meta(data):
    output_music = []
    output_meta = []
    split_data = data.split('<end>')
    for sample in split_data:
        found_music = False
        music = []
        meta = []
        for line in sample.split('\n'):
            if (line.endswith('|') or line.endswith('::') or \
              line.endswith(':|') or line.endswith('|:')) or found_music or \
              len(line) > 70:
                music.append(line.rstrip())
                found_music = True
            elif len(line) > 0 and not found_music:
                meta.append(line.rstrip())
        music.append('<end>')
        output_music.append('\n'.join(music))
        # print(len(output_music))
        output_meta.append('\n'.join(meta))
    return output_music, output_meta



def custom_softmax(output, T):
    return torch.exp(torch.div(output, T)) / torch.sum(torch.exp(torch.div(output, T)))
