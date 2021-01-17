import numpy as np
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# paul-hyun 참고한 transformer 구현 코드

# sinusoid positional encoding
def get_sinusoid_encoding_table(seq_length, hidden_size):
    def cal_angle(position, idx_hidden):
        return position / np.power(10000, 2 * (idx_hidden // 2) / hidden_size)
    def get_position_angle_vector(position):
        return [cal_angle(position, idx_hidden) for idx_hidden in range(hidden_size)]

    sinusoid_table = np.array([get_position_angle_vector(idx_seq) for idx_seq in range(seq_length)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return sinusoid_table
