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

    return torch.FloatTensor(sinusoid_table)


class TransformerConfig():
    def __init__(self,
                 encoder_vocab_size,
                 decoder_vocab_size,
                 hidden_size=512,
                 num_hidden_layers=6,
                 num_attention_heads=8,
                 hidden_act='relu',
                 feed_forward_size=2048,
                 hidden_dropout_prob=0.1,
                 attention_dropout_prob=0.1,
                 max_seq_length=512,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12):
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.feed_forward_size = feed_forward_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_dropout_prob
        self.max_position_embeddings = max_seq_length
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

if __name__ == '__main__':
    config = TransformerConfig(100,
                               100)
    inputs = torch.tensor([[3091, 3604,  206, 3958, 3760, 3590,    0,    0],
                           [ 212, 3605,   53, 3832, 3596, 3682, 3760, 3590]])

    print(inputs.eq(0).unsqueeze(1).size())
    print(inputs.eq(0).unsqueeze(1))
    inputs.masked_fill_
