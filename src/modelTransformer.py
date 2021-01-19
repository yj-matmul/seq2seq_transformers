import numpy as np
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# paul-hyun, graykode, huggingface 참고한 transformer 구현 코드
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
        self.attention_dropout_prob = attention_dropout_prob
        self.max_position_embeddings = max_seq_length
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


# sinusoid positional encoding
def get_positional_encoding_table(seq_length, hidden_size):
    def get_angle(position, idx_hidden):
        return position / np.power(10000, 2 * (idx_hidden // 2) / hidden_size)
    def get_position_angle_vector(position):
        return [get_angle(position, idx_hidden) for idx_hidden in range(hidden_size)]

    sinusoid_table = np.array([get_position_angle_vector(idx_seq) for idx_seq in range(seq_length)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.num_attention_head = config.num_attention_head
        self.attention_head_size = config.hidden_size // config.num_attention_head
        self.all_head_size = self.num_attention_head * self.attention_head_size

        self.weight_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.weight_key = nn.Linear(config.hidden_size, self.all_head_size)
        self.weight_value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.attention_dropout_prob)

    def transpose_for_attention_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_head, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, attention_mask):
        # [batch_size, seq_length, hidden_size]
        residual = query

        # [batch_size, seq_length, hidden_size]
        mixed_query_layer = self.weight_query(query)
        mixed_key_layer = self.weight_query(key)
        mixed_value_layer = self.weight_value(value)

        # [batch_size, num_attention_heads, seq_length, attention_head_size]
        query_layer = self.transpose_for_attention_scores(mixed_query_layer)
        key_layer = self.transpose_for_attention_scores(mixed_key_layer)
        value_layer = self.transpose_for_attention_scores(mixed_value_layer)

        # [batch_size, num_attention_heads, seq_length, seq_length]
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_attention_head, 1, 1)

        # [batch_size, num_attention_heads, seq_length, seq_length]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores.masked_fill_(attention_mask, -1e9)

        # [batch_size, num_attention_heads, seq_length, seq_length]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # [batch_size, num_attention_heads, seq_length, attention_head_size]
        context_layer = torch.matmul(attention_probs, value_layer).permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

        # [batch_size, seq_length, hidden_size]
        context_layer = context_layer.view(new_context_layer_shape)

        # [batch_size, seq_length, hidden_size]
        hidden_states = self.dense(context_layer)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + residual)
        return hidden_states, attention_probs




if __name__ == '__main__':
    config = TransformerConfig(100,
                               100)
    inputs = torch.tensor([[3091, 3604,  206, 3958, 3760, 3590,    0,    0],
                           [ 212, 3605,   53, 3832, 3596, 3682, 3760, 3590]])

    print(inputs.eq(0).unsqueeze(1).size())
    print(inputs.eq(0).unsqueeze(1))
