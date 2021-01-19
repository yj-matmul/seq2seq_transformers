import numpy as np
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


ACT2FN = {'gelu': nn.GELU(), 'relu': nn.ReLU()}


# paul-hyun, graykode, huggingface 참고한 transformer 구현 코드
class TransformerConfig():
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 hidden_size=512,
                 num_hidden_layers=6,
                 num_attention_heads=8,
                 hidden_act='relu',
                 feed_forward_size=2048,
                 padding_idx=0,
                 hidden_dropout_prob=0.1,
                 attention_dropout_prob=0.1,
                 max_seq_length=512,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12):
        self.encoder_vocab_size = src_vocab_size
        self.decoder_vocab_size = tgt_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.feed_forward_size = feed_forward_size
        self.padding_idx = padding_idx
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


class EncoderEmbedding(nn.Module):
    def __init__(self, config):
        super(EncoderEmbedding, self).__init__()
        self.word_embeddings = nn.Embedding(config.src_vocab_size, config.hidden_size)
        position_table = get_positional_encoding_table(config.max_seq_length + 1, config.hidden_size)
        self.position_embeddings = nn.Embedding.from_pretrained(position_table, freeze=True)

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids):  # inputs : [batch_size, seq_length]
        # [batch_size, seq_length]
        position_ids = torch.arange(input_ids.size()[1], dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids) + 1
        position_mask = input_ids.eq(self.padding_idx)  # empty token
        position_ids.masked_fill_(position_mask, 0)

        # [batch_size, seq_length, hidden_size]
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        embeddings = self.layer_norm(embeddings)
        return embeddings


class DecoderEmbedding(nn.Module):
    def __init__(self, config):
        super(DecoderEmbedding, self).__init__()
        self.word_embeddings = nn.Embedding(config.tgt_vocab_size, config.hidden_size)
        position_table = get_positional_encoding_table(config.max_seq_length + 1, config.hidden_size)
        self.position_embeddings = nn.Embedding.from_pretrained(position_table, freeze=True)

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids):  # inputs : [batch_size, seq_length]
        # [batch_size, seq_length]
        position_ids = torch.arange(input_ids.size()[1], dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids) + 1
        position_mask = input_ids.eq(self.padding_idx)  # empty token
        position_ids.masked_fill_(position_mask, 0)

        # [batch_size, seq_length, hidden_size]
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        embeddings = self.layer_norm(embeddings)
        return embeddings


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
        self.dropout = nn.Dropout(config.attention_dropout_prob)

    def transpose_for_attention_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_head, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, attention_mask):
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
        return hidden_states, attention_probs


class FeedForwardNet(nn.Module):
    def __init__(self, config):
        super(FeedForwardNet, self).__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.feed_forward_size)
        self.ffn_act = ACT2FN[config.hidden_act]
        self.dense2 = nn.Linear(config.feed_forward_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs):
        # [batch_size, seq_length, feed_forward_size]
        hidden_states = self.dense1(inputs)
        hidden_states = self.ffn_act(hidden_states)
        # [batch_size, seq_length, hidden_size]
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = FeedForwardNet(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        # self_attention_outputs : [batch_size, seq_length, hidden_size],
        # attention_probs : [batch_size, num_attention_heads, seq_length, seq_length]
        self_attention_outputs, attention_probs = self.self_attention(hidden_states, hidden_states,
                                                                      hidden_states, attention_mask)
        self_attention_outputs = self.layer_norm1(self_attention_outputs + inputs)

        # [batch_size, seq_length, hidden_size]
        ffn_outputs = self.ffn(self_attention_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + self_attention_outputs)
        return ffn_outputs, attention_probs


class Encoders(nn.Module):
    def __init__(self, config):
        super(Encoders, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):   #
        # hidden_states : [batch_size, seq_length, hidden_size]
        # attention_mask : [batch_size, seq_length, seq_length]
        attention_probs = []
        for layer in self.layers:
            hidden_states, attention_prob = layer(hidden_states, attention_mask)
            attention_probs.append(attention_prob)
        return hidden_states, attention_probs


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.masked_self_attention = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder_decoder_attention = MultiHeadAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = FeedForwardNet(config)
        self.layer_norm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, decoder_hidden_states, encoder_hidden_states, look_ahead_attention_mask, attention_mask):
        # masked_self_attention_outputs : [batch_size, seq_length, hidden_size],
        # self_attention_probs : [batch_size, num_attention_heads, seq_length, seq_length]
        masked_self_attention_outputs, self_attention_probs = self.masked_self_attention(decoder_hidden_states,
                                                                                         decoder_hidden_states,
                                                                                         decoder_hidden_states,
                                                                                         look_ahead_attention_mask)
        masked_self_attention_outputs = self.layer_norm1(masked_self_attention_outputs + decoder_hidden_states)

        encoder_decoder_attention_outpus, encoder_decoder_attention_probs = self.encoder_decoder_attention(decoder_hidden_states,
                                                                                                           encoder_hidden_states,
                                                                                                           encoder_hidden_states)





if __name__ == '__main__':
    config = TransformerConfig(100,
                               100)
    inputs = torch.tensor([[3091, 3604,  206, 3958, 3760, 3590,    0,    0],
                           [ 212, 3605,   53, 3832, 3596, 3682, 3760, 3590]])

    print(inputs.eq(0).unsqueeze(1).size())
    print(inputs.eq(0).unsqueeze(1))
