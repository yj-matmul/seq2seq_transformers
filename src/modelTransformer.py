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
                 trg_vocab_size,
                 hidden_size=128,
                 num_hidden_layers=12,
                 num_attention_head=8,
                 hidden_act='relu',
                 device='cuda',
                 feed_forward_size=1100,
                 padding_idx=0,
                 share_embeddings=False,
                 hidden_dropout_prob=0.1,
                 attention_dropout_prob=0.1,
                 max_seq_length=512,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12):
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_head = num_attention_head
        self.hidden_act = hidden_act
        self.device = device
        self.feed_forward_size = feed_forward_size
        self.padding_idx = padding_idx
        self.share_embeddings = share_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.max_seq_length = max_seq_length
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


def get_attention_mask(input_ids, padding_idx):     # input_ids : [batch_size, seq_length]
    seq_length = input_ids.size()[1]
    # attention_mask : [batch_size, seq_length, seq_length]
    attention_mask = input_ids.eq(padding_idx).unsqueeze(1).expand(-1, seq_length, seq_length)
    return attention_mask


def get_look_ahead_attention_mask(input_ids):
    seq_length = input_ids.size()[1]
    look_ahead_attention_mask = torch.ones_like(input_ids).unsqueeze(1).expand(-1, seq_length, seq_length)
    look_ahead_attention_mask = look_ahead_attention_mask.triu(diagonal=1)
    return look_ahead_attention_mask


class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.padding_idx = config.padding_idx
        self.share_embeddings = config.share_embeddings
        self.max_seq_length = config.max_seq_length
        self.device = config.device

        self.src_word_embeddings = nn.Embedding(config.src_vocab_size, config.hidden_size).to(self.device)
        self.trg_word_embeddings = nn.Embedding(config.trg_vocab_size, config.hidden_size).to(self.device)
        position_table = get_positional_encoding_table(config.max_seq_length + 1, config.hidden_size)
        self.position_encodings = nn.Embedding.from_pretrained(position_table, freeze=True).to(self.device)

    def forward(self, encoder_inputs, decoder_inputs):  # [batch_size, seq_length]
        # for change data type
        encoder_inputs = encoder_inputs.type(torch.LongTensor).to(self.device)
        decoder_inputs = decoder_inputs.type(torch.LongTensor).to(self.device)

        # encoder, decoder position encoding
        position_ids = torch.arange(encoder_inputs.size()[1], dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand_as(encoder_inputs) + 1
        position_mask = encoder_inputs.eq(self.padding_idx)
        # encoder_position_ids : [batch_size, seq_length]
        encoder_position_ids = torch.masked_fill(position_ids, position_mask, self.padding_idx)
        position_mask = decoder_inputs.eq(self.padding_idx)
        # decoder_position_ids : [batch_size, seq_length]
        decoder_position_ids = torch.masked_fill(position_ids, position_mask, self.padding_idx)

        # src_word_embeddings, src_word_embeddings : [batch_size, seq_length, hidden_size]
        src_word_embeddings = self.src_word_embeddings(encoder_inputs)
        if self.share_embeddings:
            trg_word_embeddings = src_word_embeddings
        else:
            trg_word_embeddings = self.trg_word_embeddings(decoder_inputs)

        encoder_embeddings = src_word_embeddings + self.position_encodings(encoder_position_ids)
        decoder_embeddings = trg_word_embeddings + self.position_encodings(decoder_position_ids)
        return encoder_embeddings, decoder_embeddings


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.num_attention_head = config.num_attention_head
        self.attention_head_size = config.hidden_size // config.num_attention_head
        self.all_head_size = self.num_attention_head * self.attention_head_size
        self.device = config.device

        self.weight_query = nn.Linear(config.hidden_size, self.all_head_size).to(self.device)
        self.weight_key = nn.Linear(config.hidden_size, self.all_head_size).to(self.device)
        self.weight_value = nn.Linear(config.hidden_size, self.all_head_size).to(self.device)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size).to(self.device)
        self.dropout = nn.Dropout(config.attention_dropout_prob).to(self.device)

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
        self.device = config.device

        self.dense1 = nn.Linear(config.hidden_size, config.feed_forward_size).to(self.device)
        self.ffn_act = ACT2FN[config.hidden_act]
        self.dense2 = nn.Linear(config.feed_forward_size, config.hidden_size).to(self.device)
        self.dropout = nn.Dropout(config.hidden_dropout_prob).to(self.device)

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

    def forward(self, encoder_inputs, attention_mask):
        # self_attention_outputs : [batch_size, seq_length, hidden_size],
        # attention_probs : [batch_size, num_attention_heads, seq_length, seq_length]
        self_attention_outputs, attention_probs = self.self_attention(encoder_inputs, encoder_inputs,
                                                                      encoder_inputs, attention_mask)
        self_attention_outputs = self.layer_norm1(self_attention_outputs + encoder_inputs)

        # [batch_size, seq_length, hidden_size]
        ffn_outputs = self.ffn(self_attention_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + self_attention_outputs)
        return ffn_outputs, attention_probs


class Encoders(nn.Module):
    def __init__(self, config):
        super(Encoders, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        # hidden_states : [batch_size, seq_length, hidden_size]
        # attention_mask : [batch_size, seq_length, seq_length]

        attention_probs = []
        for i,layer in enumerate(self.layers):
            hidden_states, attention_prob = layer(hidden_states, attention_mask)
            attention_probs.append(attention_prob)
            print('Encoders', i)
        return hidden_states, attention_probs


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder_decoder_attention = MultiHeadAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = FeedForwardNet(config)
        self.layer_norm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, decoder_inputs, encoder_outputs, look_ahead_attention_mask, attention_mask):
        # self_attention_outputs : [batch_size, seq_length, hidden_size],
        # self_attention_probs : [batch_size, num_attention_heads, seq_length, seq_length]
        #print('DecoderLayer', decoder_inputs.size(), encoder_outputs.size())
        self_attention_outputs, self_attention_probs = self.self_attention(decoder_inputs, decoder_inputs,
                                                                           decoder_inputs, look_ahead_attention_mask)
        self_attention_outputs = self.layer_norm1(self_attention_outputs + decoder_inputs)

        # attention_outputs : [batch_size, seq_length, hidden_size],
        # attention_probs : [batch_size, num_attention_heads, seq_length, seq_length]
        attention_outputs, attention_probs = self.encoder_decoder_attention(self_attention_outputs, encoder_outputs,
                                                                            encoder_outputs, attention_mask)
        attention_outputs = self.layer_norm2(attention_outputs + self_attention_outputs)

        # ffn_outputs : [batch_size, seq_length, hidden_size]
        ffn_outputs = self.ffn(attention_outputs)
        ffn_outputs = self.layer_norm3(ffn_outputs + attention_outputs)
        return ffn_outputs, self_attention_probs, attention_probs


class Decoders(nn.Module):
    def __init__(self, config):
        super(Decoders, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, encoder_outputs, decoder_inputs, look_ahead_attention_mask, attention_mask):
        # hidden_states : [batch_size, seq_length, hidden_size]
        hidden_states = decoder_inputs
        self_attention_probs, encoder_decoder_attention_probs = [], []
        for i, layer in enumerate(self.layers):
            hidden_states, self_attention_prob, attention_prob = layer(hidden_states, encoder_outputs,
                                                                       look_ahead_attention_mask, attention_mask)
            self_attention_probs.append(self_attention_prob)
            encoder_decoder_attention_probs.append(attention_prob)
            print('Decoders', i)
        return hidden_states, self_attention_probs, encoder_decoder_attention_probs


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.padding_idx = config.padding_idx
        self.share_embeddings = config.share_embeddings
        self.device = config.device

        self.embedding = Embedding(config)
        self.encoders = Encoders(config)
        self.decoders = Decoders(config)

    def forward(self, encoder_inputs, decoder_inputs):  # [batch_size, seq_length]
        # create_mask > [batch_size, seq_length, seq_length]
        encoder_attention_mask = get_attention_mask(encoder_inputs, self.padding_idx)
        decoder_attention_mask = get_attention_mask(decoder_inputs, self.padding_idx)
        look_ahead_attention_mask = get_look_ahead_attention_mask(decoder_inputs).to(self.device)
        look_ahead_attention_mask = torch.gt((decoder_attention_mask + look_ahead_attention_mask), 0)

        # embedding
        encoder_embeddings, decoder_embeddings = self.embedding(encoder_inputs, decoder_inputs)

        # encoder
        encoder_outputs, encoder_attention_probs = self.encoders(encoder_embeddings, encoder_attention_mask)

        # decoder
        decoder_outputs, masked_attention_probs, decoder_attention_probs \
            = self.decoders(encoder_outputs, decoder_embeddings, look_ahead_attention_mask, decoder_attention_mask)

        total_attention_probs = {}
        total_attention_probs['encoder_attention_probs'] = encoder_attention_probs
        total_attention_probs['masked_attention_probs'] = masked_attention_probs
        total_attention_probs['decoder_attention_probs'] = decoder_attention_probs
        return decoder_outputs, total_attention_probs


if __name__ == '__main__':
    config = TransformerConfig(5000,
                               5000)
    inputs = torch.randint(5000, (2, 8), dtype=torch.float, device=config.device)

    transformer = Transformer(config)
    transformer_outputs, attentions = transformer(inputs, inputs)
    print(transformer_outputs.size())
