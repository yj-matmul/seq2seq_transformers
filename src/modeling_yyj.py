import json
import logging
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)

CONFIG_NAME = ''


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {'gelu': gelu, 'relu': nn.ReLU, 'swish': swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x -u).pow(2).mean(-1, keepdim=True)
        x = (x- u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        # [vocab_size, hidden_size]
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        # [max_position_embeddings, hidden_size]
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # [type_vocab_size, hidden_size]
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        # input_ids shape is [batch_size, seq_length]
        seq_length = input_ids.size(1)
        # position_ids shape becomes [batch_size, seq_length]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        # token_type_ids shape becomes [batch_size, seq_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        # [batch_size, seq_length, hidden_size]
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        # embeddings shape is [batch_size, seq_length, hidden_size]
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # [hidden_size, hidden_size]
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        # [batch_size, seq_length, hidden_size]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # [batch_size, num_attention_heads, seq_length, attention_head_size]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # [batch_size, num_attention_heads, seq_length, seq_length]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores += attention_mask

        # [batch_size, num_attention_heads, seq_length, seq_length]
        attention_probs = nn.Softmax(dim=1)(attention_scores)

        # [batch_size, num_attention_heads, seq_length, attention_head_size]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [batch_size, seq_length, num_attention_heads, attention_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # [batch_size, seq_length, hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    def __init__(self, config):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class EncoderAttention(nn.Module):
    def __init__(self, config):
        super(EncoderAttention, self).__init__()
        self.self_attention = SelfAttention(config)
        self.self_output = SelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_attention = self.self_attention(input_tensor, attention_mask)
        attention_output = self.self_output(self_attention, input_tensor)
        return attention_output


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.feedforward_size)
        self.feedforward_act = config.hidden_act
        self.dense2 = nn.Linear(config.feedforward_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.feedforward_act(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.encoder_attention = EncoderAttention(config)
        self.feedforward = FeedForward(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.encoder_attention(hidden_states, attention_mask)
        layer_output = self.feedforward(attention_output, attention_output)
        return layer_output


class AllEncoders(nn.Module):
    def __init__(self, config):
        super(AllEncoders, self).__init__()
        layer = EncoderLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)] for _ in range(config.num_hidden_layers))

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.future_mask = torch.empty(0)

