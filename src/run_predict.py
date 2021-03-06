import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from transformer_yyj import Transformer, TransformerConfig, Decoders
from utils import text2ids
from transformers import ElectraTokenizer


def predict(config, tokenizer, model, text):
    if type(text) is str:
        texts = [text]
    sos = tokenizer.convert_tokens_to_ids(['[CLS]'])  # use cls token as start of sequence
    eos = tokenizer.convert_tokens_to_ids(['[SEP]'])  # use sep token as end of sequence

    enc_ids = text2ids(texts, tokenizer, config, 'encoder')
    dec_ids = torch.zeros(len(enc_ids), config.dec_max_seq_length, dtype=torch.long, device=config.device)

    next_idx = sos[0]
    for i in range(config.dec_max_seq_length):
        if next_idx == eos[0]:
            dec_ids = dec_ids[:, 1:i][0]  # delete start, end token
            break
        dec_ids[0][i] = next_idx
        output, _ = model(enc_ids, dec_ids)
        output = output.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_idx = output[i].item()
    ids = dec_ids.to('cpu').numpy()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    text = tokenizer.convert_tokens_to_string(tokens)
    return text


if __name__ == '__main__':
    # we use pretrained tokenizer from monologg github
    tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
    vocab = tokenizer.get_vocab()
    src_vocab_size = len(vocab)
    trg_vocab_size = len(vocab)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = TransformerConfig(src_vocab_size=src_vocab_size,
                               trg_vocab_size=trg_vocab_size,
                               hidden_size=256,
                               num_hidden_layers=6,
                               num_attn_head=4,
                               hidden_act='gelu',
                               device=device,
                               feed_forward_size=1024,
                               padding_idx=0,
                               share_embeddings=True,
                               enc_max_seq_length=256,
                               dec_max_seq_length=256)

    model = Transformer(config).to(config.device)

    model_path = './model_weight/transformer_200'
    model.load_state_dict(torch.load(model_path))

    sentence = '네, 만오천원이 아닌 만이천칠백원에 해당하는 적립금만 적립됩니다.'

    result = predict(config, tokenizer, model, sentence)
    print('predict reuslt:', result)
