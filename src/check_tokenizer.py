import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from transformer_yyj import Transformer, TransformerConfig


# this is only used with version of sentencepiece tokenizer
def make_feature(src_list, trg_list, tokenizer, config):
    bos = [tokenizer.piece_to_id('[BOS]')]
    eos = [tokenizer.piece_to_id('[EOS]')]
    pad = [tokenizer.piece_to_id('[PAD]')]
    encoder_features = []
    decoder_features = []
    trg_features = []
    max_len = 0
    for i in range(len(src_list)):
        src_text = src_list[i]
        trg_text = trg_list[i]
        encoder_feature = tokenizer.encode_as_ids(src_text)
        decoder_feature = bos + tokenizer.encode_as_ids(trg_text)
        trg_feature = tokenizer.encode_as_ids(trg_text) + eos
        max_len = max(max_len, len(trg_list[i]))
        encoder_feature += pad * (config.max_seq_length - len(encoder_feature))
        decoder_feature += pad * (config.max_seq_length - len(decoder_feature))
        trg_feature += pad * (config.max_seq_length - len(trg_feature))
        encoder_features.append(encoder_feature)
        decoder_features.append(decoder_feature)
        trg_features.append(trg_feature)
    print(max_len)
    encoder_features = torch.LongTensor(encoder_features).to(config.device)
    decoder_features = torch.LongTensor(decoder_features).to(config.device)
    trg_features = torch.LongTensor(trg_features).to(config.device)
    return encoder_features, decoder_features, trg_features


class CustomDataset(Dataset):
    def __init__(self, config, lm):
        src_file_path = 'D:/Storage/sinc/tts_script/data_filtering/철자표기.txt'
        trg_file_path = 'D:/Storage/sinc/tts_script/data_filtering/발음표기.txt'
        with open(src_file_path, 'r', encoding='utf8') as f:
            src_lines = list(map(lambda x: x.strip('\n'), f.readlines()))
        with open(trg_file_path, 'r', encoding='utf8') as f:
            trg_lines = list(map(lambda x: x.strip('\n'), f.readlines()))
        self.encoder_input, self.decoder_input, self.target = make_feature(src_lines, trg_lines, lm, config)

    def __len__(self):
        return len(self.encoder_input)

    def __getitem__(self, idx):
        x = self.encoder_input[idx]
        y = self.decoder_input[idx]
        z = self.target[idx]
        return x, y, z


if __name__ == '__main__':
    vocab_file = "./tokenizer/spm_unigram_8000.model"
    sp = spm.SentencePieceProcessor()
    sp.load(vocab_file)
    src_vocab_size = sp.vocab_size()
    trg_vocab_size = sp.vocab_size()

    config = TransformerConfig(src_vocab_size=src_vocab_size,
                               trg_vocab_size=trg_vocab_size,
                               device='cuda',
                               hidden_size=256,
                               num_attn_head=4,
                               feed_forward_size=1024,
                               max_seq_length=64,
                               share_embeddings=True)

    dataset = CustomDataset(config, sp)
    x, y, _ = dataset.__getitem__(2036)
    x, y = x.to('cpu').numpy(), y.to('cpu').numpy()
    x, y = list(map(int, x)), list(map(int, y))
    print()
    print(sp.IdToPiece(x))
    print(sp.IdToPiece(y))
