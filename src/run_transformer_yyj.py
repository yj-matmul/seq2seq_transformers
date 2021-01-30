import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformer_yyj import Transformer, TransformerConfig
from torch import optim
import sentencepiece as spm
import numpy as np


# this is only used with version of sentencepiece tokenizer
def make_feature(src_list, trg_list, tokenizer, config):
    sos, eos = [1], [2]
    encoder_features = []
    decoder_features = []
    trg_features = []
    for i in range(len(src_list)):
        src_text = src_list[i]
        trg_text = trg_list[i]
        encoder_feature = tokenizer.encode_as_ids(src_text)
        decoder_feature = sos + tokenizer.encode_as_ids(trg_text) + eos
        trg_feature = tokenizer.encode_as_ids(trg_text) + eos
        encoder_feature += [0] * (config.max_seq_length - len(encoder_feature))
        decoder_feature += [0] * (config.max_seq_length - len(decoder_feature))
        trg_feature += [0] * (config.max_seq_length - len(trg_feature))
        encoder_features.append(encoder_feature)
        decoder_features.append(decoder_feature)
        trg_features.append(trg_feature)
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
    # inputs = torch.randint(vocab_size, (100, 8), dtype=torch.float, device=config.device)
    # labels = torch.randint(vocab_size, (100, 8), dtype=torch.float, device=config.device)
    sp = spm.SentencePieceProcessor()
    vocab_file = "spm_unigram.model"
    sp.load(vocab_file)
    src_vocab_size = sp.vocab_size()
    trg_vocab_size = sp.vocab_size()

    config = TransformerConfig(src_vocab_size=src_vocab_size,
                               trg_vocab_size=trg_vocab_size,
                               device='cuda')
    model = Transformer(config).to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-6)

    total_epoch = 10
    dataset = CustomDataset(config, sp)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model.train()
    for epoch  in range(total_epoch):
        for iteration, datas in enumerate(dataloader):
            encoder_inputs, decoder_inputs, targets = datas
            optimizer.zero_grad()
            outputs, _ = model(encoder_inputs, decoder_inputs)
            outputs = outputs.view(-1, trg_vocab_size)
            target = targets.view(-1)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            if (iteration + 1) % 300 == 0:
                print('Iteration: %3d ' % (iteration + 1), '\tCost: {:.5f}'.format(loss))
        # if (epoch + 1) % 10 == 0:
        print('Epoch: %3d ' % (epoch + 1), '\tCost: {:.5f}'.format(loss))

    model.eval()
    sample_encoder_input = ['나는 안녕하세요 1+1 이벤트 진행 중이다, 가격 1300원이야.']
    sample_decoder_input = ['']
    sample_encoder_input, sample_decoder_input, _ = make_feature(sample_encoder_input, sample_decoder_input, sp, config)
    predict, _ = model(sample_encoder_input, sample_decoder_input)
    predict = torch.max(predict, dim=-1)[-1].long().to('cpu')
    mask = predict.eq(0)
    predict.masked_fill_(mask, 2)
    print(predict)
    predict = predict.squeeze(0).numpy()
    print('predict size:', predict.shape)
    predict = list(map(int, predict))

    print(predict)
    print(sp.DecodeIds(predict))
