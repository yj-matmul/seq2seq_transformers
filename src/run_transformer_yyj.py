import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformer_yyj import Transformer, TransformerConfig
from torch import optim
import sentencepiece as spm
import numpy as np


# this is only used with version of sentencepiece tokenizer
def make_feature(src_list, trg_list, tokenizer, config):
    bos, eos, pad = [sp.piece_to_id('[BOS]')], [sp.piece_to_id('[EOS]')], [sp.piece_to_id('[PAD]')]
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
        max_len = max(max_len, len(trg_feature))
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
    # inputs = torch.randint(vocab_size, (100, 8), dtype=torch.float, device=config.device)
    # labels = torch.randint(vocab_size, (100, 8), dtype=torch.float, device=config.device)
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
    model = Transformer(config).to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    total_epoch = 10
    dataset = CustomDataset(config, sp)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(total_epoch):
        for iteration, datas in enumerate(dataloader):
            encoder_inputs, decoder_inputs, targets = datas
            optimizer.zero_grad()
            logits, _ = model(encoder_inputs, decoder_inputs)
            logits = logits.contiguous().view(-1, trg_vocab_size)
            targets = targets.contiguous().view(-1)
            indices = targets.nonzero().squeeze(1)
            logits = logits.index_select(0, indices)
            targets = targets.index_select(0, indices)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            # if (iteration + 1) % 50 == 0:
            #     print('Iteration: %3d \t' % (iteration + 1), 'Cost: {:.5f}'.format(loss))
        # break
        # if (epoch + 1) % 5 == 0:
        print('Epoch: %3d \t' % (epoch + 1), 'Cost: {:.5f}'.format(loss))
        if (epoch + 1) % 100 == 0:
            model_path = './model_weight/transformer_%d' % (epoch + 1)
            torch.save(model.state_dict(), model_path)

    # model.load_state_dict(torch.load('./model_weight/transformer_500'))
    model.eval()
    sample_encoder_input = ['나는 안녕하세요 1+1 이벤트 진행 중이다, 가격 1300원이야.',
                            '가랑비에 옷 젖는 줄 모른다.',
                            '고객님, 현재 짜파게티는 1+1 상품으로 이벤트가 진행중이니 살펴보고 가세요.']
    sample_decoder_input = [''] * len(sample_encoder_input)
    sample_encoder_input, sample_decoder_input, _ = make_feature(sample_encoder_input, sample_decoder_input, sp, config)
    predicts, _ = model(sample_encoder_input, sample_decoder_input)
    print(predicts.size())
    predicts = torch.max(predicts, dim=-1)[-1].long().to('cpu')
    print(predicts.size())

    for predict in predicts:
        predict = predict.numpy()
        print('predict size:', predict.shape)
        predict = list(map(int, predict))
        print(predict)
        print('예측 결과:', sp.DecodeIds(predict))
