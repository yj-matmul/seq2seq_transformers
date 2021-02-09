import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformer_yyj import Transformer, TransformerConfig, Decoders, get_pad_mask, get_look_ahead_mask
from torch import optim
import sentencepiece as spm
from transformers import ElectraModel, ElectraTokenizer
import glob
from utils import text_normalization


# this is only used with version of sentencepiece tokenizer
def make_feature(src_list, trg_list, tokenizer, config):
    pad = tokenizer.convert_tokens_to_ids(['[PAD]'])
    sos = tokenizer.convert_tokens_to_ids(['[unused0]'])
    eos = tokenizer.convert_tokens_to_ids(['[unused1]'])
    encoder_features = []
    decoder_features = []
    trg_features = []
    max_len = 0
    for i in range(len(src_list)):
        # src_text = tokenizer.tokenize(src_list[i])
        # trg_text = tokenizer.tokenize(trg_list[i])
        src_text = tokenizer.tokenize(text_normalization(src_list[i]))
        trg_text = tokenizer.tokenize(text_normalization(trg_list[i]))
        # encoder_feature = tokenizer.convert_tokens_to_ids(src_text)
        encoder_feature = tokenizer.convert_tokens_to_ids(src_text)
        decoder_feature = sos + tokenizer.convert_tokens_to_ids(trg_text)
        trg_feature = tokenizer.convert_tokens_to_ids(trg_text) + eos
        max_len = max(max_len, len(trg_feature))
        encoder_feature += pad * (config.encoder_max_seq_length - len(encoder_feature))
        decoder_feature += pad * (config.decoder_max_seq_length - len(decoder_feature))
        trg_feature += pad * (config.decoder_max_seq_length - len(trg_feature))
        encoder_features.append(encoder_feature)
        decoder_features.append(decoder_feature)
        trg_features.append(trg_feature)
    print('decoder 최대 길이:', max_len)
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


class Spell2Pronunciation(nn.Module):
    def __init__(self, config):
        super(Spell2Pronunciation, self).__init__()
        # KoELECTRA-Small-v3
        self.encoders = ElectraModel.from_pretrained("monologg/koelectra-small-v3-discriminator")
        self.embedding = self.encoders.get_input_embeddings()
        self.embedding_projection = nn.Linear(128, config.hidden_size)
        self.decoders = Decoders(config)
        self.dense = nn.Linear(config.hidden_size, config.trg_vocab_size)

        self.padding_idx = config.padding_idx

    def forward(self, encoder_iuputs, decoder_inputs):
        decoder_attn_mask = get_pad_mask(decoder_inputs, decoder_inputs, self.padding_idx)
        look_ahead_attn_mask = get_look_ahead_mask(decoder_inputs)
        look_ahead_attn_mask = torch.gt((decoder_attn_mask + look_ahead_attn_mask), 0)
        decoder_attn_mask = get_pad_mask(decoder_inputs, encoder_iuputs, self.padding_idx)
        decoder_embeddings = self.embedding_projection(self.embedding(decoder_inputs))
        encoder_outputs = self.encoders(encoder_iuputs).last_hidden_state
        decoder_outputs, _, _ = self.decoders(encoder_outputs, decoder_embeddings, look_ahead_attn_mask, decoder_attn_mask)
        model_output = self.dense(decoder_outputs)
        return model_output


if __name__ == '__main__':
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    # inputs = torch.randint(vocab_size, (100, 8), dtype=torch.float, device=config.device)
    # labels = torch.randint(vocab_size, (100, 8), dtype=torch.float, device=config.device)
    src_vocab_size = 35000
    trg_vocab_size = 35000

    config = TransformerConfig(src_vocab_size=src_vocab_size,
                               trg_vocab_size=trg_vocab_size,
                               device='cuda',
                               hidden_size=256,
                               num_attn_head=4,
                               feed_forward_size=1024,
                               encoder_max_seq_length=512,
                               decoder_max_seq_length=64,
                               share_embeddings=True)
    model = Spell2Pronunciation(config).to(config.device)

    # class_weight = torch.tensor([1e-6, 0.01, 0.01, 0.01])
    # preserve = torch.ones(trg_vocab_size - class_weight.size()[0])
    # class_weight = torch.cat((class_weight, preserve), dim=0).to(config.device)
    # criterion = nn.CrossEntropyLoss(weight=class_weight)
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
    #                                                  mode='min',
    #                                                  patience=2)
    #
    # train_continue = True
    # if train_continue:
    #     weights = glob.glob('./model_weight/transformer_normal_*')
    #     last_epoch = int(weights[-1].split('_')[-1])
    #     weight_path = weights[-1].replace('\\', '/')
    #     print('weight info of last epoch', weight_path)
    #     model.load_state_dict(torch.load(weight_path))
    #     plus_epoch = 20
    #     total_epoch = last_epoch + plus_epoch
    # else:
    #     last_epoch = 0
    #     plus_epoch = 20
    #     total_epoch = plus_epoch
    #
    # dataset = CustomDataset(config, tokenizer)
    # data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    #
    # model.train()
    # for epoch in range(plus_epoch):
    #     total_loss = 0
    #     for iteration, datas in enumerate(data_loader):
    #         encoder_inputs, decoder_inputs, targets = datas
    #         optimizer.zero_grad()
    #         logits = model(encoder_inputs, decoder_inputs)
    #         logits = logits.contiguous().view(-1, trg_vocab_size)
    #         targets = targets.contiguous().view(-1)
    #         # indices = targets.nonzero().squeeze(1)
    #         # logits = logits.index_select(0, indices)
    #         # targets = targets.index_select(0, indices)
    #         loss = criterion(logits, targets)
    #         loss.backward()
    #         nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    #         optimizer.step()
    #         total_loss += loss
    #     scheduler.step(total_loss)
    #         # if (iteration + 1) % 50 == 0:
    #         #     print('Iteration: %3d \t' % (iteration + 1), 'Cost: {:.5f}'.format(loss))
    #     # break
    #     # if (epoch + 1) % 5 == 0:
    #     print('Epoch: %3d\t' % (last_epoch + epoch + 1), 'Cost: {:.5f}'.format(total_loss/(iteration + 1)))
    #     # if (epoch + 1) % 100 == 0:
    # model_path = './model_weight/transformer_normal_%d' % total_epoch
    # torch.save(model.state_dict(), model_path)

    model.load_state_dict(torch.load('./model_weight/transformer_normal_20'))
    model.eval()
    # sample_encoder_input = ['나는 안녕하세요 1+1 이벤트 진행 중이다, 가격 1300원이야.',
    #                         '가랑비에 옷 젖는 줄 모른다.',
    #                         '고객님, 현재 짜파게티는 1+1 상품으로 이벤트가 진행중이니 살펴보고 가세요.',
    #                         '확인해 드릴게요, 세금을 포함해서 102만 원이라고 나오네요.']
    sample_encoder_input = ['확인해 드릴게요, 세금을 포함해서 102만 원이라고 나오네요.']
    sample_decoder_input = [''] * len(sample_encoder_input)
    sample_encoder_input, sample_decoder_input, _ = make_feature(sample_encoder_input, sample_decoder_input,
                                                                 tokenizer, config)

    sample_output = torch.zeros(1, config.decoder_max_seq_length, dtype=torch.long, device=config.device)

    encoder_output = model.encoders(sample_encoder_input).last_hidden_state
    next_symbol = sample_decoder_input[0][0]
    print(sample_encoder_input.size(), sample_decoder_input.size())
    print('sample output:', sample_output.size())
    print('next symbol:', next_symbol)
    for i in range(config.decoder_max_seq_length):
        sample_output[0][i] = next_symbol
        decoder_attn_mask = get_pad_mask(sample_output, sample_output, config.padding_idx)
        look_ahead_attn_mask = get_look_ahead_mask(sample_output)
        look_ahead_attn_mask = torch.gt((decoder_attn_mask + look_ahead_attn_mask), 0)
        decoder_attn_mask = get_pad_mask(sample_output, sample_encoder_input, config.padding_idx)
        sample_embedding = model.embedding_projection(model.embedding(sample_output))
        prob, _, _ = model.decoders(encoder_output, sample_embedding, look_ahead_attn_mask, decoder_attn_mask)
        prob = model.dense(prob)
        prob = prob.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
        # print('prob:', prob.size())
        # print('next_word:', next_word.size())
        # print('next_symbol:', next_symbol)

    eos_idx = int(torch.where(sample_output[0] == 34001)[0][0])
    sample_output = sample_output[0][1:eos_idx]
    print(sample_output)

    result = sample_output.to('cpu').numpy()
    result = list(map(int, result))
    result = tokenizer.convert_ids_to_tokens(result)
    print('예측 토큰 결과:', result)
    result2 = tokenizer.convert_tokens_to_string(result)
    print('최종 문장:', result2)

    # predicts = model(sample_encoder_input, sample_decoder_input)
    # print(predicts.size())
    # predicts = torch.max(predicts, dim=-1)[-1].long().to('cpu')

    # for predict in predicts:
    #     predict = predict.numpy()
    #     print('predict size:', predict.shape)
    #     predict = list(map(int, predict))
    #     predict = tokenizer.convert_ids_to_tokens(predict)
    #     print('예측 결과:', tokenizer.convert_tokens_to_string(predict))
