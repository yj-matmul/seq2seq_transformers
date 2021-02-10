import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from transformer_yyj import Transformer, TransformerConfig, Decoders
from utils import text2ids
from transformers import ElectraTokenizer


class CustomDataset(Dataset):
    def __init__(self, tokenizer, config):
        src_file_path = 'sample_src.txt'
        trg_file_path = 'sample_trg.txt'
        with open(src_file_path, 'r', encoding='utf8') as f:
            src_lines = list(map(lambda x: x.strip('\n'), f.readlines()))
        with open(trg_file_path, 'r', encoding='utf8') as f:
            trg_lines = list(map(lambda x: x.strip('\n'), f.readlines()))
        self.enc_input = text2ids(src_lines, tokenizer, config, 'encoder')
        self.dec_input = text2ids(trg_lines, tokenizer, config, 'decoder')
        self.target = text2ids(trg_lines, tokenizer, config, 'target')

    def __len__(self):
        return len(self.enc_input)

    def __getitem__(self, idx):
        x = self.enc_input[idx]
        y = self.dec_input[idx]
        z = self.target[idx]
        return x, y, z


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

    dataset = CustomDataset(tokenizer, config)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=2)

    total_epoch = 200
    for epoch in range(total_epoch):
        epoch_loss = 0
        for iteration, data in enumerate(data_loader):
            encoder_inputs, decoder_inputs, targets = data
            optimizer.zero_grad()
            logits, _ = model(encoder_inputs, decoder_inputs)
            logits = logits.contiguous().view(-1, trg_vocab_size)
            targets = targets.contiguous().view(-1)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            epoch_loss += loss
        scheduler.step(epoch_loss)
        if (epoch + 1) % 10 == 0:
            print('Epoch: %3d\t' % (epoch + 1), 'Cost: {:.5f}'.format(epoch_loss / (iteration + 1)))
    model_path = './model_weight/transformer_%d' % total_epoch
    torch.save(model.state_dict(), model_path)
