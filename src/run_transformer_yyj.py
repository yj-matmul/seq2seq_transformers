import torch
from torch import nn
from transformer_yyj import Transformer, TransformerConfig
from torch import optim

if __name__ == '__main__':
    vocab_size = 100
    config = TransformerConfig(src_vocab_size=vocab_size,
                               trg_vocab_size=vocab_size,
                               device='cuda')

    inputs = torch.randint(vocab_size, (100, 8), dtype=torch.float, device=config.device)
    labels = torch.randint(vocab_size, (100, 8), dtype=torch.float, device=config.device)

    model = Transformer(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    total_epoch = 50

    model.train()
    for epoch in range(total_epoch):
        optimizer.zero_grad()
        outputs, _ = model(inputs, labels)
        outputs = outputs.contiguous().view(-1, vocab_size)
        target = labels.contiguous().view(-1).long()
        loss = criterion(outputs, target)
        if epoch % 10 == 9:
            print('Epoch: %3d ' % (epoch + 1), '\tCost: {:.5f}'.format(loss))
        loss.backward()
        optimizer.step()

    model.eval()
    predict, _ = model(inputs, labels)
    predict = torch.max(predict, dim=-1)[-1]
    print(predict.size())
    print(labels.size())
    print('predict\n', predict[0])
    print('labes\n', labels[0])
