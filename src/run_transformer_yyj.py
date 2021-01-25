import torch
from torch import nn
from transformer_yyj import Transformer, TransformerConfig
from torch import optim

if __name__ == '__main__':
    config = TransformerConfig(src_vocab_size=5000,
                               trg_vocab_size=5000,
                               device='cuda')

    inputs = torch.randint(5000, (4, 12), dtype=torch.float, device=config.device)
    labels = torch.randint(5000, (4, 12), dtype=torch.float, device=config.device)

    model = Transformer(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    total_epoch = 10

    model.train()
    for epoch in range(total_epoch):
        optimizer.zero_grad()
        outputs, _ = model(inputs, labels)
        outputs = outputs.contiguous().view(-1, 5000)
        target = labels.contiguous().view(-1).long()
        # outputs = torch.max(outputs, dim=-1)[-1].float()
        loss = criterion(outputs, target)
        print('Epoch: %3d ' % (epoch + 1), '\tCost: {:.5f}'.format(loss))
        loss.backward()
        optimizer.step()

    model.eval()
    predict, _ = model(inputs, labels)
    predict = predict.data.max(1, keepdim=True)[1]
    print(predict.size())
    print(labels.size())
