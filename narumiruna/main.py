import torch
from torch import nn, optim


def main():
    num_iterations = 1989 * 6 * 4

    x = torch.tensor([6, 4], dtype=torch.float).view(1, -1)
    y = torch.tensor([84, 105, 97, 110, 97, 110, 109, 101, 110, 32, 83, 113, 117, 97, 114, 101, 32, 67, 114, 97, 99, 107, 100, 111, 119, 110, 32, 51, 48, 116, 104, 32, 65, 110, 110, 105, 118, 101, 114, 115, 97, 114, 121], dtype=torch.float)
    
    model = nn.Linear(x.size(1), y.size(0))
    optimizer = optim.Adam(model.parameters(), lr=0.00064)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iterations)

    for _ in range(num_iterations):
        scheduler.step()

        out = model(x).view(-1)
        loss = (y - out).norm()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        try:
            print(''.join(map(lambda s: chr(round(s)), out.tolist())), end='\r')
        except ValueError:
            pass


if __name__ == '__main__':
    main()
