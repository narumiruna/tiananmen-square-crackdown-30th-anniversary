import argparse

import torch
from torch import nn, optim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--string', type=str,
                        default='Tiananmen Square Crackdown 30th Anniversary')
    parser.add_argument('-bs', '--batch-size', type=int, default=9)
    parser.add_argument('-d', '--dim', type=int, default=3)
    parser.add_argument('--num-iterations', type=int, default=15000)
    return parser.parse_args()

def main():
    args = parse_args()

    x = torch.Tensor([args.batch_size, args.dim])
    y = torch.Tensor([ord(s) for s in args.string])
    net = nn.Linear(x.size(0), y.size(0))
    optimizer = optim.Adam(net.parameters())
    for _ in range(args.num_iterations):
        out = net(x)
        loss = (y - out).norm()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        try:
            print(''.join(map(lambda s: chr(round(s)), out.tolist())))
        except ValueError:
            pass


if __name__ == '__main__':
    main()
