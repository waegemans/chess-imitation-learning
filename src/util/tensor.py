import torch

def shift(x):
    return torch.cat((x[-1:],x[:-1]),dim=1)
