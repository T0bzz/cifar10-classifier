import torch
from torch import nn

# Flatten a tensor except for the first dimension (batch size)
def flatten(x):
    return x.reshape(x.size(0), -1) # reshape handles non-contiguous tensors


print(flatten(torch.randn(4, 64, 8, 8)).shape)  # Example usage