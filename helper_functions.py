import torch
from torch import nn

# Flatten a tensor except for the first dimension (batch size)
def flatten(x):
    """
    Flatten a tensor except for the first dimension (batch size)
    Args:
        x (torch.Tensor): Input tensor of shape (Batch_size, C, H, W)

    Returns:
        flattened_size (int): Size of the flattened tensor excluding batch size.
        flattened_tensor (torch.Tensor): Flattened tensor of shape (Batch_size, flattened_size)
    """

    flattened_size = x.numel() // x.size(0)
    flattened_tensor = x.reshape(x.size(0), -1)
    return flattened_size, flattened_tensor  # reshape handles non-contiguous tensors