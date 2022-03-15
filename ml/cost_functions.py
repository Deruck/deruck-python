import torch

def mse(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return ((pred - label)**2).sum() / pred.shape[0]

def squared_loss(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return (pred - label).norm()