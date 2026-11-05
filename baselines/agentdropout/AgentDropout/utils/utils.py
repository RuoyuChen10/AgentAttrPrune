"""Original graph regularizers, extracted without changing their formulas."""
import torch

def nuclear_norm(matrix):
    _, S, _ = torch.svd(matrix)
    return torch.sum(S)


def frobenius_norm(A, S):
    return torch.norm(A - S, p='fro')
