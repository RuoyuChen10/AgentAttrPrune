"""Original AgentPrune GSM8K raw-logit regularizer."""
import torch

def graph_regularization(graph, fixed_masks, delta):
    """Author fix d95b52b: nuclear norm + Frobenius hinge on RAW logits.

    Source: https://github.com/yanweiyue/AgentPrune/issues/4#issuecomment-2763167690
    Use the author's actual parameterization, rather than silently replacing
    logits by probabilities. Each term has coefficient one in that implementation.
    """
    terms = []
    for edge_type in ('spatial', 'temporal'):
        if not getattr(graph, f'optimized_{edge_type}'):
            continue
        logits = getattr(graph, f'{edge_type}_logits')
        matrix = logits.reshape_as(fixed_masks[edge_type])
        distance = torch.linalg.matrix_norm(fixed_masks[edge_type] - matrix, ord='fro')
        terms.append(torch.linalg.matrix_norm(matrix, ord='nuc') + torch.relu(distance - delta))
    return torch.stack(terms).sum()
