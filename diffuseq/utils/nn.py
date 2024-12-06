"""
Various utilities for neural networks.
"""

import math

import jittor as jt

def mean_flat(tensor:jt.Var):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = jt.exp(
        -math.log(max_period) * jt.arange(start=0, end=half, dtype=jt.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = jt.cat([jt.cos(args), jt.sin(args)], dim=-1)
    if dim % 2:
        embedding = jt.cat([embedding, jt.zeros_like(embedding[:, :1])], dim=-1)
    return embedding