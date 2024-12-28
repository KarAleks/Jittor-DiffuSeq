"""
Various utilities for neural networks.
"""

import math

import jittor as jt

def mean_flat(tensor:jt.Var):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dims=list(range(1, len(tensor.shape))))

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
    device = "cuda" if jt.has_cuda else "cpu"
    freqs = jt.exp(
        -math.log(max_period) * jt.arange(start=0, end=half) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = jt.cat([jt.cos(args), jt.sin(args)], dim=-1)
    if dim % 2:
        embedding = jt.cat([embedding, jt.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    # import copy
    new_target = []
    for targ, src in zip(target_params, source_params):
        # tar1 = copy.deepcopy(targ)
        new_target.append(targ*rate+src*(1-rate))

    return new_target
        # targ.detach().update(targ.detach() * rate + src * (1 - rate))
        # import jittor as jt
        # print("Bozi txa bug:{}  {}".format(tar1.shape,jt.sum(tar1==targ)))