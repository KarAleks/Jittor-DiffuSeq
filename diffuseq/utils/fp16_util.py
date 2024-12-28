"""
Helpers to train with 16-bit precision.
"""

# import torch.nn as nn
# from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import jittor as jt
from jittor import nn

def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        l.bias.data = l.bias.data.float()


def make_master_params(model_params):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """
    master_params = [param.detach().float32() for param in model_params]
    for param in master_params:
        # print("param type: ",param.dtype)
        param.requires_grad = True
    return master_params


def model_grads_to_master_grads(model_params, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    for model_param, master_param in zip(model_params, master_params):
        if model_param.grad is not None:
            master_param.grad = model_param.grad.float32()


def master_params_to_model_params(model_params, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    model_params = list(model_params)

    for model_param, master_param in zip(model_params, master_params):
        model_param.update(master_param.detach())


def unflatten_master_params(model_params, master_params):
    """
    Unflatten the master parameters to look like model_params.
    """
    # return _unflatten_dense_tensors(master_params[0].detach(), model_params)
    offsets = [0]
    for param in model_params:
        offsets.append(offsets[-1] + param.numel())

    master_params_flat = master_params[0].detach()
    return [
        master_params_flat[offsets[i] : offsets[i + 1]].reshape(param.shape)
        for i, param in enumerate(model_params)
    ]

def zero_grad(model_params,optim):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        try:
            grad = param.opt_grad(optim)
            if grad is not None:
                #param.grad.detach_()
                grad.zero_()
        except:
            pass
        # ecx