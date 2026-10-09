"""Explicit Torch serving weight preparation without changing FP32 masters.

PyTorch disables autocast weight caching in inference_mode. Prepare BF16 linear
copies for the LLT backend model while keeping FP32 embedding and master weights.
The serving comparison accounts for these copies in its memory measurements.
"""
from types import MethodType
import torch
from torch.nn import functional as F
from model.tensor_backend import BackendTransformer


def prepare(model):
    if not isinstance(model, BackendTransformer):
        raise TypeError('prepare expects the LLT backend model; historical nanoGPT adapters are archived')
    casts = {}

    def cast(parameter):
        if not isinstance(parameter, torch.nn.Parameter):
            return parameter  # Folded tensors are already produced in BF16.
        key = id(parameter)
        version = (parameter._version, parameter.data_ptr())
        if key not in casts or casts[key][0] != version:
            casts[key] = (version, parameter.to(torch.bfloat16))
        return casts[key][1]

    def linear(self, x, weight):
        if not torch.is_inference_mode_enabled():
            raise ValueError('prepared model requires inference_mode')
        return F.linear(x, cast(weight))

    model.linear = MethodType(linear, model)
    model.prepared_weight_casts = casts
    return model
