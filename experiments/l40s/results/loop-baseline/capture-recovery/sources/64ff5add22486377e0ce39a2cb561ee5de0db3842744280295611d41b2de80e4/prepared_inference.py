"""Explicit Torch serving weight preparation without changing FP32 masters.

PyTorch disables autocast weight caching in inference_mode. Prepare BF16 linear
copies explicitly, including a separate projection copy of nanoGPT's tied
embedding parameter. Embedding lookup keeps the original FP32 master. Numerical
operations remain Torch; this adapter is only used in the serving comparison.
"""
from types import MethodType
import torch
from torch.nn import functional as F
from tensor_model import BackendTransformer
from nanogpt_adapter import graph_compatible


def prepare(model):
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

    def module_linear(self, x):
        if not torch.is_inference_mode_enabled():
            raise ValueError('prepared model requires inference_mode')
        return F.linear(x, cast(self.weight), cast(self.bias) if self.bias is not None else None)

    if isinstance(model, BackendTransformer):
        model.linear = MethodType(linear, model)
    else:
        graph_compatible(model)
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.forward = MethodType(module_linear, module)
    model.prepared_weight_casts = casts
    return model
