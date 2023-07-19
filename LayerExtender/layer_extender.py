from torch import nn, Tensor
from typing import Optional, Tuple, Union
import traceback
import loralib as lora
import fnmatch


class BaseModuleExtender(nn.Module):
    """
    Base code to wrap an arbitary nn.Module into a new class.

    The original module is saved as self.wrapped_module and called automatically via the foerward pass.

    You can modify the values passed into wrapped layer via the arg and kwarg params
    """
    def __init__(self, config, wrapped_module: nn.Module, **kwargs):
        super().__init__()
        
        self.wrapped_module = wrapped_module

    def forward(self, input: Tensor, * args, **kwargs):
        wrapped_outputs = self.wrapped_module(input, * args, **kwargs)
        return wrapped_outputs   
    
    @staticmethod
    def is_match(name_list: str = "", type_list: str = ""):
        return False
    

class BaseLayerExtender(BaseModuleExtender):
    """
    Case code for wrapping LLM Layers.

    Layers are the fundamental unit for pipe parallel.

    This class is designed to interface our custom layers with the huggingface Trainer class and DeepSpeed.
    """
    def __init__(self, config, wrapped_layer: nn.Module, **kwargs):
        super().__init__(config, wrapped_layer)

    def forward(self, input: Tensor, * args, **kwargs):        
        wrapped_outputs = self.wrapped_module(input, * args, **kwargs)
        return wrapped_outputs
    
class LoraExtender(BaseModuleExtender):
    """
    Wrapper for Lora adapaters
    """
    def __init__(self, config, wrapped_module: nn.Module, **kwargs):
        weight = wrapped_module.weight
        bias = wrapped_module.bias
        wrapped_module = lora.MergedLinear(wrapped_module.in_features, wrapped_module.out_features, r = 4, enable_lora=[True])
        wrapped_module.weight = weight
        wrapped_module.bias = bias

        if "test" in kwargs:
            print(kwargs["test"])

        super().__init__(config, wrapped_module)

    def forward(self, input: Tensor, * args, **kwargs):        
        wrapped_outputs = self.wrapped_module(input, * args, **kwargs)
        return wrapped_outputs
    
    @staticmethod
    def is_match(name_list: str = "", type_list: str = ""):
        model_name = type_list
        first_token = model_name.find('.')
        if first_token >= 0:
            model_name = model_name[0:first_token]

        name_match = False
        type_match = False
        if model_name == "GPTNeoXModel":
            name_match = fnmatch.fnmatchcase(name_list, "*query_key_value")
            type_match = fnmatch.fnmatchcase(type_list, "*Linear")

        return name_match and type_match
