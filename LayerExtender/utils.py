from torch import nn
from layer_extender import BaseLayerExtender, LoraExtender
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
import loralib as lora
from dataclasses import dataclass, field
import importlib
import fnmatch

@dataclass
class ExtensionDefinition:
    new_layer: str
    match_name: str = "*"
    match_type: str = "*"
    params: dict = field(default_factory=dict)
    use_default_match: bool = False
    

def convert_model(model, config):
    extension_defs = config["extend_layers"]

    extensions = [ExtensionDefinition(**extension) for extension in extension_defs]


    return convert_model_internal(model, config, extensions)

def convert_model_internal(model, config, extensions, name_list: str = "", type_list: str = ""):
    for child_name, child in model.named_children():
        new_layer = None

        name_list += f'{child_name}' if name_list == "" else f'.{child_name}'
        type_list += f'{type(child).__name__}' if type_list == "" else f'.{type(child).__name__}'

        for extension in extensions:
            class_type = getattr(importlib.import_module("layer_extender"), extension.new_layer)
            if extension.use_default_match:
                if class_type.is_match(name_list, type_list):
                    new_layer = class_type(config, child, **extension.params)
            else:
                name_match = True
                if not extension.match_name == "*":
                    name_match = fnmatch.fnmatchcase(name_list, extension.match_name)

                type_match = True
                if not extension.match_type == "*":
                    type_match = fnmatch.fnmatchcase(type_list, extension.match_type)

                if type_match and name_match:
                    new_layer = class_type(config, child, **extension.params)
        
        if not new_layer is None:
            setattr(model, child_name, new_layer)

        convert_model_internal(child, config, extensions, name_list, type_list)
    return model


def print_model(model, indent = ""):
    for child_name, child in model.named_children():
        print(f'{indent}{child_name} ({type(child).__name__}):')
        print_model(child, f'{indent}  ')
