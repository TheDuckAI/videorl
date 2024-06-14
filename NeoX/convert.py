from torch import nn
import loralib as lora

def convert_model_lora(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.Linear) and child_name == "query_key_value":
            weight = child.weight
            bias = child.bias
            new = lora.MergedLinear(child.in_features, child.out_features, r = 4)
            new.weight = weight
            new.bias = bias
            setattr(model, child_name, new)
        # elif isinstance(child, nn.Conv2d):
        #     weight = child.weight
        #     bias = child.bias
        #     new = lora.Conv2d(child.in_channels, child.out_channels, child.kernel_size[0], r = 4)#kernel size would 
        #     new.weight = weight
        #     new.bias = bias
        #     setattr(model, child_name, new)
        # elif isinstance(child, nn.Embedding):
        #     weight = child.weight
        #     new = lora.Embedding(child.num_embeddings, child.embedding_dim, r = 4)
        #     new.weight = weight
        #     setattr(model, child_name, new)
        else:
            convert_model_lora(child)
    return model