
#dummy config
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from convert import convert_model_lora
import torch
initial = True
from safetensors.torch import save_file, load_file

if initial:#Load up a GPT Neo-x model specified by the config, convert to the lora model desired.
    
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    model = convert_model_lora(model)
    
    # torch.save(model.state_dict(), "./model.pt")
    model.save_pretrained("./", safe_serialization = "True")

else:
    #We want to load a model
    
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped")#Is it possible to just load from config without this issue...
    model = convert_model_lora(model)
    #We could skip the above step if we coded something that has the new architecture - this seems bad though because we'd need to do per adapter method
    
    loaded = load_file("./model.safetensors")
    model.load_state_dict(loaded)
    