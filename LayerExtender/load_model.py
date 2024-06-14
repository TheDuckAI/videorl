
#dummy config
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from utils import print_model, convert_model
import torch
from safetensors.torch import save_file, load_file
import json

with open('videorl/LayerExtender/config.json', 'r') as file:
    config = json.load(file)

initial = True

if initial:#Load up a GPT Neo-x model specified by the config, convert to the lora model desired.
    
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    model = convert_model(model, config)

    print(config)

    print_model(model)

    model.save_pretrained("./", safe_serialization = "True")

    prompt = "Test"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.000001,
        max_length=100,
    )
    
    print(tokenizer.batch_decode(gen_tokens)[0])
    

else:
    #We want to load a model
    
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped")#Is it possible to just load from config without this issue...
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    model = convert_model(model, {})
    #We could skip the above step if we coded something that has the new architecture - this seems bad though because we'd need to do per adapter method
    
    loaded = load_file("./model.safetensors")
    model.load_state_dict(loaded)

    prompt = "Test"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.000001,
        max_length=100,
    )
    
    print(tokenizer.batch_decode(gen_tokens)[0])
    