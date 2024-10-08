import os
import pandas as pd
from datetime import datetime
from datasets import Dataset
import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import AutoPeftModelForCausalLM, PeftConfig
from datasets import load_dataset



eval_dataset = load_dataset("json", data_files="test_dataset.json",split="train")

peft_model_path = "/home/namnt/md1/mlflow/DATN/07_10_2024/checkpoint-210"
    
bnb_config2 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoPeftModelForCausalLM.from_pretrained(
                                        peft_model_path,
                                        is_trainable=False,
                                        quantization_config=bnb_config2,
                                        device_map = 'cuda')

tokenizer = AutoTokenizer.from_pretrained(peft_model_path)

# load into pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Test on sample
prompt = pipe.tokenizer.apply_chat_template(eval_dataset[6]["messages"][:2], tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, 
               max_new_tokens=1024, 
               do_sample=False,
               temperature=0.1, 
               top_k=50,
               top_p=0.1, 
               eos_token_id=pipe.tokenizer.eos_token_id, 
               pad_token_id=pipe.tokenizer.pad_token_id)

print(f"Original Answer:\n{eval_dataset[6]['messages'][2]['content']}")
print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")