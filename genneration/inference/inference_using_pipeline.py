import os
import pandas as pd
from datetime import datetime
from datasets import Dataset
import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from trl import setup_chat_format
import time
# Load the evaluation dataset
eval_dataset = load_dataset("json", data_files="test_dataset.json", split="train")

# Path to the merged model (the model with LoRA layers merged into the base model)
merged_model_path = "/home/namnt/md1/mlflow/DATN/09_10_2024/merge_model"

# Load the merged model using AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    merged_model_path,
    torch_dtype=torch.float16,  # Use float16 for inference on GPU
    device_map='cuda'  # Ensure the model is loaded on GPU
)

# Load the tokenizer associated with the merged model
tokenizer = AutoTokenizer.from_pretrained(merged_model_path)


# Load into the pipeline for text generation
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

eval_dataset = load_dataset("json", data_files="test_dataset.json",split="train")

list_data_test = []
for i in range(50):
    messages = eval_dataset[i]["messages"][:2]
    list_data_test.append(messages)

# Test on a sample
prompt = pipe.tokenizer.apply_chat_template(list_data_test, tokenize=False, add_generation_prompt=True)

# Generate text based on the sample
t1 = time.time()
outputs = pipe(prompt,
               max_new_tokens=1024,
               do_sample=False,
               temperature=0,
               top_k=50,
               top_p=0.1,
               eos_token_id=pipe.tokenizer.eos_token_id,
               pad_token_id=pipe.tokenizer.pad_token_id)
t2 = time.time()
print(f"Total time: {t2 - t1}")
# Print original and generated answers
# print(f"Original Answer:\n{eval_dataset[4]['messages'][2]['content']}")
# output = outputs[0]['generated_text'][len(prompt):].strip()

# print(f"output: {output}")

# final_output = output.split("system")[0].strip()
# print(f"-------------Generated Answer:\n{final_output}")
# print(f"----------Generated answer: {outputs}")
