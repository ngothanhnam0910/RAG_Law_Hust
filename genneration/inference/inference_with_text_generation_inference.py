import os
import pandas as pd
from datetime import datetime
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from trl import setup_chat_format
import requests as r

# Load the evaluation dataset
eval_dataset = load_dataset("json", data_files="test_dataset.json", split="train")

# Path to the merged model (the model with LoRA layers merged into the base model)
merged_model_path = "/home/namnt/md1/mlflow/DATN/07_10_2024/merge_model"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

# Prompt
prompt = tokenizer.apply_chat_template(eval_dataset[7]["messages"][:2], tokenize=False, add_generation_prompt=True)

print(f"prompt: {prompt}")
exit()

request= {"inputs":prompt,"parameters":{"temperature":0, "top_p": 0.1, "max_new_tokens": 1024}}

# send request to inference server
resp = r.post("http://127.0.0.1:8080/generate", json=request)

print(f"---------------------------")
print(resp)

# output = resp.json()["generated_text"].strip()
# time_per_token = resp.headers.get("x-time-per-token")
# time_prompt_tokens = resp.headers.get("x-prompt-tokens")
 
# Print results
# print(f"Query:\n{eval_dataset[7]['messages'][1]['content']}")
# print(f"Original Answer:\n{eval_dataset[7]['messages'][2]['content']}")
# print(f"Generated Answer:\n{output}")
# print(f"Latency per token: {time_per_token}ms")
# print(f"Latency prompt encoding: {time_prompt_tokens}ms")