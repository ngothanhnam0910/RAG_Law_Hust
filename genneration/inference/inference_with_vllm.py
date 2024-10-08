# !pip install vllm==0.3.3
from vllm import LLM, SamplingParams
from datasets import load_dataset
import time


llm = LLM(model="/home/namnt/md1/mlflow/DATN/07_10_2024/merge_model",
           max_model_len=2048,
           gpu_memory_utilization=0.8)
tokenizer = llm.get_tokenizer()

# Define message
eval_dataset = load_dataset("json", data_files="test_dataset.json",split="train")
messages = eval_dataset[7]["messages"][:2]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=False)


sampling = SamplingParams(max_tokens=512, seed=42, temperature=0)

t1 = time.time()
outputs = llm.generate(prompt, sampling)
t2 = time.time()
print(f"Total time : {t2 - t1}")
results = [output.outputs[0].text for output in outputs]

print("-----results: {}".format(results[0].replace("assistant","").split("system")[0].strip()))
