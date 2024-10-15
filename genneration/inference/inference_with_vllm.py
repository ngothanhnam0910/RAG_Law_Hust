# !pip install vllm==0.3.3
from vllm import LLM, SamplingParams
from datasets import load_dataset
import time


llm = LLM(model="/home/namnt/md1/mlflow/DATN/09_10_2024/merge_model",
           max_model_len=2048,
           gpu_memory_utilization=0.8)
tokenizer = llm.get_tokenizer()

# Define message
eval_dataset = load_dataset("json", data_files="test_dataset.json",split="train")

list_data_test = []
for i in range(50):
    messages = eval_dataset[i]["messages"][:2]
    list_data_test.append(messages)
    

prompt = tokenizer.apply_chat_template(list_data_test, tokenize=False, add_special_tokens=False)


sampling = SamplingParams(max_tokens=512, seed=42, temperature=0)

t1 = time.time()
outputs = llm.generate(prompt, sampling)
t2 = time.time()
print(f"Total time : {t2 - t1}")
results = [output.outputs[0].text for output in outputs]

print(f"Results : {results}")

# print("-----results: {}".format(results[0].replace("assistant","").strip()))
