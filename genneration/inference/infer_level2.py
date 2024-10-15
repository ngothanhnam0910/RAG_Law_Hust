from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import torch
from datasets import load_dataset

def extract_assistant_content(output_text):
    # Tìm vị trí của "assistant"
    assistant_start = output_text.find("assistant")
    
    assistant_start += len("assistant")  # Chuyển vị trí sau "assistant"
    
    # Bỏ qua các ký tự không cần thiết (vd: khoảng trắng, dấu xuống dòng)
    while assistant_start < len(output_text) and (output_text[assistant_start] == ':' or output_text[assistant_start].isspace()):
        assistant_start += 1
    
    # Tìm kết thúc của assistant content (có thể là bắt đầu của phần "system" hoặc "user")
    assistant_end = output_text.find("system", assistant_start)
    if assistant_end == -1:
        assistant_end = len(output_text)  # Nếu không tìm thấy "system", lấy đến hết chuỗi
    
    # Trả về nội dung của assistant
    return output_text[assistant_start:assistant_end].strip()

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

messages = eval_dataset[9]["messages"][:2]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=False)
input_ids = tokenizer([prompt], return_tensors="pt").to("cuda")

t1 = time.time()
outputs = model.generate(**input_ids, max_length=1024, do_sample=True, top_k=50, top_p=0.4)
t2 = time.time()
print(f"Total")
decode_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print(f"decode_output: {decode_output}")

final_output = extract_assistant_content(decode_output)
print(f"final_output: {final_output}")
