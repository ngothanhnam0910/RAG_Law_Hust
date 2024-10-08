from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from trl import setup_chat_format

# Đường dẫn đến mô hình base và mô hình LoRA đã fine-tune
base_model_path = "/home/namnt/md1/NAMNT_DA2/base_llm/T-VisStar-7B-v0.1"  # Base model (GPT-2 hoặc model khác bạn đã sử dụng)
peft_model_path = "/home/namnt/md1/mlflow/DATN/07_10_2024/checkpoint-210"  # Đường dẫn đến LoRA fine-tuned model

# Tải tokenizer từ base model hoặc từ LoRA model nếu có thêm các token đặc biệt
tokenizer = AutoTokenizer.from_pretrained(peft_model_path)

# In kích thước từ vựng trước khi setup_chat_format
print("----------------Vocab size before setup_chat_format:", len(tokenizer))

# Tải base model trước 
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, 
                                                  torch_dtype=torch.bfloat16,
                                                  low_cpu_mem_usage=True)

# Setup chat format (nếu cần)
base_model, tokenizer = setup_chat_format(base_model, tokenizer)

# In kích thước từ vựng sau khi setup_chat_format
print("-----------------Vocab size after setup_chat_format:", len(tokenizer))

# Áp dụng LoRA adapter từ mô hình fine-tuned vào base model
peft_model = PeftModel.from_pretrained(base_model, peft_model_path)

# Merge LoRA layers vào base model
peft_model = peft_model.merge_and_unload()

# Lưu lại mô hình đã merge để sử dụng cho suy luận sau này
output_path = "/home/namnt/md1/mlflow/DATN/07_10_2024/merge_model"
peft_model.save_pretrained(output_path, max_shard_size="2GB")
tokenizer.save_pretrained(output_path)
