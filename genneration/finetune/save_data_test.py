import os
import pandas as pd
from datetime import datetime
from datasets import Dataset
import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import setup_chat_format,SFTTrainer

# define function for convert dataset to type of conversation
system_message = """Là một chuyên gia đọc hiểu, hãy trả lời question dưới đây dựa vào context mà tôi cung cấp.
Câu trả lời ngắn gọn, chính xác. Nếu câu nào không có câu trả lời thì hãy trả về không tìm thấy thông tin.
Dưới đây là thông tin của context: {context}
"""
 
def create_conversation(row):
  return {
    "messages": [
      {"role": "system", "content": system_message.format(context=row["context"])},
      {"role": "user", "content": row["question"]},
      {"role": "assistant", "content": row["answer"]}
    ]
  }
  
  

test_csv = '/point/namnt/DATN/genneration/data/data_test.csv'
test_df = pd.read_csv(test_csv)
test_dataset = Dataset.from_pandas(test_df)
test_dataset = test_dataset.map(create_conversation)

test_dataset.to_json("test_dataset.json", orient="records", force_ascii=False)

