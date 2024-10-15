import os
import pandas as pd
from datetime import datetime
from datasets import Dataset
import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import setup_chat_format, SFTTrainer
from bert_score import BERTScorer

def compute_bertscore(pred, gt, lang='vi'):
    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    scorer = BERTScorer(lang=lang, model_type=model_name)
    P, R, F1 = scorer.score([pred], [gt])
    return  F1.mean().item()

class TrainerClass:
    def __init__(self, base_model_id, output_dir, train_csv, test_csv):
        # Initialize paths and directories
        self.base_model_id = base_model_id
        self.output_dir = output_dir
        self.train_csv = train_csv
        self.test_csv = test_csv

        # Initialize tokenizer and model (will be done in respective methods)
        self.tokenizer = None
        self.model = None

        # Define system message template
        self.system_message = """Là một chuyên gia đọc hiểu, hãy trả lời question dưới đây dựa vào context mà tôi cung cấp.
        Câu trả lời ngắn gọn, chính xác. Nếu câu nào không có câu trả lời thì hãy trả về không tìm thấy thông tin.
        Dưới đây là thông tin của context: {context}
        """

    def create_conversation(self, row):
        return {
            "messages": [
                {"role": "system", "content": self.system_message.format(context=row["context"])},
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]}
            ]
        }

    def load_datasets(self):
        # Load training and testing datasets from CSV
        train_df = pd.read_csv(self.train_csv)
        test_df = pd.read_csv(self.test_csv)

        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

        # Apply conversation template to datasets
        train_dataset = train_dataset.map(self.create_conversation)
        test_dataset = test_dataset.map(self.create_conversation)

        return train_dataset, test_dataset

    def initialize_tokenizer(self):
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        self.tokenizer.padding_side = 'left'

    def load_model(self):
        # Load quantized model with 4-bit precision
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id, 
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )

        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

    def setup_peft(self):
        # Setup LoRA (Low-Rank Adaptation) configuration for parameter-efficient fine-tuning
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", 
                "up_proj", "down_proj", "lm_head"
            ],
            bias="none",
        )
        return peft_config

    def setup_training_args(self):
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,                 # directory to save model and logs
            num_train_epochs=10, 
            do_train = True,
            do_eval= True,
            per_device_train_batch_size=2,              # batch size per device
            gradient_accumulation_steps=2,              # accumulate gradients to simulate larger batch size
            optim="adamw_torch_fused",                  # optimizer
            logging_steps=20,                           # log every 20 steps
            save_strategy="epoch",                      # save checkpoint every epoch
            learning_rate=2e-5,                         # learning rate
            bf16=True,                                  # use bfloat16 precision
            tf32=True,                                  # use tf32 precision
            max_grad_norm=0.3,                          # max gradient norm
            warmup_ratio=0.03,                          # warmup ratio
            lr_scheduler_type="constant",               # use constant learning rate scheduler
            push_to_hub=False,                          # whether to push the model to hub
            report_to="tensorboard",                    # report metrics to tensorboard
        )
        return training_args

    def train_model(self):
        # Load datasets
        train_dataset, test_dataset = self.load_datasets()

        # Initialize tokenizer
        self.initialize_tokenizer()

        # Load model
        self.load_model()

        # Set up chat format
        self.model, self.tokenizer = setup_chat_format(self.model, self.tokenizer)

        # Set up PEFT configuration
        peft_config = self.setup_peft()

        # Set up training arguments
        training_args = self.setup_training_args()
        
        # define
        def compute_metrics(pred):
            predictions, labels = pred
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            bert_scores = [compute_bertscore(pred,label) for pred, label in zip(decoded_preds, decoded_labels)]
            return {
                "bert_score": sum(bert_scores) / len(bert_scores)
            }
            
        # Initialize trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset, 
            eval_dataset=test_dataset,
            peft_config=peft_config,
            max_seq_length=3072,
            tokenizer=self.tokenizer,
            compute_metrics = compute_metrics,
            packing=True,
            dataset_kwargs={
                "add_special_tokens": False,  # We template with special tokens
                "append_concat_token": False, # No need to add additional separator token
            }
        )

        # Start training
        trainer.train()

# Example usage
if __name__ == "__main__":
    base_model_id = "/home/namnt/md1/NAMNT_DA2/base_llm/T-VisStar-7B-v0.1"
    output_dir = "/mnt/md1/mlflow/DATN/09_10_2024"
    train_csv = '/point/namnt/DATN/genneration/data/qa_output.csv'
    test_csv = '/point/namnt/DATN/genneration/data/data_test.csv'

    trainer_class = TrainerClass(base_model_id, output_dir, train_csv, test_csv)
    trainer_class.train_model()
