import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, pipeline
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datetime import datetime
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import mlflow
from mlflow.models import infer_signature


class SFT:
    PROMPT_TEMPLATE = """Là một chuyên gia đọc hiểu, hãy trả lời question dưới đây dựa vào context mà tôi cung cấp. 
    Câu trả lời ngắn gọn, chính xác. Nếu câu nào không có câu trả lời thì hãy trả về không tìm thấy thông tin.

    ### Question:
    {question}

    ### Context:
    {context}

    ### Response:
    {output}"""

    def __init__(self, base_model_id, output_dir, max_length: int =4096):
        self.base_model_id = base_model_id
        self.max_length = max_length
        self.output_dir = output_dir
        self.train_dataset = None
        self.test_dataset = None
        self.tokenizer = None
        self.model = None

    def apply_prompt_template(self, row):
        """Format the dataset with the prompt template."""
        prompt = self.PROMPT_TEMPLATE.format(
            question=row["question"],
            context=row["context"],
            output=row["answer"],
        )
        return {"prompt": prompt}
    
    def load_datasets(self, train_csv, test_csv):
        """Load and split dataset."""
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        
        self.train_dataset = Dataset.from_pandas(train_df)
        self.test_dataset = Dataset.from_pandas(test_df)
        
        # Apply the prompt template to the train dataset
        self.train_dataset = self.train_dataset.map(self.apply_prompt_template)
        self.test_dataset = self.test_dataset.map(self.apply_prompt_template)


    def initialize_tokenizer(self):
        """Load the tokenizer and set pad_token."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            model_max_length=self.max_length,
            padding_side="left",
            add_eos_token=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_data(self):
        """Tokenize the dataset and pad to a fixed length."""
        def tokenize_and_pad_to_fixed_length(sample):
            result = self.tokenizer(
                sample["prompt"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result

        self.train_dataset = self.train_dataset.map(tokenize_and_pad_to_fixed_length)

    def initialize_model(self):
        """Load model with 4-bit quantization and set tokenizer."""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id, 
            quantization_config=quantization_config,
            device_map="auto"
        )

        # Enable gradient checkpointing for efficiency
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

    def initialize_peft(self):
        """Set up the PEFT LoRA configuration."""
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", 
                "up_proj", "down_proj", "lm_head"
            ],
            bias="none",
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def train_model(self):
        """Train the model."""
        training_args = TrainingArguments(
            report_to="mlflow",
            run_name=f"Vistral-QLoRA-{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}",
            output_dir=self.output_dir,
            do_train = True,
            do_eval= True,
            per_device_train_batch_size=1,
            per_gpu_eval_batch_size=1,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            bf16=True,
            learning_rate=1e-5,
            lr_scheduler_type="constant",
            eval_strategy = "steps",
            save_strategy = "steps",
            max_steps=5000,
            save_steps=250,
            logging_steps=250,
            warmup_steps=5,
            ddp_find_unused_parameters=False,
        )

        trainer = Trainer(
            model= self.model,
            train_dataset= self.train_dataset, 
            eval_dataset = self.test_dataset,
            data_collator= DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            args=training_args,
        )

        self.model.config.use_cache = False
        trainer.train()

    # def save_model(self):
    #     """Save the model to MLflow."""
    #     prompt_template = """You are a powerful text-to-SQL model. Given the SQL tables and natural language question, your job is to write SQL query that answers the question.

    #     {prompt}

    #     ### Response:
    #     """

    #     # Sample from the dataset for signature inference
    #     sample = self.train_dataset[1]
    #     signature = infer_signature(
    #         model_input=sample["prompt"],
    #         model_output=sample["answer"],
    #         params={"max_new_tokens": 256, "repetition_penalty": 1.15, "return_full_text": False},
    #     )

    #     # Get the ID of the last MLflow run
    #     last_run_id = mlflow.last_active_run().info.run_id
    #     tokenizer_no_pad = AutoTokenizer.from_pretrained(self.base_model_id, add_bos_token=True)

    #     with mlflow.start_run(run_id=last_run_id):
    #         mlflow.log_params({"max_steps": 20, "lr": 2e-5})  # Log parameters
    #         mlflow.transformers.log_model(
    #             transformers_model={"model": self.model, "tokenizer": tokenizer_no_pad},
    #             prompt_template=prompt_template,
    #             signature=signature,
    #             artifact_path="model"
    #         )


# Main function to run the process
if __name__ == "__main__":
    
    # Define path csv file
    train_csv = '/point/namnt/DATN/genneration/data/qa_output.csv'
    test_csv = '/point/namnt/DATN/genneration/data/data_test.csv'
    
    # Define model, output save checkpoint
    base_model_id = "/home/namnt/md1/NAMNT_DA2/base_llm/T-VisStar-7B-v0.1"
    output_dir = "/mnt/md1/mlflow/DATN/26_09_2024"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Init Finetuner
    finetuner = SFT(base_model_id, output_dir)
    
    # Load dataset
    finetuner.load_datasets(train_csv, test_csv)

    # Initialize tokenizer and model
    finetuner.initialize_tokenizer()
    finetuner.tokenize_data()
    finetuner.initialize_model()

    # Set up LoRA and train the model
    finetuner.initialize_peft()
    finetuner.train_model()

    # # Save model to MLflow
    # finetuner.save_model()
