import os
from dotenv import load_dotenv
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import wandb
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("him_training")

# Load environment variables
load_dotenv()

class HIMTrainer:
    def __init__(self):
        self.model_name = os.getenv("MODEL_REPOSITORY", "TeleologyHI/HIM-self")
        self.base_model = os.getenv("BASE_MODEL", "deepseek-ai/deepseek-llm-7b-base")
        self.dataset_name = "TeleologyHI/consciousness-dataset"
        self.batch_size = int(os.getenv("TRAINING_BATCH_SIZE", "8"))
        self.learning_rate = float(os.getenv("LEARNING_RATE", "2e-5"))
        self.num_epochs = int(os.getenv("NUM_EPOCHS", "3"))
        
        # Configure wandb
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if not wandb_api_key:
            raise ValueError("WANDB_API_KEY not found in environment variables")
        
        os.environ["WANDB_API_KEY"] = wandb_api_key
        wandb.init(
            project="him-training",
            entity="TeleologyHI",
            config={
                "model_name": self.model_name,
                "base_model": self.base_model,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs
            }
        )
        
        # Configure HuggingFace token
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")

    def prepare_dataset(self):
        logger.info("Loading dataset...")
        dataset = load_dataset(self.dataset_name)
        
        # Tokenize dataset
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        def tokenize_function(examples):
            return tokenizer(
                examples["prompt"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        return tokenized_dataset, tokenizer

    def train(self):
        logger.info("Starting training setup...")
        
        # Prepare dataset and tokenizer
        tokenized_dataset, tokenizer = self.prepare_dataset()
        
        # Load model
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=self.learning_rate,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="epoch",
            push_to_hub=True,
            hub_model_id=self.model_name,
            hub_strategy="every_save",
            report_to="wandb",
            hub_token=os.getenv("HUGGINGFACE_TOKEN")
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=data_collator,
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Push final model to hub
        logger.info("Saving final model...")
        trainer.push_to_hub()
        
        logger.info("Training completed successfully!")

if __name__ == "__main__":
    trainer = HIMTrainer()
    trainer.train()

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import wandb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("him_training")

class HIMTrainer:
    def __init__(self):
        self.model_name = os.getenv("MODEL_REPOSITORY", "TeleologyHI/HIM-self")
        self.base_model = os.getenv("BASE_MODEL", "deepseek-ai/deepseek-llm-7b-base")
        self.dataset_name = "TeleologyHI/consciousness-dataset"
        self.batch_size = int(os.getenv("TRAINING_BATCH_SIZE", "8"))
        self.learning_rate = float(os.getenv("LEARNING_RATE", "2e-5"))
        self.num_epochs = int(os.getenv("NUM_EPOCHS", "3"))
        
        # Configure wandb with API key from .env
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if not wandb_api_key:
            raise ValueError("WANDB_API_KEY not found in environment variables")
        
        os.environ["WANDB_API_KEY"] = wandb_api_key
        wandb.init(project="him-training", entity="TeleologyHI")

    def prepare_dataset(self):
        logger.info("Loading dataset...")
        dataset = load_dataset(self.dataset_name)
        
        # Tokenize dataset
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        tokenizer.pad_token = tokenizer.eos_token
        
        def tokenize_function(examples):
            return tokenizer(
                examples["prompt"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        return tokenized_dataset, tokenizer

    def train(self):
        logger.info("Starting training setup...")
        
        # Prepare dataset and tokenizer
        tokenized_dataset, tokenizer = self.prepare_dataset()
        
        # Load model
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=self.learning_rate,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="epoch",
            push_to_hub=True,
            hub_model_id=self.model_name,
            hub_strategy="every_save",
            report_to="wandb"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=data_collator,
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Push final model to hub
        logger.info("Saving final model...")
        trainer.push_to_hub()
        
        logger.info("Training completed successfully!")

if __name__ == "__main__":
    trainer = HIMTrainer()
    trainer.train()

