import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)

class ModelFineTuner:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def fine_tune_llm(self, dataset: Dataset, output_dir: str):
        """Fine-tune a pre-trained model on a custom dataset."""
        try:
            # Prepare training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=3,
                per_device_train_batch_size=4,
                save_steps=1000,
                save_total_limit=2,
                learning_rate=2e-5,
                warmup_steps=500,
                logging_dir="./logs"
            )
            
            # Prepare dataset
            processed_dataset = self._preprocess_dataset(dataset)
            
            # Initialize trainer
            trainer = self._setup_trainer(processed_dataset, training_args)
            
            # Train model
            trainer.train()
            
            logger.info("Successfully completed fine-tuning")
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to fine-tune model: {str(e)}")
            raise
            
    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Preprocess and tokenize the dataset."""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
            
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
        
    def _setup_trainer(self, dataset, training_args):
        """Set up the trainer with appropriate callbacks and configuration."""
        # Implement trainer setup with learning rate scheduling and checkpointing
        from transformers import Trainer, TrainerCallback
        
        class CustomCallback(TrainerCallback):
            def on_epoch_end(self, args, state, control, **kwargs):
                logger.info(f"Completed epoch {state.epoch}")
                
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            callbacks=[CustomCallback()]
        )
