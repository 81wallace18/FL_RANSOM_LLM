import os
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForMaskedLM,
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import PeftModel
from datasets import load_from_disk

class ClientTrainer:
    """
    Manages the training process for a single client in a federated learning round.
    """
    def __init__(self, client_id, config):
        self.client_id = client_id
        self.config = config
        self.model_name = config['model_name']
        self.use_lora = config['lora']

    def _load_model_for_training(self, round_num):
        """Loads the global model from the previous round and prepares it for training."""
        model_path = os.path.join(
            self.config['results_path'],
            self.config['simulation_name'],
            f'round_{round_num - 1}',
            'global_model'
        )
        print(f"Client {self.client_id}: Loading model from {model_path}")

        quantization_config = {
            'load_in_4bit': True,
            'bnb_4bit_quant_type': 'nf4',
            'bnb_4bit_compute_dtype': torch.float16,
            'bnb_4bit_use_double_quant': True,
        }

        if 'bert' in self.model_name.lower():
            if self.use_lora:
                model = AutoModelForMaskedLM.from_pretrained(self.model_name)
                model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
            else:
                model = AutoModelForMaskedLM.from_pretrained(model_path)
        else:
            if self.use_lora:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto", # Adicionado para carregar o modelo quantizado diretamente na GPU correta
                    **{k: v for k, v in quantization_config.items() if k != 'bnb_4bit_compute_dtype'}
                )
                model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path)
        
        return model

    def train(self, round_num, learning_rate):
        """
        Executes a single round of training for the client.
        """
        print(f"--- Starting training for Client {self.client_id} in Round {round_num} ---")
        
        # 1. Load model and dataset
        model = self._load_model_for_training(round_num)
        client_dataset_path = os.path.join(
            self.config['results_path'],
            self.config['simulation_name'],
            'client_data',
            f'client_{self.client_id}'
        )
        client_dataset = load_from_disk(client_dataset_path)
        client_dataset = client_dataset.shuffle(seed=round_num)

        # 2. Setup Training Arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(self.config['results_path'], self.config['simulation_name'], "client_training_output"),
            logging_dir=os.path.join(self.config['results_path'], self.config['simulation_name'], "logs"),
            logging_steps=self.config['max_steps'] + 1, # Avoid logging during training
            learning_rate=learning_rate,
            weight_decay=0.01,
            max_steps=self.config['max_steps'],
            fp16=True,
            optim='paged_adamw_8bit',
            per_device_train_batch_size=self.config['batch_size'],
            lr_scheduler_type=self.config['lr_scheduler_type'],
            save_strategy="no", # We save manually
            #evaluation_strategy="no",
        )

        # 3. Setup Trainer
        if 'bert' in self.model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=True, mlm_probability=0.15
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=client_dataset,
                data_collator=data_collator
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=client_dataset,
            )
        
        # 4. Run Training
        trainer.train()
        print(f"--- Client {self.client_id} training complete. ---")

        # 5. Extract LoRA adapters to CPU and return them
        # This prevents VRAM overflow on the server by not returning the whole model.
        if self.use_lora:
            cpu_adapters = {
                name: param.to('cpu')
                for name, param in model.state_dict().items()
                if "lora_" in name
            }
        else:
            # For full fine-tuning, return the entire state dict on CPU
            cpu_adapters = {name: param.to('cpu') for name, param in model.state_dict().items()}

        # Explicitly free up VRAM
        del model
        del trainer
        torch.cuda.empty_cache()

        return cpu_adapters
