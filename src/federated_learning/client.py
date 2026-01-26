import os
import json
from dataclasses import dataclass
from typing import Any
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from peft import PeftModel
from datasets import load_from_disk

from src.utils.hf import hf_from_pretrained_kwargs


@dataclass
class CausalLMDataCollatorWithPadding:
    """
    Dynamic padding + label masking for causal language modeling.

    We intentionally mask padded positions using attention_mask (not pad_token_id),
    because this project sets pad_token = eos_token for GPT-like tokenizers.
    """

    tokenizer: Any
    pad_to_multiple_of: int | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


class FedProxTrainer(Trainer):
    """
    Custom Trainer that implements FedProx by adding a proximal term to the loss.

    FedProx adds a regularization term that penalizes deviation from the global model:
    L_total = L_task + (mu/2) * ||w - w_global||^2

    This helps reduce client drift in non-IID scenarios.
    Reference: https://arxiv.org/abs/1812.06127
    """

    def __init__(self, global_state_dict=None, fedprox_mu=0.01, **kwargs):
        super().__init__(**kwargs)
        self.global_state_dict = global_state_dict
        self.fedprox_mu = fedprox_mu

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the total loss = task loss + proximal term.
        """
        # Compute standard task loss
        outputs = model(**inputs)
        task_loss = outputs.loss

        # Add proximal term if FedProx is enabled (mu > 0 and global weights available)
        if self.fedprox_mu > 0 and self.global_state_dict is not None:
            proximal_loss = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.global_state_dict:
                    global_param = self.global_state_dict[name].to(param.device)
                    proximal_loss += ((param - global_param) ** 2).sum()

            total_loss = task_loss + (self.fedprox_mu / 2.0) * proximal_loss
        else:
            total_loss = task_loss

        return (total_loss, outputs) if return_outputs else total_loss


class ClientTrainer:
    """
    Manages the training process for a single client in a federated learning round.
    """

    def __init__(self, client_id, config, gpu_id=0):
        self.client_id = client_id
        self.config = config
        self.gpu_id = gpu_id  # Armazena o ID da GPU
        self.model_name = config["model_name"]
        self.use_lora = config["lora"]

    def _load_model_for_training(self, round_num):
        """Loads the global model from the previous round and prepares it for training."""
        hf_kwargs = hf_from_pretrained_kwargs(self.config)
        model_path = os.path.join(
            self.config["results_path"],
            self.config["simulation_name"],
            f"round_{round_num - 1}",
            "global_model",
        )
        print(f"Client {self.client_id}: Loading model from {model_path}")

        legacy_4bit = bool(self.config.get("legacy_4bit_training", False))

        if "bert" in self.model_name.lower():
            if self.use_lora:
                model = AutoModelForMaskedLM.from_pretrained(
                    self.model_name, **hf_kwargs
                )
                if torch.cuda.is_available():
                    model = model.to(f"cuda:{self.gpu_id}")
                model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
            else:
                model = AutoModelForMaskedLM.from_pretrained(model_path, **hf_kwargs)
        else:
            if self.use_lora:
                extra_kwargs = {}
                if legacy_4bit:
                    # Match legado/utils.py quantization_config (NF4 4-bit).
                    extra_kwargs.update(
                        {
                            "load_in_4bit": True,
                            "bnb_4bit_quant_type": "nf4",
                            "bnb_4bit_use_double_quant": True,
                        }
                    )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    **extra_kwargs,
                    **hf_kwargs,
                )
                # In legacy 4-bit mode, avoid forcing .to() (bnb/accelerate handles placement).
                if torch.cuda.is_available() and not legacy_4bit:
                    model = model.to(f"cuda:{self.gpu_id}")
                model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path, **hf_kwargs)

        return model

    def train(self, round_num, learning_rate):
        """
        Executes a single round of training for the client.
        Supports FedProx regularization when fedprox_mu > 0 in config.
        """
        print(
            f"--- Starting training for Client {self.client_id} in Round {round_num} ---"
        )

        # 1. Load model and dataset
        model = self._load_model_for_training(round_num)
        if bool(self.config.get("legacy_flresults_layout", False)):
            client_dataset_path = os.path.join(
                self.config["results_path"],
                self.config["simulation_name"],
                "round_0",
                f"client_{self.client_id}",
            )
        else:
            client_dataset_path = os.path.join(
                self.config["results_path"],
                self.config["simulation_name"],
                "client_data",
                f"client_{self.client_id}",
            )
        client_dataset = load_from_disk(client_dataset_path)
        if len(client_dataset) == 0:
            print(f"  Client {self.client_id}: No local samples. Skipping training.")
            return None
        client_dataset = client_dataset.shuffle(seed=round_num)

        # 2. Capture global model state for FedProx (before training)
        fedprox_mu = self.config.get("fedprox_mu", 0.0)
        global_state_dict = None
        if fedprox_mu > 0:
            # Clone the global model parameters (only trainable ones for efficiency)
            global_state_dict = {
                name: param.data.clone().detach()
                for name, param in model.named_parameters()
                if param.requires_grad
            }
            print(f"  FedProx enabled with mu={fedprox_mu}")

        # 3. Setup Training Arguments
        legacy_training_args = bool(self.config.get("legacy_training_args", False))
        if legacy_training_args:
            training_args = TrainingArguments(
                output_dir="./fl-results",
                logging_dir="./logs",
                logging_steps=self.config["max_steps"] + 11,
                learning_rate=learning_rate,
                weight_decay=0.01,
                max_steps=self.config["max_steps"],
                num_train_epochs=1,
                save_steps=1000,
                evaluation_strategy="steps",
                eval_steps=self.config["max_steps"] + 1,
                fp16=True,
                optim="paged_adamw_8bit",
                per_device_train_batch_size=self.config["batch_size"],
                gradient_accumulation_steps=self.config.get(
                    "gradient_accumulation_steps", 1
                ),
                lr_scheduler_type=self.config.get(
                    "trainer_lr_scheduler_type", self.config["lr_scheduler_type"]
                ),
                warmup_ratio=float(self.config.get("warmup_ratio", 0.0)),
                max_grad_norm=float(self.config.get("max_grad_norm", 1.0)),
                save_strategy="no",
            )
        else:
            training_args = TrainingArguments(
                output_dir=os.path.join(
                    self.config["results_path"],
                    self.config["simulation_name"],
                    "client_training_output",
                ),
                logging_dir=os.path.join(
                    self.config["results_path"], self.config["simulation_name"], "logs"
                ),
                logging_steps=self.config["max_steps"]
                + 1,  # Avoid logging during training
                learning_rate=learning_rate,
                weight_decay=0.01,
                max_steps=self.config["max_steps"],
                fp16=True,
                optim="paged_adamw_8bit",
                per_device_train_batch_size=self.config["batch_size"],
                gradient_accumulation_steps=self.config.get(
                    "gradient_accumulation_steps", 1
                ),
                lr_scheduler_type=self.config.get(
                    "trainer_lr_scheduler_type", self.config["lr_scheduler_type"]
                ),
                warmup_ratio=float(self.config.get("warmup_ratio", 0.0)),
                max_grad_norm=float(self.config.get("max_grad_norm", 1.0)),
                save_strategy="no",  # We save manually
            )

        # 4. Setup Trainer (FedProxTrainer if mu > 0, else standard Trainer)
        if "bert" in self.model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, **hf_from_pretrained_kwargs(self.config)
            )
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=True, mlm_probability=0.15
            )
            if fedprox_mu > 0:
                trainer = FedProxTrainer(
                    global_state_dict=global_state_dict,
                    fedprox_mu=fedprox_mu,
                    model=model,
                    args=training_args,
                    train_dataset=client_dataset,
                    data_collator=data_collator,
                )
            else:
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=client_dataset,
                    data_collator=data_collator,
                )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, **hf_from_pretrained_kwargs(self.config)
            )
            tokenizer.pad_token = tokenizer.eos_token

            # Legacy mode: don't use data_collator (like original utils.py)
            use_legacy_trainer = self.config.get("use_legacy_trainer", False)
            if use_legacy_trainer:
                data_collator = None
            else:
                data_collator = CausalLMDataCollatorWithPadding(tokenizer=tokenizer)

            if fedprox_mu > 0:
                trainer = FedProxTrainer(
                    global_state_dict=global_state_dict,
                    fedprox_mu=fedprox_mu,
                    model=model,
                    args=training_args,
                    train_dataset=client_dataset,
                    data_collator=data_collator,
                    tokenizer=tokenizer if use_legacy_trainer else None,
                )
            else:
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=client_dataset,
                    data_collator=data_collator,
                    tokenizer=tokenizer if use_legacy_trainer else None,
                )

            # Legacy behavior: eval_dataset=client_dataset when evaluation is enabled.
            if (
                legacy_training_args
                and getattr(training_args, "evaluation_strategy", "no") != "no"
            ):
                trainer.eval_dataset = client_dataset

        # 5. Run Training
        trainer.train()
        print(f"--- Client {self.client_id} training complete. ---")

        # 5. Extract LoRA adapters to CPU and return them
        # This prevents VRAM overflow on the server by not returning the whole model.
        if self.use_lora:
            cpu_adapters = {
                name: param.to("cpu")
                for name, param in model.state_dict().items()
                if "lora_" in name
            }
        else:
            # For full fine-tuning, return the entire state dict on CPU
            cpu_adapters = {
                name: param.to("cpu") for name, param in model.state_dict().items()
            }

        # Explicitly free up VRAM
        del model
        del trainer
        torch.cuda.empty_cache()

        return cpu_adapters
