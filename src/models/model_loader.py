import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from peft import LoraConfig, get_peft_model

def initialize_global_model(config):
    """
    Initializes a model and tokenizer from Hugging Face, applies LoRA if configured,
    and saves it as the initial global model for round 0.

    Args:
        config (dict): The experiment configuration dictionary.

    Returns:
        tuple: A tuple containing the initialized model and tokenizer.
    """
    model_name = config['model_name']
    use_lora = config['lora']
    
    print(f"Initializing model: {model_name}")

    # Determine model type (e.g., BERT vs. GPT-like)
    if 'bert' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)

    if use_lora:
        print(f"Applying LoRA with rank={config['lora_rank']}...")
        target_modules = config.get("lora_target_modules", None)
        if isinstance(target_modules, str):
            # allow comma-separated string in YAML
            target_modules = [m.strip() for m in target_modules.split(",") if m.strip()]
        if target_modules:
            print(f"  LoRA target_modules: {target_modules}")
        lora_config = LoraConfig(
            r=config['lora_rank'],
            lora_alpha=config['lora_rank'] * config['lora_alpha_multiplier'],
            lora_dropout=config['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        print("LoRA applied successfully.")
        model.print_trainable_parameters()

    # Save the initial model state for round 0
    initial_model_path = os.path.join(
        config['results_path'], 
        config['simulation_name'], 
        'round_0', 
        'global_model'
    )
    os.makedirs(initial_model_path, exist_ok=True)
    model.save_pretrained(initial_model_path)
    
    print(f"Initial global model saved to: {initial_model_path}")

    return model, tokenizer
