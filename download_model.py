#!/usr/bin/env python3
"""
Script para baixar o modelo e tokenizer para uso local
Execute este script no nó de login (com internet) antes de rodar no cluster
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

def download_model_for_hpc():
    """
    Baixa o modelo SmolLM-135M e tokenizer para uso offline no cluster
    """
    model_name = "HuggingFaceTB/SmolLM-135M"
    local_path = "./models/SmolLM-135M"

    print(f"Baixando modelo {model_name} para {local_path}...")

    # Criar diretório se não existir
    os.makedirs(local_path, exist_ok=True)

    # Baixar tokenizer
    print("Baixando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(local_path)

    # Baixar modelo
    print("Baixando modelo...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(local_path)

    print(f"Modelo salvo em: {local_path}")
    print("Agora você pode usar './models/SmolLM-135M' como model_name no config")

if __name__ == "__main__":
    download_model_for_hpc()