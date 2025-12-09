import os
import re
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# Mantendo a referência original, assumindo que BaseProcessor existe no seu projeto
from .base_processor import BaseProcessor

class EdgeRansomwareProcessor(BaseProcessor):
    """
    Data processor for the CIC-BCCC-NRC-Edge-IIoTSet-2022 ransomware subset.
    
    IMPROVEMENTS:
    - Feature Selection: Includes statistical features (IAT, Variance, Flags) critical for 
      detecting ransomware encryption and scanning behavior.
    - Sanitization: Masks ONLY IP addresses, preserving numerical metrics so the LLM 
      can learn magnitude and distributions.
    """

    def _load_raw_csvs(self):
        """
        Loads benign and ransomware CSVs from the raw directory.
        """
        benign_path = os.path.join(self.raw_path, "Benign%20Traffic.csv")
        ransomware_path = os.path.join(self.raw_path, "Ransomware.csv")

        if not os.path.exists(benign_path):
            raise FileNotFoundError(
                f"{benign_path} not found. Make sure Benign%20Traffic.csv is in {self.raw_path}"
            )
        if not os.path.exists(ransomware_path):
            raise FileNotFoundError(
                f"{ransomware_path} not found. Make sure Ransomware.csv is in {self.raw_path}"
            )

        print(f"Loading benign flows from: {benign_path}")
        benign_df = pd.read_csv(benign_path)

        print(f"Loading ransomware flows from: {ransomware_path}")
        ransomware_df = pd.read_csv(ransomware_path)

        return benign_df, ransomware_df

    def _build_text_from_row(self, row):
        """
        Builds a textual representation of a single flow using RICH features.
        
        Ransomware indicators included:
        1. Encryption -> High 'Packet Length Variance' & 'Bytes/s'
        2. Automation -> Specific 'IAT' (Inter-Arrival Time) patterns
        3. Scanning -> 'SYN' and 'FIN' flag counts
        """
        
        # Helper to safely get value or 0 if missing/NaN
        def get_val(key):
            val = row.get(key, 0)
            if pd.isna(val) or str(val).strip() == '':
                return 0
            return val

        parts = [
            # Identificadores básicos (sem IP)
            f"proto {get_val('Protocol')}",
            
            # Métricas de Tempo (Crítico para automação)
            f"duration {get_val('Flow Duration')}",
            f"iat_mean {get_val('Flow IAT Mean')}",
            
            # Métricas de Volume (Crítico para exfiltração)
            f"pkts_fwd {get_val('Total Fwd Packet')}",
            f"pkts_bwd {get_val('Total Bwd packets')}",
            f"bytes_rate {get_val('Flow Bytes/s')}",
            
            # Métricas de Conteúdo/Criptografia (Variância alta = Criptografia)
            f"pkt_len_mean {get_val('Packet Length Mean')}",
            f"pkt_len_var {get_val('Packet Length Variance')}",
            
            # Flags TCP (Crítico para Scanning/Kill chain)
            f"syn_flags {get_val('SYN Flag Count')}",
            f"fin_flags {get_val('FIN Flag Count')}",
            f"rst_flags {get_val('RST Flag Count')}",
        ]
        return " ".join(str(p) for p in parts)

    def create_sessions(self):
        """
        Creates sessions from the Edge-IIoTSet ransomware and benign flows.
        """
        benign_df, ransomware_df = self._load_raw_csvs()

        # Garante tipos consistentes de rótulo
        benign_df["Label"] = 0
        ransomware_df["Label"] = 1

        combined_df = pd.concat([benign_df, ransomware_df], ignore_index=True)

        print("Building textual representation with ENHANCED features...")
        # Aplica a nova lógica de construção de texto
        combined_df["Content"] = combined_df.apply(self._build_text_from_row, axis=1)

        # Mantém apenas as colunas necessárias
        final_df = combined_df[["Content", "Label"]].copy()

        # Embaralha e faz split treino/teste
        final_df = final_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        train_len = int(0.8 * len(final_df))
        train_df = final_df[:train_len]
        test_df = final_df[train_len:]

        os.makedirs(self.processed_path, exist_ok=True)
        train_df.to_csv(os.path.join(self.processed_path, "train.csv"), index=False)
        test_df.to_csv(os.path.join(self.processed_path, "test.csv"), index=False)

        print(f"Created train.csv ({len(train_df)} samples) and test.csv ({len(test_df)} samples)")

    def preprocess_and_sanitize(self):
        """
        Applies TARGETED sanitization.
        
        Changes:
        - Masks IP addresses (Privacy)
        - KEEPS numbers (Metrics needed for detection)
        """
        # Regex para identificar IPs (IPv4)
        ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"

        def apply_regex(text):
            text = str(text)
            # Substitui IPs por um token genérico
            text = re.sub(ip_pattern, "IP_ADDR", text)
            # Remove espaços extras
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        print("Sanitizing: Masking IPs but PRESERVING numerical metrics...")
        for split in ["train", "test"]:
            file_path = os.path.join(self.processed_path, f"{split}.csv")
            if not os.path.exists(file_path):
                continue
                
            df = pd.read_csv(file_path)
            df["Content"] = df["Content"].apply(apply_regex)
            df.to_csv(file_path, index=False)
        print("Sanitization complete.")

    def tokenize_dataset(self):
        """
        Tokenizes the preprocessed data.
        Only normal data (Label == 0) is used for training in the initial phase if
        doing anomaly detection, BUT for supervised/classification fine-tuning, 
        ensure logic matches your training goal.
        
        Assuming Anomaly Detection (train on benign):
        """
        train_path = os.path.join(self.processed_path, "train.csv")
        try:
            df = pd.read_csv(train_path)
        except FileNotFoundError:
            print(f"Error: {train_path} not found.")
            return

        # ATENÇÃO: Se o seu objetivo é treinar o modelo para entender "o que é normal",
        # filtre apenas Label == 0. Se for classificação supervisionada, use tudo.
        # Mantendo lógica original de Anomaly Detection (apenas benigno):
        df_normal = df[df["Label"] == 0].copy()
        
        if df_normal.empty:
            print("Warning: No normal data (Label == 0) found for training.")
            return

        print(f"Tokenizing {len(df_normal)} benign samples...")
        dataset = Dataset.from_pandas(df_normal[["Content"]].rename(columns={"Content": "text"}))

        tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        tokenizer.pad_token = tokenizer.eos_token

        def preprocess_function(examples):
            examples["text"] = [text + tokenizer.eos_token for text in examples["text"]]
            
            # REDUZIDO max_length para 512. 
            # Logs de rede raramente passam disso e economiza MUITA memória/tempo.
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512, 
            )

        tokenized = dataset.map(preprocess_function, batched=True, remove_columns=["text"])
        tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]}, batched=True)

        final_dataset = DatasetDict({"train": tokenized})
        final_dataset.save_to_disk(self.tokenized_path)
        print(f"Tokenized dataset saved to {self.tokenized_path}")