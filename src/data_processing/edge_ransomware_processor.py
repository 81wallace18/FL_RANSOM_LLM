import os
import re
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from .base_processor import BaseProcessor


class EdgeRansomwareProcessor(BaseProcessor):
    """
    Data processor for the CIC-BCCC-NRC-Edge-IIoTSet-2022 ransomware subset.

    It uses:
      - data/ids_ransomware/raw/Benign%20Traffic.csv      -> Label = 0
      - data/ids_ransomware/raw/Ransomware.csv           -> Label = 1
    and converts flow-level tabular data into textual sessions for the LLM.
    """

    def _load_raw_csvs(self):
        """
        Loads benign and ransomware CSVs from the raw directory.
        """
        benign_path = os.path.join(self.raw_path, "Benign%20Traffic.csv")
        ransomware_path = os.path.join(self.raw_path, "Ransomware.csv")

        if not os.path.exists(benign_path):
            raise FileNotFoundError(
                f"{benign_path} not found. Make sure Benign%%20Traffic.csv is in {self.raw_path}"
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
        Builds a richer textual representation of a single flow.
        """

        def get_val(col_name):
            return row.get(col_name, "")

        parts = [
            f"proto {get_val('Protocol')}",
            f"dur {get_val('Flow Duration')}",
            f"iat {get_val('Flow IAT Mean')}",
            f"pkts_fwd {get_val('Total Fwd Packet')}",
            f"pkts_bwd {get_val('Total Bwd packets')}",
            f"rate {get_val('Flow Bytes/s')}",
            f"len_mean {get_val('Packet Length Mean')}",
            f"len_var {get_val('Packet Length Variance')}",
            f"syn {get_val('SYN Flag Count')}",
            f"fin {get_val('FIN Flag Count')}",
        ]
        return " ".join(str(p) for p in parts)

    def create_sessions(self):
        """
        Creates sessions from the Edge-IIoTSet ransomware and benign flows.
        Groups by Src IP and chunks flows into sessions to capture temporal context.
        """
        benign_df, ransomware_df = self._load_raw_csvs()

        # Garante tipos consistentes de rótulo
        benign_df["Label"] = 0
        ransomware_df["Label"] = 1

        combined_df = pd.concat([benign_df, ransomware_df], ignore_index=True)
        
        # Parse Timestamp
        print("Parsing timestamps...")
        combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], format='%d/%m/%Y %I:%M:%S %p', errors='coerce')
        combined_df = combined_df.dropna(subset=['Timestamp'])
        
        # Build textual representation
        print("Building textual representation for each flow...")
        combined_df["FlowText"] = combined_df.apply(self._build_text_from_row, axis=1)

        sessions = []
        labels = []
        src_ips = []
        timestamps = [] # Start time of session

        # Chunking configuration
        chunk_size = 20
        sep = " ;-; "

        print("Grouping by Src IP and creating sessions...")
        for src_ip, group in combined_df.groupby('Src IP'):
            group = group.sort_values('Timestamp')
            
            group_texts = group['FlowText'].values
            group_labels = group['Label'].values
            group_times = group['Timestamp'].values
            
            for i in range(0, len(group_texts), chunk_size):
                chunk_texts = group_texts[i:i+chunk_size]
                chunk_labels = group_labels[i:i+chunk_size]
                chunk_times = group_times[i:i+chunk_size]

                # Join flows into one session string
                session_str = sep.join(chunk_texts)
                
                # Malicious if any flow in chunk is malicious
                is_malicious = 1 if 1 in chunk_labels else 0
                
                sessions.append(session_str)
                labels.append(is_malicious)
                src_ips.append(src_ip)
                timestamps.append(chunk_times[0]) # Use start time of first flow

        final_df = pd.DataFrame({
            'Content': sessions,
            'Label': labels,
            'Src IP': src_ips,
            'Timestamp': timestamps
        })

        print(f"Total sessions created: {len(final_df)}")
        print(f"Label distribution:\n{final_df['Label'].value_counts()}")

        # Stratified Split
        train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42, stratify=final_df['Label'])

        os.makedirs(self.processed_path, exist_ok=True)
        train_df.to_csv(os.path.join(self.processed_path, "train.csv"), index=False)
        test_df.to_csv(os.path.join(self.processed_path, "test.csv"), index=False)

        print(f"Created train.csv ({len(train_df)}) and test.csv ({len(test_df)}) in {self.processed_path}")

    def preprocess_and_sanitize(self):
        """
        Applies regex-based sanitization to the Content field.

        We mask sensitive / high-cardinality tokens such as:
          - IP addresses
          - Long numeric IDs
        """
        # Nova sanitização: mantém números (duração, bytes, IAT etc.) para que
        # o modelo consiga capturar magnitudes e padrões estatísticos, mas
        # mascara endereços IP para preservar privacidade.
        ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"

        def apply_regex(text):
            text = str(text)
            # Substitui apenas IPs dentro do texto por um token genérico.
            text = re.sub(ip_pattern, "IP_ADDR", text)
            return text

        print("Sanitizing Edge-IIoTSet ransomware train and test sets...")
        for split in ["train", "test"]:
            file_path = os.path.join(self.processed_path, f"{split}.csv")
            df = pd.read_csv(file_path)
            df["Content"] = df["Content"].apply(apply_regex)
            df.to_csv(file_path, index=False)
        print("Sanitization complete.")

    def tokenize_dataset(self):
        """
        Tokenizes the preprocessed data using the configured model's tokenizer.
        Only normal data (Label == 0) is used for training, following the
        anomaly-detection-as-language-modeling approach.
        """
        train_path = os.path.join(self.processed_path, "train.csv")
        try:
            df = pd.read_csv(train_path)
        except FileNotFoundError:
            print(f"Error: {train_path} not found. Did create_sessions run correctly?")
            return

        df_normal = df[df["Label"] == 0].copy()
        if df_normal.empty:
            print("Warning: No normal data (Label == 0) found for training.")
            return

        # Mantém metadados úteis para estratégias Non-IID por dispositivo no FL.
        # Essas colunas NÃO são usadas no forward do modelo (Trainer remove colunas
        # não utilizadas quando remove_unused_columns=True).
        dataset = Dataset.from_pandas(
            df_normal[["Content", "Src IP", "Timestamp"]].rename(
                columns={"Content": "text", "Src IP": "src_ip", "Timestamp": "timestamp"}
            )
        )

        tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        tokenizer.pad_token = tokenizer.eos_token

        def preprocess_function(examples):
            examples["text"] = [text + tokenizer.eos_token for text in examples["text"]]
            max_len = int(self.config.get("max_length", self.config.get("eval_max_length", 1024)))
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_len,
            )

        tokenized = dataset.map(preprocess_function, batched=True, remove_columns=["text"])
        tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]}, batched=True)

        final_dataset = DatasetDict({"train": tokenized})
        final_dataset.save_to_disk(self.tokenized_path)
        print(f"Tokenized Edge-IIoTSet ransomware dataset saved to {self.tokenized_path}")
