import os
import re
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

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
        Builds a textual representation of a single flow.
        This keeps only a subset of the most relevant fields to avoid
        extremely long sequences.
        """
        parts = [
            f"flow_id {row.get('Flow ID', '')}",
            f"src {row.get('Src IP', '')} port {row.get('Src Port', '')}",
            f"dst {row.get('Dst IP', '')} port {row.get('Dst Port', '')}",
            f"proto {row.get('Protocol', '')}",
            f"duration {row.get('Flow Duration', '')}",
            f"fwd_pkts {row.get('Total Fwd Packet', '')}",
            f"bwd_pkts {row.get('Total Bwd packets', '')}",
            f"fwd_bytes {row.get('Total Length of Fwd Packet', '')}",
            f"bwd_bytes {row.get('Total Length of Bwd Packet', '')}",
            f"attack {row.get('Attack Name', '')}",
        ]
        return " ".join(str(p) for p in parts)

    def create_sessions(self):
        """
        Creates sessions from the Edge-IIoTSet ransomware and benign flows.

        For simplicidade e eficiência inicial, cada fluxo é tratado como
        uma sessão individual, com seu conteúdo textual derivado dos campos
        principais. Isso já é suficiente para treinar o LLM para distinguir
        tráfego benigno de ransomware nesse contexto.
        """
        benign_df, ransomware_df = self._load_raw_csvs()

        # Garante tipos consistentes de rótulo
        benign_df["Label"] = 0
        ransomware_df["Label"] = 1

        combined_df = pd.concat([benign_df, ransomware_df], ignore_index=True)

        # Constrói a coluna textual "Content" a partir de um subconjunto de campos
        print("Building textual representation for each flow...")
        combined_df["Content"] = combined_df.apply(self._build_text_from_row, axis=1)

        # Mantém apenas as colunas necessárias para o restante do pipeline
        final_df = combined_df[["Content", "Label"]].copy()

        # Embaralha e faz split treino/teste
        final_df = final_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        train_len = int(0.8 * len(final_df))
        train_df = final_df[:train_len]
        test_df = final_df[train_len:]

        os.makedirs(self.processed_path, exist_ok=True)
        train_df.to_csv(os.path.join(self.processed_path, "train.csv"), index=False)
        test_df.to_csv(os.path.join(self.processed_path, "test.csv"), index=False)

        print(f"Created train.csv and test.csv in {self.processed_path}")

    def preprocess_and_sanitize(self):
        """
        Applies regex-based sanitization to the Content field.

        We mask sensitive / high-cardinality tokens such as:
          - IP addresses
          - Long numeric IDs
        """
        regex_patterns = [
            r"\d+\.\d+\.\d+\.\d+",  # IP addresses
            r"(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$",  # generic numbers
        ]

        def apply_regex(line):
            text = str(line)
            for pattern in regex_patterns:
                text = re.sub(pattern, "<*>", text)
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

        dataset = Dataset.from_pandas(df_normal[["Content"]].rename(columns={"Content": "text"}))

        tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        tokenizer.pad_token = tokenizer.eos_token

        def preprocess_function(examples):
            examples["text"] = [text + tokenizer.eos_token for text in examples["text"]]
            # Use a shorter maximum sequence length to reduce VRAM usage.
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=256,
            )

        tokenized = dataset.map(preprocess_function, batched=True, remove_columns=["text"])
        tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]}, batched=True)

        final_dataset = DatasetDict({"train": tokenized})
        final_dataset.save_to_disk(self.tokenized_path)
        print(f"Tokenized Edge-IIoTSet ransomware dataset saved to {self.tokenized_path}")
