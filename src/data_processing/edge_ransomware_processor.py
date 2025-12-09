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
        Builds a richer textual representation of a single flow.

        Instead of apenas cabeçalhos e contagens brutas, expõe métricas que
        capturam o comportamento do tráfego, como IAT, taxa de bytes,
        variância de tamanho de pacotes e flags TCP. Esses campos funcionam
        como “assinaturas” de ransomware e foram fundamentais para melhorar
        o F1 em experimentos anteriores.
        """

        def get_val(col_name):
            return row.get(col_name, "")

        parts = [
            f"proto {get_val('Protocol')}",
            f"duration {get_val('Flow Duration')}",
            # Automação / periodicidade
            f"iat_mean {get_val('Flow IAT Mean')}",
            # Volume e velocidade de tráfego
            f"pkts_fwd {get_val('Total Fwd Packet')}",
            f"pkts_bwd {get_val('Total Bwd packets')}",
            f"bytes_rate {get_val('Flow Bytes/s')}",
            # Padrões de tamanho de pacote (entropia / criptografia)
            f"pkt_len_mean {get_val('Packet Length Mean')}",
            f"pkt_len_var {get_val('Packet Length Variance')}",
            # Comportamento de conexão (scanning / encerramento anômalo)
            f"syn_flags {get_val('SYN Flag Count')}",
            f"fin_flags {get_val('FIN Flag Count')}",
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

        # Mantém as colunas necessárias para o pipeline e avaliação temporal
        # (Content/Label para treino e métricas clássicas; Timestamp/Src IP/Attack Name
        #  para métricas de detecção precoce e análise por dispositivo).
        final_df = combined_df[
            ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Attack Name", "Content", "Label"]
        ].copy()

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
                max_length=1024,
            )

        tokenized = dataset.map(preprocess_function, batched=True, remove_columns=["text"])
        tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]}, batched=True)

        final_dataset = DatasetDict({"train": tokenized})
        final_dataset.save_to_disk(self.tokenized_path)
        print(f"Tokenized Edge-IIoTSet ransomware dataset saved to {self.tokenized_path}")
