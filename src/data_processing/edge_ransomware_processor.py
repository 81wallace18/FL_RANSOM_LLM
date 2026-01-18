import os
import re
import pandas as pd
import numpy as np
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

    def _to_float(self, value):
        try:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return np.nan
            return float(value)
        except Exception:
            return np.nan

    def _compute_quantile_edges(self, values, num_bins, *, transform="none"):
        arr = np.array([self._to_float(v) for v in values], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None, transform

        transform = str(transform or "none").lower()
        if transform == "log1p":
            arr = np.log1p(np.maximum(arr, 0.0))
        else:
            transform = "none"

        qs = np.linspace(0.0, 1.0, int(num_bins) + 1)
        edges = np.quantile(arr, qs)
        edges = np.unique(edges)
        if edges.size < 2:
            return None, transform
        return edges, transform

    def _bin_token(self, value, edges, *, transform="none", prefix="b"):
        x = self._to_float(value)
        if not np.isfinite(x) or edges is None:
            return f"{prefix}na"

        if transform == "log1p":
            x = np.log1p(max(x, 0.0))

        # edges has length nbins+1; result bin in [0, nbins-1]
        idx = int(np.searchsorted(edges, x, side="right") - 1)
        idx = max(0, min(idx, int(len(edges) - 2)))
        return f"{prefix}{idx}"

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

        benign_df = self._apply_ip_filters(benign_df, label="benign")
        ransomware_df = self._apply_ip_filters(ransomware_df, label="ransomware")

        return benign_df, ransomware_df

    def _apply_ip_filters(self, df: pd.DataFrame, *, label: str) -> pd.DataFrame:
        """
        Optional IP filtering for stronger experimental rigor in device-based evaluation.

        Supported config keys:
          - filter_ipv4_only: bool (default False)
          - drop_zero_ips: bool (default False)
        """
        if df is None or df.empty:
            return df

        if "Src IP" not in df.columns:
            return df

        filter_ipv4_only = bool(self.config.get("filter_ipv4_only", False))
        drop_zero_ips = bool(self.config.get("drop_zero_ips", False))

        if not (filter_ipv4_only or drop_zero_ips):
            return df

        before = len(df)
        src = df["Src IP"].astype(str)

        mask = pd.Series(True, index=df.index)

        if drop_zero_ips:
            bad = {"0", "0.0.0.0", "::", ""}
            mask &= ~src.isin(bad)

        if filter_ipv4_only:
            ipv4_rx = re.compile(r"^(?:\\d{1,3}\\.){3}\\d{1,3}$")
            mask &= src.apply(lambda x: bool(ipv4_rx.match(x)))

        out = df.loc[mask].copy()
        after = len(out)
        if before != after:
            print(f"IP filter ({label}): {before} -> {after} rows (filter_ipv4_only={filter_ipv4_only}, drop_zero_ips={drop_zero_ips})")
        return out

    def _build_text_from_row_raw(self, row):
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

    def _build_text_from_row_binned(self, row, *, edges_by_key, transform_by_key):
        def get_val(col_name):
            return row.get(col_name, "")

        pkts_fwd = get_val("Total Fwd Packet")
        pkts_bwd = get_val("Total Bwd packets")
        pkts_total = self._to_float(pkts_fwd) + self._to_float(pkts_bwd)

        tokens = [
            f"proto={get_val('Protocol')}",
            f"dur={self._bin_token(get_val('Flow Duration'), edges_by_key.get('dur'), transform=transform_by_key.get('dur', 'none'))}",
            f"iat_mean={self._bin_token(get_val('Flow IAT Mean'), edges_by_key.get('iat_mean'), transform=transform_by_key.get('iat_mean', 'none'))}",
            f"pkts_fwd={self._bin_token(pkts_fwd, edges_by_key.get('pkts_fwd'), transform=transform_by_key.get('pkts_fwd', 'none'))}",
            f"pkts_bwd={self._bin_token(pkts_bwd, edges_by_key.get('pkts_bwd'), transform=transform_by_key.get('pkts_bwd', 'none'))}",
            f"pkts_total={self._bin_token(pkts_total, edges_by_key.get('pkts_total'), transform=transform_by_key.get('pkts_total', 'none'))}",
            f"bytes_rate={self._bin_token(get_val('Flow Bytes/s'), edges_by_key.get('bytes_rate'), transform=transform_by_key.get('bytes_rate', 'none'))}",
            f"pkt_len_mean={self._bin_token(get_val('Packet Length Mean'), edges_by_key.get('pkt_len_mean'), transform=transform_by_key.get('pkt_len_mean', 'none'))}",
            f"pkt_len_var={self._bin_token(get_val('Packet Length Variance'), edges_by_key.get('pkt_len_var'), transform=transform_by_key.get('pkt_len_var', 'none'))}",
            f"syn_flags={self._bin_token(get_val('SYN Flag Count'), edges_by_key.get('syn_flags'), transform=transform_by_key.get('syn_flags', 'none'))}",
            f"fin_flags={self._bin_token(get_val('FIN Flag Count'), edges_by_key.get('fin_flags'), transform=transform_by_key.get('fin_flags', 'none'))}",
        ]
        return " | ".join(str(t) for t in tokens)

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

        benign_train_frac = float(self.config.get("benign_train_fraction", 0.8))
        benign_train_frac = min(max(benign_train_frac, 0.0), 1.0)
        benign_calib_frac = float(self.config.get("benign_calibration_fraction", 0.0))
        benign_calib_frac = min(max(benign_calib_frac, 0.0), 1.0)

        cols_base = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Attack Name", "Label"]

        # Constrói a coluna textual "Content".
        # Modo padrão = raw (compatível com runs anteriores).
        content_mode = str(self.config.get("content_mode", "raw")).lower()
        print(f"Building textual representation for each flow (mode={content_mode})...")

        if content_mode == "binned":
            # Split paper-grade benign-only:
            # - Treino: apenas benigno (benign-only training)
            # - Teste: holdout benigno + TODO ransomware (evita "quebrar" o ataque no split)
            benign_shuffled = benign_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            train_len = int(benign_train_frac * len(benign_shuffled))
            calib_len = int(benign_calib_frac * len(benign_shuffled))
            if train_len + calib_len > len(benign_shuffled):
                calib_len = max(0, len(benign_shuffled) - train_len)

            benign_train = benign_shuffled.iloc[:train_len].copy()
            benign_calib = benign_shuffled.iloc[train_len:train_len + calib_len].copy()
            benign_holdout = benign_shuffled.iloc[train_len + calib_len:].copy()

            # Compute bin edges from benign training split ONLY (avoid leakage).
            num_bins = int(self.config.get("binning_num_bins", 32))
            transform = str(self.config.get("binning_transform", "log1p")).lower()

            feature_map = {
                "dur": "Flow Duration",
                "iat_mean": "Flow IAT Mean",
                "pkts_fwd": "Total Fwd Packet",
                "pkts_bwd": "Total Bwd packets",
                "bytes_rate": "Flow Bytes/s",
                "pkt_len_mean": "Packet Length Mean",
                "pkt_len_var": "Packet Length Variance",
                "syn_flags": "SYN Flag Count",
                "fin_flags": "FIN Flag Count",
            }

            edges_by_key = {}
            transform_by_key = {}
            for key, col in feature_map.items():
                edges, used_transform = self._compute_quantile_edges(benign_train[col].values, num_bins, transform=transform)
                edges_by_key[key] = edges
                transform_by_key[key] = used_transform

            # derived: pkts_total
            pkts_total_vals = (
                benign_train["Total Fwd Packet"].apply(self._to_float).values
                + benign_train["Total Bwd packets"].apply(self._to_float).values
            )
            edges, used_transform = self._compute_quantile_edges(pkts_total_vals, num_bins, transform="none")
            edges_by_key["pkts_total"] = edges
            transform_by_key["pkts_total"] = used_transform

            benign_train["Content"] = benign_train.apply(
                lambda r: self._build_text_from_row_binned(r, edges_by_key=edges_by_key, transform_by_key=transform_by_key),
                axis=1,
            )
            benign_calib["Content"] = benign_calib.apply(
                lambda r: self._build_text_from_row_binned(r, edges_by_key=edges_by_key, transform_by_key=transform_by_key),
                axis=1,
            )
            benign_holdout["Content"] = benign_holdout.apply(
                lambda r: self._build_text_from_row_binned(r, edges_by_key=edges_by_key, transform_by_key=transform_by_key),
                axis=1,
            )
            ransomware_part = ransomware_df.copy()
            ransomware_part["Content"] = ransomware_part.apply(
                lambda r: self._build_text_from_row_binned(r, edges_by_key=edges_by_key, transform_by_key=transform_by_key),
                axis=1,
            )

            train_df = benign_train[cols_base + ["Content"]].reset_index(drop=True)
            calib_df = benign_calib[cols_base + ["Content"]].reset_index(drop=True)
            test_df = pd.concat(
                [benign_holdout[cols_base + ["Content"]], ransomware_part[cols_base + ["Content"]]],
                ignore_index=True,
            ).sample(frac=1.0, random_state=42).reset_index(drop=True)
        else:
            # Raw mode: build from the full raw frames first, then split.
            benign_full = benign_df.copy()
            ransomware_full = ransomware_df.copy()
            benign_full["Content"] = benign_full.apply(self._build_text_from_row_raw, axis=1)
            ransomware_full["Content"] = ransomware_full.apply(self._build_text_from_row_raw, axis=1)

            benign_final = benign_full[cols_base + ["Content"]].sample(frac=1.0, random_state=42).reset_index(drop=True)
            train_len = int(benign_train_frac * len(benign_final))
            calib_len = int(benign_calib_frac * len(benign_final))
            if train_len + calib_len > len(benign_final):
                calib_len = max(0, len(benign_final) - train_len)

            train_df = benign_final[:train_len].reset_index(drop=True)
            calib_df = benign_final[train_len:train_len + calib_len].reset_index(drop=True)
            benign_test_df = benign_final[train_len + calib_len:].reset_index(drop=True)

            ransomware_final = ransomware_full[cols_base + ["Content"]].copy()
            test_df = pd.concat([benign_test_df, ransomware_final], ignore_index=True)
            test_df = test_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        os.makedirs(self.processed_path, exist_ok=True)
        train_df.to_csv(os.path.join(self.processed_path, "train.csv"), index=False)
        if benign_calib_frac > 0.0:
            calib_df.to_csv(os.path.join(self.processed_path, "calibration.csv"), index=False)
        test_df.to_csv(os.path.join(self.processed_path, "test.csv"), index=False)

        calib_benign = int((calib_df["Label"] == 0).sum()) if benign_calib_frac > 0.0 and "Label" in calib_df.columns else 0
        test_benign = int((test_df["Label"] == 0).sum()) if "Label" in test_df.columns else 0
        test_ransomware = int((test_df["Label"] == 1).sum()) if "Label" in test_df.columns else 0
        print(
            f"Created train.csv and test.csv in {self.processed_path} "
            f"(train benign={len(train_df)}, calib benign={calib_benign}, test benign={test_benign}, test ransomware={test_ransomware})"
        )

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
                truncation=True,
                max_length=max_len,
            )

        tokenized = dataset.map(preprocess_function, batched=True, remove_columns=["text"])

        final_dataset = DatasetDict({"train": tokenized})
        final_dataset.save_to_disk(self.tokenized_path)
        print(f"Tokenized Edge-IIoTSet ransomware dataset saved to {self.tokenized_path}")
