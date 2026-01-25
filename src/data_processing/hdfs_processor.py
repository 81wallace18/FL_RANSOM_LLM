import os
import re
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from .base_processor import BaseProcessor
from src.utils.hf import hf_from_pretrained_kwargs

class HDFSProcessor(BaseProcessor):
    """
    Data processor implementation for the HDFS dataset,
    migrated from the original project's scripts.
    """

    def _structure_log(self, log_file, regex, headers):
        """Helper function to parse raw log file into a DataFrame."""
        log_messages = []
        with open(log_file, 'r', encoding='latin-1') as fin:
            for line in fin:
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                except Exception:
                    pass
        return pd.DataFrame(log_messages, columns=headers)

    def create_sessions(self):
        """
        Reads raw HDFS.log, structures it, creates log sessions by BlockId,
        and saves train.csv and test.csv.
        """
        print("Structuring HDFS.log...")
        log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
        headers, regex = re.split(r'(<[^<>]+>)', log_format), ''
        for k in range(len(headers)):
            if k % 2 == 0:
                regex += re.sub(' +', '\\\s+', headers[k])
            else:
                header = headers[k].strip('<').strip('>')
                regex += f'(?P<{header}>.*?)'
        regex = re.compile('^' + regex + '$')
        headers = [h.strip('<').strip('>') for h in headers if h.startswith('<')]

        raw_log_path = os.path.join(self.raw_path, 'HDFS.log')
        if not os.path.exists(raw_log_path):
            raise FileNotFoundError(
                f"{raw_log_path} not found. "
                "Please download the HDFS dataset first. You may need to find and run 'download_hdfs.sh' from the original project."
            )
        
        df_log = self._structure_log(raw_log_path, regex, headers)

        print("Creating sessions by BlockId...")
        data_dict_content = defaultdict(list)
        for _, row in tqdm(df_log.iterrows(), total=len(df_log)):
            blk_list = re.findall(r'(blk_-?\d+)', row['Content'])
            for blk_id in set(blk_list):
                data_dict_content[blk_id].append(row["Content"])
        
        data_df = pd.DataFrame(list(data_dict_content.items()), columns=['BlockId', 'Content'])

        # Add labels
        label_file = os.path.join(self.raw_path, 'anomaly_label.csv')
        label_df = pd.read_csv(label_file)
        blk_label_dict = {row["BlockId"]: 1 if row["Label"] == "Anomaly" else 0 for _, row in label_df.iterrows()}
        data_df["Label"] = data_df["BlockId"].apply(lambda x: blk_label_dict.get(x))
        data_df = data_df.dropna(subset=['Label']) # Remove blocks without labels
        data_df['Label'] = data_df['Label'].astype(int)

        # Join content and split data
        data_df["Content"] = data_df["Content"].apply(lambda x: self.config['session_window_spliter'].join(x))
        
        # Shuffle and split
        data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_len = int(0.8 * len(data_df))
        train_df = data_df[:train_len]
        test_df = data_df[train_len:]

        train_df.to_csv(os.path.join(self.processed_path, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(self.processed_path, 'test.csv'), index=False)
        print(f"Created train.csv and test.csv in {self.processed_path}")

    def preprocess_and_sanitize(self):
        """Applies HDFS-specific regex to sanitize log content."""
        regex_patterns = [
            r"(?<=blk_)[-\d]+",
            r'\d+\.\d+\.\d+\.\d+',
            r"(/[-\w]+)+",
            r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',
        ]

        def apply_regex(line):
            for r in regex_patterns:
                line = re.sub(r, '<*>', str(line))
            return line

        print("Sanitizing HDFS train and test sets...")
        for split in ['train', 'test']:
            file_path = os.path.join(self.processed_path, f'{split}.csv')
            df = pd.read_csv(file_path)
            df['Content'] = df['Content'].apply(apply_regex)
            df.to_csv(file_path, index=False)
        print("Sanitization complete.")

    def tokenize_dataset(self):
        """Tokenizes the preprocessed HDFS data."""
        train_path = os.path.join(self.processed_path, 'train.csv')
        try:
            df = pd.read_csv(train_path)
        except FileNotFoundError:
            print(f"Error: {train_path} not found. Did create_sessions run correctly?")
            return
        
        df_normal = df[df['Label'] == 0].copy()
        
        if df_normal.empty:
            print("Warning: No normal data (Label == 0) found for training.")
            return

        dataset = Dataset.from_pandas(df_normal[['Content']].rename(columns={'Content': 'text'}))
        
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'], **hf_from_pretrained_kwargs(self.config))
        tokenizer.pad_token = tokenizer.eos_token

        def preprocess_function(examples):
            examples["text"] = [text + tokenizer.eos_token for text in examples["text"]]
            max_len = int(self.config.get("max_length", self.config.get("eval_max_length", 1024)))
            # Legacy mode: use padding_side="right" like original tokenize_dataset.py
            if self.config.get('use_legacy_tokenization', False):
                return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_len, padding_side="right")
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_len)

        tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['text'])

        # Add labels column (same as input_ids for language modeling)
        tokenized_datasets = tokenized_datasets.map(lambda x: {"labels": x["input_ids"]}, batched=True)
        
        final_dataset = DatasetDict({"train": tokenized_datasets})
        final_dataset.save_to_disk(self.tokenized_path)
        print(f"Tokenized HDFS dataset saved to {self.tokenized_path}")
