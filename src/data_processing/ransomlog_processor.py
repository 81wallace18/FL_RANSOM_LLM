import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import re
import os

from .base_processor import BaseProcessor

class RansomLogProcessor(BaseProcessor):
    """
    Data processor implementation for the RansomLog dataset.
    """
    def create_sessions(self):
        # TODO: Implement your logic to read raw RansomLog files.
        # This is highly specific to the format of your dataset.
        # Example:
        # raw_log_file = os.path.join(self.raw_path, 'ransomlog_events.csv')
        # df = pd.read_csv(raw_log_file)
        
        # TODO: Implement your session creation logic.
        # For example, grouping by a time window or process ID.
        # You need to produce a DataFrame with 'Content' and 'Label' columns.
        # 'Content' should be a string of log messages joined by the spliter from config.
        
        print("TODO: Implement create_sessions for RansomLog")
        # Placeholder: creating dummy files for demonstration purposes.
        # Replace this with your actual data loading and sessionizing logic.
        dummy_train = pd.DataFrame({
            'Content': ["file access C:\\Users\\a.doc" + self.config['session_window_spliter'] + "file encrypted C:\\Users\\a.doc.enc", "normal process started"],
            'Label': [1, 0]
        })
        dummy_test = pd.DataFrame({
            'Content': ["file access C:\\Users\\b.doc" + self.config['session_window_spliter'] + "file encrypted C:\\Users\\b.doc.enc", "normal process ended"],
            'Label': [1, 0]
        })
        
        dummy_train.to_csv(os.path.join(self.processed_path, 'train.csv'), index=False)
        dummy_test.to_csv(os.path.join(self.processed_path, 'test.csv'), index=False)
        print(f"Created dummy train.csv and test.csv in {self.processed_path}")


    def preprocess_and_sanitize(self):
        # TODO: Define regex patterns specific to your logs (e.g., Windows paths, usernames, registry keys).
        regex_patterns = [
            r"(C:(\\[\w\s.-]+)+)",  # Example: Windows file paths
            r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Example: Numbers
        ]

        def apply_regex(line):
            for r in regex_patterns:
                line = re.sub(r, '<*>', str(line))
            return line

        print("Sanitizing train and test sets...")
        for split in ['train', 'test']:
            file_path = os.path.join(self.processed_path, f'{split}.csv')
            df = pd.read_csv(file_path)
            df['Content'] = df['Content'].apply(apply_regex)
            df.to_csv(file_path, index=False)
        print("Sanitization complete.")

    def tokenize_dataset(self):
        # This function is largely reusable across datasets.
        train_path = os.path.join(self.processed_path, 'train.csv')
        try:
            df = pd.read_csv(train_path)
        except FileNotFoundError:
            print(f"Error: {train_path} not found. Did create_sessions run correctly?")
            return
        
        # The model is trained only on normal data (Label == 0)
        df_normal = df[df['Label'] == 0].copy()
        
        if df_normal.empty:
            print("Warning: No normal data (Label == 0) found for training. Tokenization skipped.")
            return

        dataset = Dataset.from_pandas(df_normal[['Content']].rename(columns={'Content': 'text'}))
        
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        tokenizer.pad_token = tokenizer.eos_token

        def preprocess_function(examples):
            examples["text"] = [text + tokenizer.eos_token for text in examples["text"]]
            # You might want to adjust max_length based on your session sizes
            max_len = int(self.config.get("max_length", self.config.get("eval_max_length", 1024)))
            return tokenizer(examples["text"], truncation=True, max_length=max_len)

        tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['text'])
        
        final_dataset = DatasetDict({"train": tokenized_datasets})
        final_dataset.save_to_disk(self.tokenized_path)
        print(f"Tokenized dataset saved to {self.tokenized_path}")
