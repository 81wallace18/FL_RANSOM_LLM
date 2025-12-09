import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import re
import os
import numpy as np
from datetime import datetime, timedelta

from .base_processor import BaseProcessor

class RansomLogProcessor(BaseProcessor):
    """
    Data processor implementation for the RansomLog dataset.
    """
    def create_sessions(self):
        """
        Reads raw RansomLog files, groups them into sessions, and creates train/test splits.
        Supports multiple log formats and creates sessions based on process IDs or time windows.
        """
        print("Processing RansomLog dataset...")

        # Try to load different possible file formats
        raw_data = []

        # Option 1: CSV format
        csv_file = os.path.join(self.raw_path, 'ransomlog_events.csv')
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            if all(col in df.columns for col in ['timestamp', 'process_id', 'event_type', 'description']):
                raw_data = self._process_csv_format(df)
            elif all(col in df.columns for col in ['Content', 'Label']):
                # Already processed format
                self._split_processed_data(df)
                return

        # Option 2: JSON format
        json_file = os.path.join(self.raw_path, 'ransomlog_events.json')
        if os.path.exists(json_file) and not raw_data:
            df = pd.read_json(json_file)
            raw_data = self._process_json_format(df)

        # Option 3: Log file format
        log_file = os.path.join(self.raw_path, 'ransomlog.log')
        if os.path.exists(log_file) and not raw_data:
            raw_data = self._process_log_format(log_file)

        # Option 4: If no real data found, create enhanced realistic dummy data
        if not raw_data:
            print("Warning: No real data found. Creating enhanced synthetic dataset...")
            raw_data = self._create_enhanced_synthetic_data()

        # Convert to DataFrame
        df = pd.DataFrame(raw_data)

        # Split into train/test (80/20 stratified split)
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df['Label']
        )

        # Save processed files
        train_df.to_csv(os.path.join(self.processed_path, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(self.processed_path, 'test.csv'), index=False)

        print(f"Dataset processed: {len(train_df)} training samples, {len(test_df)} test samples")
        print(f"Class distribution - Train: {train_df['Label'].value_counts().to_dict()}")
        print(f"Class distribution - Test: {test_df['Label'].value_counts().to_dict()}")

    def _process_csv_format(self, df):
        """Process CSV format with timestamp, process_id, event_type, description"""
        sessions = []

        # Sort by timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Group by process_id to create sessions
        for process_id, group in df.groupby('process_id'):
            events = []
            for _, row in group.iterrows():
                event = f"{row['event_type']}: {row['description']}"
                events.append(event)

            # Join events with session splitter
            content = self.config['session_window_spliter'].join(events)

            # Determine label (anomalous if contains suspicious patterns)
            label = 1 if any(keyword in ' '.join(events).lower() for keyword in
                           ['encrypt', 'ransom', 'delete', 'modify', '.enc', '.locked']) else 0

            sessions.append({'Content': content, 'Label': label})

        return sessions

    def _process_json_format(self, df):
        """Process JSON log format"""
        sessions = []

        for _, row in df.iterrows():
            # Assuming JSON has 'events' array and 'is_anomaly' flag
            if isinstance(row.get('events'), list):
                content = self.config['session_window_spliter'].join(row['events'])
                label = 1 if row.get('is_anomaly', False) else 0
                sessions.append({'Content': content, 'Label': label})

        return sessions

    def _process_log_format(self, log_file):
        """Process raw log file format"""
        sessions = []
        current_session = []
        current_process = None
        session_start_time = None

        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Extract timestamp and process ID if available
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                process_match = re.search(r'PID:(\d+)', line)

                if process_match:
                    process_id = process_match.group(1)

                    # New process detected, save previous session
                    if current_process and process_id != current_process:
                        if current_session:
                            content = self.config['session_window_spliter'].join(current_session)
                            label = self._determine_anomaly(content)
                            sessions.append({'Content': content, 'Label': label})
                        current_session = []

                    current_process = process_id

                current_session.append(line)

        # Save last session
        if current_session:
            content = self.config['session_window_spliter'].join(current_session)
            label = self._determine_anomaly(content)
            sessions.append({'Content': content, 'Label': label})

        return sessions

    def _create_enhanced_synthetic_data(self):
        """Create enhanced synthetic ransomware and normal logs"""
        normal_events = [
            "process_started: winword.exe",
            "file_access: C:\\Users\\Documents\\report.docx",
            "registry_access: HKLM\\SOFTWARE\\Microsoft\\Office",
            "network_connection: 192.168.1.100:445",
            "file_write: C:\\Users\\Documents\\temp.tmp",
            "process_ended: winword.exe",
            "service_start: Windows Update",
            "file_read: C:\\Windows\\system32\\kernel32.dll",
            "memory_allocation: 2048KB",
            "dll_load: oleaut32.dll"
        ]

        ransomware_events = [
            "process_started: cryptolocker.exe",
            "file_scan: C:\\Users\\Documents\\",
            "file_access: C:\\Users\\Documents\\important.docx",
            "encryption_start: C:\\Users\\Documents\\important.docx",
            "file_rename: C:\\Users\\Documents\\important.docx -> important.docx.enc",
            "deletion: Volume Shadow Copy",
            "registry_modify: HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\\cryptolocker",
            "network_request: http://malicious-c2.com/pay",
            "wallpaper_change: C:\\Users\\Desktop\\ransom_note.txt",
            "process_ended: cryptolocker.exe"
        ]

        sessions = []

        # Create normal sessions (70% of data)
        for _ in range(700):
            num_events = np.random.randint(3, 8)
            events = np.random.choice(normal_events, num_events, replace=False).tolist()
            content = self.config['session_window_spliter'].join(events)
            sessions.append({'Content': content, 'Label': 0})

        # Create anomalous sessions (30% of data)
        for _ in range(300):
            # Mix normal and ransomware events
            num_normal = np.random.randint(2, 5)
            num_ransom = np.random.randint(3, 7)

            normal_sel = np.random.choice(normal_events, num_normal, replace=False).tolist()
            ransom_sel = np.random.choice(ransomware_events, num_ransom, replace=False).tolist()

            # Interleave events
            all_events = normal_sel + ransom_sel
            np.random.shuffle(all_events)

            content = self.config['session_window_spliter'].join(all_events)
            sessions.append({'Content': content, 'Label': 1})

        return sessions

    def _determine_anomaly(self, content):
        """Determine if a session contains anomalous behavior"""
        anomaly_keywords = [
            'encrypt', 'ransom', '.enc', '.locked', 'cryption',
            'bitcoin', 'pay', 'decrypt', 'wallpaper', 'delete shadow',
            'modify registry', 'startup', 'persistence'
        ]

        content_lower = content.lower()
        return 1 if any(keyword in content_lower for keyword in anomaly_keywords) else 0

    def _split_processed_data(self, df):
        """Split already processed data into train/test"""
        from sklearn.model_selection import train_test_split

        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df['Label']
        )

        train_df.to_csv(os.path.join(self.processed_path, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(self.processed_path, 'test.csv'), index=False)

        print(f"Dataset split: {len(train_df)} training, {len(test_df)} test samples")
        return


    def preprocess_and_sanitize(self):
        """
        Applies more intelligent regex patterns to sanitize log content while preserving
        important structural information for anomaly detection.
        """
        # More specific regex patterns that preserve security-relevant information
        regex_patterns = [
            # Replace specific file paths with generic placeholders but preserve structure
            (r"C:\\Users\\[^\\]+\\", "C:\\Users\\<user>\\"),
            (r"C:\\ProgramData\\[^\\]+\\", "C:\\ProgramData\\<app>\\"),
            (r"C:\\Program Files\\[^\\]+\\", "C:\\Program Files\\<app>\\"),
            (r"C:\\Windows\\[^\\]+\\", "C:\\Windows\\<path>\\"),

            # Replace specific IPs but preserve network structure
            (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "<ip_address>"),

            # Replace PIDs but keep track that it's a process
            (r"PID:(\d+)", "PID:<pid>"),

            # Replace memory sizes but preserve units
            (r"\b(\d+)KB\b", "<size>KB"),
            (r"\b(\d+)MB\b", "<size>MB"),
            (r"\b(\d+)GB\b", "<size>GB"),

            # Replace specific timestamps but preserve structure
            (r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "<timestamp>"),

            # Replace registry keys but preserve hive structure
            (r"HKLM\\SOFTWARE\\[^\\]+\\", "HKLM\\SOFTWARE\\<vendor>\\"),
            (r"HKCU\\SOFTWARE\\[^\\]+\\", "HKCU\\SOFTWARE\\<vendor>\\"),

            # Replace port numbers but keep port indication
            (r":(\d{4,5})", ":<port>"),

            # Generic number replacement for other cases
            (r'(?<=[^A-Za-z0-9])(\-?\+?\d{3,})(?=[^A-Za-z0-9])', "<number>"),
        ]

        def apply_regex(line):
            for pattern, replacement in regex_patterns:
                line = re.sub(pattern, replacement, str(line))
            return line

        print("Intelligent sanitization of train and test sets...")
        print("Preserving structural information while generalizing specific values...")

        # Analyze before sanitization
        for split in ['train', 'test']:
            file_path = os.path.join(self.processed_path, f'{split}.csv')
            df = pd.read_csv(file_path)

            print(f"\n{split.upper()} set stats before sanitization:")
            print(f"  Total samples: {len(df)}")
            print(f"  Class distribution: {df['Label'].value_counts().to_dict()}")

            # Calculate average sequence length
            avg_len = df['Content'].apply(lambda x: len(str(x).split())).mean()
            print(f"  Average sequence length: {avg_len:.1f} tokens")

            # Apply sanitization
            df['Content'] = df['Content'].apply(apply_regex)
            df.to_csv(file_path, index=False)

            # Calculate after sanitization
            avg_len_after = df['Content'].apply(lambda x: len(str(x).split())).mean()
            print(f"  Average sequence length after sanitization: {avg_len_after:.1f} tokens")

        print("\nSanitization complete. Structural information preserved.")

    def tokenize_dataset(self):
        """
        Tokenizes the preprocessed data with dynamic max_length adjustment
        and improved handling of sequence lengths.
        """
        train_path = os.path.join(self.processed_path, 'train.csv')
        try:
            df = pd.read_csv(train_path)
        except FileNotFoundError:
            print(f"Error: {train_path} not found. Did create_sessions run correctly?")
            return

        # Analyze sequence lengths to determine optimal max_length
        print("Analyzing sequence lengths...")
        df_normal = df[df['Label'] == 0].copy()

        if df_normal.empty:
            print("Warning: No normal data (Label == 0) found for training. Tokenization skipped.")
            return

        # Calculate token lengths
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        tokenizer.pad_token = tokenizer.eos_token

        # Sample token lengths to determine optimal max_length
        sample_size = min(1000, len(df_normal))
        sample_texts = df_normal['Content'].sample(sample_size).tolist()

        token_lengths = []
        for text in sample_texts:
            tokens = tokenizer(str(text), truncation=False, add_special_tokens=False)['input_ids']
            token_lengths.append(len(tokens))

        # Set max_length based on 95th percentile with some buffer
        percentile_95 = np.percentile(token_lengths, 95)
        max_length = min(1024, max(128, int(percentile_95 * 1.1)))

        print(f"Token length statistics:")
        print(f"  Mean: {np.mean(token_lengths):.1f}")
        print(f"  95th percentile: {percentile_95:.1f}")
        print(f"  Using max_length: {max_length}")

        # Check for class balance
        print(f"\nClass distribution in training data:")
        print(f"  Normal (Label=0): {len(df_normal)} samples")
        print(f"  Anomaly (Label=1): {len(df[df['Label'] == 1])} samples")
        print(f"  Training only on normal data: {len(df_normal)} samples")

        # Create dataset
        dataset = Dataset.from_pandas(df_normal[['Content']].rename(columns={'Content': 'text'}))

        # Enhanced preprocessing with data augmentation for normal data
        def preprocess_function(examples):
            # Add EOS token
            texts = [str(text) + tokenizer.eos_token for text in examples["text"]]

            # Tokenize
            tokenized = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors=None
            )

            return tokenized

        # Process in smaller batches to handle memory better
        print(f"\nTokenizing {len(dataset)} samples...")
        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=100,
            remove_columns=['text'],
            desc="Tokenizing"
        )

        # Create labels (same as input_ids for language modeling)
        tokenized_datasets = tokenized_datasets.map(
            lambda x: {"labels": x["input_ids"]},
            batched=True,
            desc="Adding labels"
        )

        # Save additional metadata
        metadata = {
            'max_length': max_length,
            'vocab_size': tokenizer.vocab_size,
            'num_training_samples': len(df_normal),
            'avg_sequence_length': float(np.mean(token_lengths)),
            'percentile_95_length': float(percentile_95)
        }

        final_dataset = DatasetDict({"train": tokenized_datasets})
        final_dataset.save_to_disk(self.tokenized_path)

        # Save metadata separately
        import json
        metadata_path = os.path.join(self.tokenized_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nTokenized dataset saved to {self.tokenized_path}")
        print(f"Metadata saved to {metadata_path}")
        print(f"Ready for federated learning with {len(df_normal)} training samples")
