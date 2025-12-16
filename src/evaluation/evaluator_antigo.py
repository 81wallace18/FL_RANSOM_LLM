import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings

warnings.filterwarnings("ignore")

class Evaluator:
    """
    Handles the evaluation of the trained global models from each round
    using the EXACT SAME LOGIC as the original `federated_evaluation.py` script.
    This is intended for comparison purposes.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = os.path.join(config['results_path'], config['simulation_name'])
        self.test_content, self.test_labels = self._load_test_data()

    def _load_test_data(self):
        """
        Loads the test dataset and balances it to 1000 normal and 1000 anomalous
        samples, exactly like the original script.
        """
        test_path = os.path.join(
            self.config['data_base_path'],
            self.config['dataset_name'],
            'processed',
            'test.csv'
        )
        df = pd.read_csv(test_path)
        
        # Shuffle and balance the dataset
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        balanced_df = pd.concat([
            df[df['Label'] == 1].head(1000), 
            df[df['Label'] == 0].head(1000)
        ])
        
        print(f"Loaded and balanced test data: {len(balanced_df[balanced_df['Label'] == 1])} anomaly, {len(balanced_df[balanced_df['Label'] == 0])} normal.")
        
        content = [str(i) for i in balanced_df["Content"].tolist()]
        labels = balanced_df["Label"].tolist()
        return content, labels

    def _load_calibration_benign(self):
        """
        Loads benign-only calibration data from processed/train.csv (recommended),
        used when threshold_selection == 'fpr_target'.
        """
        source = self.config.get('calibration_source', 'train_benign')
        split = "train" if source == "train_benign" else "test"
        path = os.path.join(
            self.config['data_base_path'],
            self.config['dataset_name'],
            'processed',
            f'{split}.csv'
        )
        df = pd.read_csv(path)
        if 'Label' in df.columns:
            df = df[df['Label'] == 0].copy()
        n = int(self.config.get('calibration_num_samples', 2000))
        if n > 0 and len(df) > n:
            df = df.sample(n=n, random_state=0).reset_index(drop=True)
        return [str(i) for i in df["Content"].tolist()]

    def _is_in_top_k(self, top_k_preds, target_token):
        """Helper function from the original script."""
        return target_token in top_k_preds

    def _calculate_top_k_accuracy(self, model, tokenizer, texts=None):
        """
        Calculates top-k accuracy using the original script's logic.
        It does NOT shift the tokens for next-token prediction.
        """
        accuracies = {f'top{k}': [] for k in self.config['top_k_values']}
        
        model.to(self.device)
        model.eval()

        if texts is None:
            texts = self.test_content
        total_texts = len(texts)
        with torch.no_grad():
            for i, text in enumerate(texts):
                if (i + 1) % 100 == 0:
                    print(f"\r  Calculating accuracy (Antigo Method)... {i + 1}/{total_texts}", end="")

                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                
                outputs = model(**inputs)
                logits = outputs.logits

                for k in self.config['top_k_values']:
                    top_k_preds_indices = torch.topk(logits, k, dim=-1).indices
                    top_k_preds_indices = top_k_preds_indices.cpu().numpy()
                    
                    input_tokens = inputs['input_ids'][0].cpu().numpy()
                    
                    # This is the original, non-shifted logic
                    correct_predictions = sum(self._is_in_top_k(top_k_preds_indices[0, i], token) for i, token in enumerate(input_tokens))
                    total_tokens = len(input_tokens)
                    
                    accuracies[f'top{k}'].append(correct_predictions / total_tokens if total_tokens > 0 else 0)
        
        return pd.DataFrame(accuracies)

    def evaluate(self):
        """
        Main evaluation loop, adapted from the original script.
        """
        print("--- Starting Evaluation (Antigo Method) ---")
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        tokenizer.pad_token = tokenizer.eos_token

        all_f1_results = []
        threshold_mode = self.config.get('threshold_selection', 'f1_max')

        for round_num in range(1, self.config['num_rounds'] + 1):
            print(f"\n--- Evaluating Round {round_num} (Antigo Method) ---")
            model_path = os.path.join(self.results_dir, f'round_{round_num}', 'global_model')
            
            if not os.path.exists(model_path):
                print(f"Model for round {round_num} not found. Skipping.")
                continue

            base_model = AutoModelForCausalLM.from_pretrained(self.config['model_name'])
            if self.config['lora']:
                model = PeftModel.from_pretrained(base_model, model_path)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path)

            acc_df = self._calculate_top_k_accuracy(model, tokenizer)
            acc_df['label'] = self.test_labels

            calib_acc_df = None
            if threshold_mode == 'fpr_target':
                calib_texts = self._load_calibration_benign()
                if calib_texts:
                    calib_acc_df = self._calculate_top_k_accuracy(model, tokenizer, calib_texts)
            
            for k in self.config['top_k_values']:
                scores = acc_df[f'top{k}'].values
                labels = np.array(acc_df['label'].values)

                if threshold_mode == 'fpr_target' and calib_acc_df is not None:
                    target = float(self.config.get('fpr_target', 0.01))
                    target = min(max(target, 0.0), 1.0)
                    th = float(np.quantile(calib_acc_df[f'top{k}'].values, target))
                    preds = (scores < th).astype(int)
                    f1 = float(f1_score(labels, preds))
                else:
                    thresholds = np.linspace(0, 1, int(self.config['f1_threshold_steps']))
                    best_f1 = -1.0
                    th = 0.0
                    for cand in thresholds:
                        preds = (scores < cand).astype(int)
                        f1_cand = f1_score(labels, preds)
                        if f1_cand > best_f1:
                            best_f1 = f1_cand
                            th = float(cand)
                    preds = (scores < th).astype(int)
                    f1 = float(best_f1)

                precision = float(precision_score(labels, preds))
                recall = float(recall_score(labels, preds))
                benign_fpr = float((preds[labels == 0] == 1).mean()) if (labels == 0).any() else float('nan')

                print(f"  K={k}: F1={f1:.4f} at Threshold={th:.4f} ({threshold_mode})")
                
                all_f1_results.append({
                    'round': round_num,
                    'k': k,
                    'threshold_mode': threshold_mode,
                    'fpr_target': float(self.config.get('fpr_target', np.nan)) if threshold_mode == 'fpr_target' else np.nan,
                    'f1_score': f1,
                    'threshold': th,
                    'precision': precision,
                    'recall': recall,
                    'benign_fpr': benign_fpr,
                })

        f1_df = pd.DataFrame(all_f1_results)
        f1_results_path = os.path.join(self.results_dir, 'f1_scores_antigo.csv')
        f1_df.to_csv(f1_results_path, index=False)
        print(f"\nEvaluation (Antigo Method) complete. Results saved to {f1_results_path}")
