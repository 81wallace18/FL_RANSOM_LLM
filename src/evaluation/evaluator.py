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
    Handles the evaluation of the trained global models from each round.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = os.path.join(config['results_path'], config['simulation_name'])
        self.test_df = self._load_test_data()

    def _load_test_data(self):
        """Loads and prepares the test dataset."""
        test_path = os.path.join(
            self.config['data_base_path'],
            self.config['dataset_name'],
            'processed',
            'test.csv'
        )
        df = pd.read_csv(test_path)
        # Optional: Balance the test set for more stable metrics
        # df = pd.concat([
        #     df[df['Label'] == 1].head(1000), 
        #     df[df['Label'] == 0].head(1000)
        # ]).sample(frac=1).reset_index(drop=True)
        return df

    def _calculate_top_k_accuracy(self, model, tokenizer):
        """
        Calculates the top-k next-token prediction accuracy for each log sequence.
        """
        print("Calculating top-k accuracy...")
        accuracies = {f'top{k}': [] for k in self.config['top_k_values']}
        print("Model loaded. Starting evaluation...")
        model.to(self.device)
        model.eval()

        total_texts = len(self.test_df)
        with torch.no_grad():
            for i, text in enumerate(self.test_df["Content"]):
                if (i + 1) % 100 == 0:
                    print(f"\r  Calculating accuracy... {i + 1}/{total_texts}", end="")
            
                inputs = tokenizer(str(text), return_tensors="pt", padding=True, truncation=True, max_length=1024)
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                
                outputs = model(**inputs)
                logits = outputs.logits

                # Shift logits and labels for next-token prediction
                shifted_logits = logits[..., :-1, :].contiguous()
                labels = inputs['input_ids'][..., 1:].contiguous()

                for k in self.config['top_k_values']:
                    top_k_preds = torch.topk(shifted_logits, k, dim=-1).indices
                    
                    # Check if the true next token is in the top-k predictions
                    correct = torch.sum(top_k_preds == labels.unsqueeze(-1)).item()
                    total = labels.numel()
                    
                    accuracies[f'top{k}'].append(correct / total if total > 0 else 0)
        
        return pd.DataFrame(accuracies)

    def evaluate(self):
        """
        Main evaluation loop. Iterates through saved models, calculates metrics,
        and saves results.
        """
        print("--- Starting Evaluation ---")
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        tokenizer.pad_token = tokenizer.eos_token

        all_f1_results = []
        print(f"Evaluating {self.config['num_rounds']} rounds...")

        for round_num in range(1, self.config['num_rounds'] + 1):
            print(f"\n--- Evaluating Round {round_num} ---")
            model_path = os.path.join(self.results_dir, f'round_{round_num}', 'global_model')
            
            if not os.path.exists(model_path):
                print(f"Model for round {round_num} not found. Skipping.")
                continue

            print("Loading model...")
            # Load model for the current round
            base_model = AutoModelForCausalLM.from_pretrained(self.config['model_name'])
            if self.config['lora']:
                model = PeftModel.from_pretrained(base_model, model_path)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path)

            print(f"Calculating accuracies... {model}")
            # Calculate accuracies
            acc_df = self._calculate_top_k_accuracy(model, tokenizer)
            print("Accuracies calculated:")
            acc_df['label'] = self.test_df['Label']
            print(acc_df.describe())
            # Find best F1 score for each k
            for k in self.config['top_k_values']:
                print(f"Finding best F1 for top-{k} accuracy...")
                best_f1 = 0
                best_th = 0
                thresholds = np.linspace(0, 1, self.config['f1_threshold_steps'])
                
                for th in thresholds:
                    print(f"  Testing threshold {th:.4f}...", end='\r')
                    # Anomaly is when accuracy is LOW
                    preds = (acc_df[f'top{k}'] < th).astype(int)
                    f1 = f1_score(acc_df['label'], preds)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_th = th
                
                final_preds = (acc_df[f'top{k}'] < best_th).astype(int)
                precision = precision_score(acc_df['label'], final_preds)
                recall = recall_score(acc_df['label'], final_preds)

                print(f"  K={k}: Best F1={best_f1:.4f} at Threshold={best_th:.4f} | Precision={precision:.4f}, Recall={recall:.4f}")
                
                all_f1_results.append({
                    'round': round_num,
                    'k': k,
                    'f1_score': best_f1,
                    'threshold': best_th,
                    'precision': precision,
                    'recall': recall
                })

        # Save final results
        f1_df = pd.DataFrame(all_f1_results)
        f1_results_path = os.path.join(self.results_dir, 'f1_scores.csv')
        f1_df.to_csv(f1_results_path, index=False)
        print(f"\nEvaluation complete. F1 score results saved to {f1_results_path}")
