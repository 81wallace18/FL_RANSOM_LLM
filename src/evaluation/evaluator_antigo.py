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

    def _is_in_top_k(self, top_k_preds, target_token):
        """Helper function from the original script."""
        return target_token in top_k_preds

    def _calculate_top_k_accuracy(self, model, tokenizer):
        """
        Calculates top-k accuracy using the original script's logic.
        It does NOT shift the tokens for next-token prediction.
        """
        accuracies = {f'top{k}': [] for k in self.config['top_k_values']}
        
        model.to(self.device)
        model.eval()

        with torch.no_grad():
            for text in self.test_content:
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
            
            for k in self.config['top_k_values']:
                best_f1 = 0
                best_th = 0
                thresholds = np.linspace(0, 1, self.config['f1_threshold_steps'])
                
                for th in thresholds:
                    preds = (acc_df[f'top{k}'] < th).astype(int)
                    f1 = f1_score(acc_df['label'], preds)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_th = th
                
                final_preds = (acc_df[f'top{k}'] < best_th).astype(int)
                precision = precision_score(acc_df['label'], final_preds)
                recall = recall_score(acc_df['label'], final_preds)

                print(f"  K={k}: Best F1={best_f1:.4f} at Threshold={best_th:.4f}")
                
                all_f1_results.append({
                    'round': round_num,
                    'k': k,
                    'f1_score': best_f1,
                    'threshold': best_th,
                    'precision': precision,
                    'recall': recall
                })

        f1_df = pd.DataFrame(all_f1_results)
        f1_results_path = os.path.join(self.results_dir, 'f1_scores_antigo.csv')
        f1_df.to_csv(f1_results_path, index=False)
        print(f"\nEvaluation (Antigo Method) complete. Results saved to {f1_results_path}")
