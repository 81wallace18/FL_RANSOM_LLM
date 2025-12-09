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

    def _compute_temporal_metrics(self, preds, round_num, k):
        """
        Computes early-stage detection metrics (TTD, coverage, FPR) when
        temporal and device information is available in the test set.

        - TTD (Time-to-Detection): tempo médio/mediano entre o início do ataque
          em um dispositivo e a primeira detecção.
        - Detection coverage: fração de dispositivos atacados para os quais
          houve detecção.
        - Benign FPR: taxa de falsos positivos em tráfego benigno.
        """
        required_cols = ['Timestamp', 'Src IP', 'Label']
        if not all(col in self.test_df.columns for col in required_cols):
            print("Temporal metrics skipped: required columns not found in test.csv.")
            return None

        df = self.test_df.copy()
        df['pred'] = preds.values if hasattr(preds, "values") else preds

        # Converte timestamp para datetime, ignorando formatos inválidos
        df['Timestamp_parsed'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Timestamp_parsed'])

        ttds = []
        detection_coverage = 0
        attacked_devices = 0

        # Métricas por dispositivo (Src IP)
        for device_id, group in df.groupby('Src IP'):
            group = group.sort_values('Timestamp_parsed')
            if (group['Label'] == 1).any():
                attacked_devices += 1
                attack_start_time = group.loc[group['Label'] == 1, 'Timestamp_parsed'].min()
                detections = group[
                    (group['Timestamp_parsed'] >= attack_start_time) & (group['pred'] == 1)
                ]
                if not detections.empty:
                    first_detection_time = detections['Timestamp_parsed'].min()
                    delta = (first_detection_time - attack_start_time).total_seconds()
                    ttds.append(max(delta, 0.0))
                    detection_coverage += 1

        if attacked_devices > 0 and ttds:
            mean_ttd = float(np.mean(ttds))
            median_ttd = float(np.median(ttds))
            detection_coverage_ratio = detection_coverage / attacked_devices
        else:
            mean_ttd = float('nan')
            median_ttd = float('nan')
            detection_coverage_ratio = 0.0

        # FPR em tráfego benigno
        benign = df[df['Label'] == 0]
        if not benign.empty:
            benign_fpr = float((benign['pred'] == 1).mean())
        else:
            benign_fpr = float('nan')

        return {
            'round': round_num,
            'k': k,
            'mean_ttd_seconds': mean_ttd,
            'median_ttd_seconds': median_ttd,
            'detection_coverage': detection_coverage_ratio,
            'benign_fpr': benign_fpr,
            'num_attacked_devices': attacked_devices
        }

    def evaluate(self):
        """
        Main evaluation loop. Iterates through saved models, calculates metrics,
        and saves results.
        """
        print("--- Starting Evaluation ---")
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        tokenizer.pad_token = tokenizer.eos_token

        all_f1_results = []
        all_temporal_results = []
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

                # Métricas temporais (se habilitado e se as colunas existirem)
                if self.config.get('enable_temporal_metrics', True):
                    temporal_metrics = self._compute_temporal_metrics(final_preds, round_num, k)
                    if temporal_metrics is not None:
                        all_temporal_results.append(temporal_metrics)

        # Save final results
        f1_df = pd.DataFrame(all_f1_results)
        f1_results_path = os.path.join(self.results_dir, 'f1_scores.csv')
        f1_df.to_csv(f1_results_path, index=False)
        print(f"\nEvaluation complete. F1 score results saved to {f1_results_path}")

        # Salva métricas temporais, se tiverem sido calculadas
        if all_temporal_results:
            temporal_df = pd.DataFrame(all_temporal_results)
            temporal_results_path = os.path.join(self.results_dir, 'temporal_metrics.csv')
            temporal_df.to_csv(temporal_results_path, index=False)
            print(f"Temporal evaluation complete. Results saved to {temporal_results_path}")
