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

    def _find_optimal_threshold_adaptive(self, accuracies, labels):
        """
        Finds optimal threshold using cross-validation and adaptive search.
        More robust than simple grid search.
        """
        from sklearn.model_selection import StratifiedKFold

        if len(np.unique(labels)) < 2:
            # Edge case: all samples are from one class
            return 0.5, 0.5

        # Use fewer CV folds if we have limited data
        n_folds = min(5, max(2, len(labels) // 10))
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        # First pass: Coarse grid search
        coarse_thresholds = np.linspace(0, 1, 50)
        best_coarse_f1 = 0
        best_coarse_th = 0

        for th in coarse_thresholds:
            fold_f1s = []
            for train_idx, val_idx in skf.split(accuracies, labels):
                train_acc, train_labels = accuracies[train_idx], labels[train_idx]
                val_acc, val_labels = accuracies[val_idx], labels[val_idx]

                # Predict on validation set
                val_preds = (val_acc < th).astype(int)
                if len(np.unique(val_preds)) > 1:  # Only calculate if we have both classes
                    f1 = f1_score(val_labels, val_preds, zero_division=0)
                    fold_f1s.append(f1)

            if fold_f1s:
                avg_f1 = np.mean(fold_f1s)
                if avg_f1 > best_coarse_f1:
                    best_coarse_f1 = avg_f1
                    best_coarse_th = th

        # Second pass: Fine search around best coarse threshold
        fine_range = 0.1  # Search ±10% around best coarse
        fine_start = max(0, best_coarse_th - fine_range)
        fine_end = min(1, best_coarse_th + fine_range)
        fine_thresholds = np.linspace(fine_start, fine_end, 20)

        best_f1 = 0
        best_th = best_coarse_th

        for th in fine_thresholds:
            fold_f1s = []
            for train_idx, val_idx in skf.split(accuracies, labels):
                train_acc, train_labels = accuracies[train_idx], labels[train_idx]
                val_acc, val_labels = accuracies[val_idx], labels[val_idx]

                val_preds = (val_acc < th).astype(int)
                if len(np.unique(val_preds)) > 1:
                    f1 = f1_score(val_labels, val_preds, zero_division=0)
                    fold_f1s.append(f1)

            if fold_f1s:
                avg_f1 = np.mean(fold_f1s)
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    best_th = th

        return best_f1, best_th

    def _load_test_data(self):
        """
        Loads and prepares the test dataset with strategic balancing
        for more stable and representative F1 score calculation.
        """
        test_path = os.path.join(
            self.config['data_base_path'],
            self.config['dataset_name'],
            'processed',
            'test.csv'
        )
        df = pd.read_csv(test_path)

        print("Original test set distribution:")
        print(f"  Total samples: {len(df)}")
        print(f"  Normal (Label=0): {len(df[df['Label'] == 0])}")
        print(f"  Anomaly (Label=1): {len(df[df['Label'] == 1])}")

        # Separate classes
        normal_samples = df[df['Label'] == 0]
        anomaly_samples = df[df['Label'] == 1]

        # Strategy 1: If highly imbalanced, apply balanced sampling
        normal_count = len(normal_samples)
        anomaly_count = len(anomaly_samples)
        imbalance_ratio = max(normal_count, anomaly_count) / min(normal_count, anomaly_count)

        if imbalance_ratio > 3:  # If one class is 3x+ more than the other
            print(f"\nTest set is imbalanced (ratio: {imbalance_ratio:.2f})")
            print("Applying balanced sampling for more stable metrics...")

            # Use the smaller class size as baseline
            min_count = min(normal_count, anomaly_count)

            # Sample equally from both classes
            normal_balanced = normal_samples.sample(n=min_count, random_state=42)
            anomaly_balanced = anomaly_samples.sample(n=min_count, random_state=42)

            # Combine and shuffle
            balanced_df = pd.concat([normal_balanced, anomaly_balanced])
            balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

            print(f"Balanced test set:")
            print(f"  Total samples: {len(balanced_df)}")
            print(f"  Normal: {len(balanced_df[balanced_df['Label'] == 0])}")
            print(f"  Anomaly: {len(balanced_df[balanced_df['Label'] == 1])}")

            # Store original distribution for reference
            self.original_distribution = {
                'total': len(df),
                'normal': normal_count,
                'anomaly': anomaly_count,
                'imbalance_ratio': imbalance_ratio
            }

            return balanced_df

        else:
            # If reasonably balanced, use original distribution
            print("\nTest set is reasonably balanced. Using original distribution.")
            self.original_distribution = {
                'total': len(df),
                'normal': normal_count,
                'anomaly': anomaly_count,
                'imbalance_ratio': 1.0
            }
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
            # Find best F1 score for each k using adaptive threshold
            for k in self.config['top_k_values']:
                print(f"Finding best adaptive F1 for top-{k} accuracy...")
                best_f1, best_th = self._find_optimal_threshold_adaptive(
                    acc_df[f'top{k}'].values,
                    acc_df['label'].values
                )

                final_preds = (acc_df[f'top{k}'] < best_th).astype(int)
                precision = precision_score(acc_df['label'], final_preds, zero_division=0)
                recall = recall_score(acc_df['label'], final_preds, zero_division=0)

                # Calculate additional metrics
                from sklearn.metrics import accuracy_score, roc_auc_score
                accuracy = accuracy_score(acc_df['label'], final_preds)

                # Calculate AUC if we have enough variation in predictions
                try:
                    auc = roc_auc_score(acc_df['label'], 1 - acc_df[f'top{k}'])
                except:
                    auc = 0.5  # Default when AUC cannot be calculated

                print(f"  K={k}: Best Adaptive F1={best_f1:.4f} at Threshold={best_th:.4f}")
                print(f"       | Precision={precision:.4f}, Recall={recall:.4f}")
                print(f"       | Accuracy={accuracy:.4f}, AUC={auc:.4f}")

                all_f1_results.append({
                    'round': round_num,
                    'k': k,
                    'f1_score': best_f1,
                    'threshold': best_th,
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy,
                    'auc': auc
                })

        # Save final results
        f1_df = pd.DataFrame(all_f1_results)
        f1_results_path = os.path.join(self.results_dir, 'f1_scores.csv')
        f1_df.to_csv(f1_results_path, index=False)
        print(f"\nEvaluation complete. F1 score results saved to {f1_results_path}")
