import os
import pandas as pd
import numpy as np
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings

warnings.filterwarnings("ignore")

from src.utils.hf import hf_from_pretrained_kwargs

class Evaluator:
    """
    Handles the evaluation of the trained global models from each round.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = os.path.join(config['results_path'], config['simulation_name'])
        self.output_dir = self.config.get("evaluation_output_dir", self.results_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.test_df = self._load_test_data()

    def _output_csv_path(self, basename: str) -> str:
        """
        Returns an output path for a CSV basename, suffixing by threshold mode
        when not using the oracle `f1_max` selection.
        """
        threshold_mode = str(self.config.get("threshold_selection", "f1_max")).lower()
        suffix = "" if threshold_mode == "f1_max" else f"_{threshold_mode}"
        return os.path.join(self.output_dir, f"{basename}{suffix}.csv")

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

    def _parse_timestamps(self, series: pd.Series) -> pd.Series:
        """
        Parses Edge-IIoTSet timestamps robustly.

        The dataset commonly uses the format: '%d/%m/%Y %I:%M:%S %p'. We try that
        first for consistency, then fall back to a more permissive parser.
        """
        ts = pd.to_datetime(series, format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
        missing = ts.isna()
        if missing.any():
            ts.loc[missing] = pd.to_datetime(series[missing], errors="coerce", dayfirst=True)
        return ts

    def _load_calibration_data(self):
        """
        Loads a benign-only calibration set for operational threshold selection.

        calibration_source:
          - "train_benign": uses processed/train.csv filtered to Label==0
          - "calibration_benign": uses processed/calibration.csv filtered to Label==0 (recommended)
          - "test_benign":  uses processed/test.csv filtered to Label==0 (not recommended for strict evaluation)
        """
        source = self.config.get('calibration_source', 'train_benign')
        if source == "train_benign":
            split = "train"
        elif source == "calibration_benign":
            split = "calibration"
        else:
            split = "test"
        path = os.path.join(
            self.config['data_base_path'],
            self.config['dataset_name'],
            'processed',
            f'{split}.csv'
        )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Calibration source '{source}' requires {path}, but it does not exist. "
                "Re-run preprocessing with `benign_calibration_fraction > 0`."
            )
        df = pd.read_csv(path)
        if 'Label' in df.columns:
            df = df[df['Label'] == 0].copy()

        n = int(self.config.get('calibration_num_samples', 2000))
        if n > 0 and len(df) > n:
            df = df.sample(n=n, random_state=0).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        return df

    def _is_in_top_k(self, top_k_preds, target_token):
        """Helper function: checks if target_token is in the top-k predictions."""
        return target_token in top_k_preds

    def _calculate_top_k_accuracy_for_texts(self, model, tokenizer, texts, *, progress_label=""):
        """Calculates the top-k prediction accuracy for each log sequence.

        Supports two methods, configured via `accuracy_method` in the YAML:
          - "shifted": usa próximo-token (logits shiftados vs labels shiftados)
          - "original": replica a lógica do `evaluator_antigo` (sem shift)
        """
        print(f"Calculating top-k accuracy{(' for ' + progress_label) if progress_label else ''}...")
        accuracies = {f'top{k}': [] for k in self.config['top_k_values']}
        model.to(self.device)
        model.eval()

        method = self.config.get('accuracy_method', 'shifted')

        total_texts = len(texts)
        with torch.no_grad():
            for i, text in enumerate(texts):
                if (i + 1) % 100 == 0:
                    label = f" ({progress_label})" if progress_label else ""
                    print(f"\r  Calculating accuracy{label}... {i + 1}/{total_texts}", end="")

                inputs = tokenizer(
                    str(text),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(self.config.get('eval_max_length', self.config.get('max_length', 1024))),
                )
                inputs = {key: val.to(self.device) for key, val in inputs.items()}

                outputs = model(**inputs)
                logits = outputs.logits

                if method == "original":
                    # Mesma lógica do evaluator_antigo (sem shift)
                    for k in self.config['top_k_values']:
                        top_k_preds_indices = torch.topk(logits, k, dim=-1).indices
                        top_k_preds_indices = top_k_preds_indices.cpu().numpy()

                        input_tokens = inputs['input_ids'][0].cpu().numpy()

                        correct_predictions = sum(
                            self._is_in_top_k(top_k_preds_indices[0, idx], token)
                            for idx, token in enumerate(input_tokens)
                        )
                        total_tokens = len(input_tokens)

                        accuracies[f'top{k}'].append(
                            correct_predictions / total_tokens if total_tokens > 0 else 0
                        )
                else:
                    # Método "shifted": próximo-token (mais canônico para LM)
                    shifted_logits = logits[..., :-1, :].contiguous()
                    labels = inputs['input_ids'][..., 1:].contiguous()

                    for k in self.config['top_k_values']:
                        top_k_preds = torch.topk(shifted_logits, k, dim=-1).indices

                        # Check if the true next token is in the top-k predictions
                        correct = torch.sum(top_k_preds == labels.unsqueeze(-1)).item()
                        total = labels.numel()

                        accuracies[f'top{k}'].append(
                            correct / total if total > 0 else 0
                        )

        return pd.DataFrame(accuracies)

    def _calculate_top_k_accuracy(self, model, tokenizer):
        """Convenience wrapper to evaluate on the configured test set."""
        return self._calculate_top_k_accuracy_for_texts(
            model,
            tokenizer,
            self.test_df["Content"].astype(str).tolist(),
            progress_label="test",
        )

    def _select_threshold_f1_max(self, scores, labels):
        thresholds = np.linspace(0, 1, int(self.config['f1_threshold_steps']))
        best_f1 = -1.0
        best_th = 0.0
        for th in thresholds:
            preds = (scores < th).astype(int)
            f1 = f1_score(labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_th = float(th)
        return best_th, float(best_f1)

    def _select_threshold_fpr_target(self, benign_scores):
        target = float(self.config.get('fpr_target', 0.01))
        target = min(max(target, 0.0), 1.0)
        if len(benign_scores) == 0:
            return 0.0
        # Want P(score < th) ~= target for benign-only calibration set.
        return float(np.quantile(benign_scores, target))

    def _benchmark_inference(self, model, tokenizer, texts, *, round_num):
        if not texts:
            return None

        warmup = int(self.config.get('benchmark_warmup', 10))
        max_len = int(self.config.get('eval_max_length', self.config.get('max_length', 1024)))
        timings_ms = []

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model.to(self.device)
        model.eval()

        with torch.no_grad():
            # Warmup
            for text in texts[:warmup]:
                inputs = tokenizer(
                    str(text),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                _ = model(**inputs)

            # Timed
            for text in texts[warmup:]:
                inputs = tokenizer(
                    str(text),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                start = time.perf_counter()
                _ = model(**inputs)
                end = time.perf_counter()
                timings_ms.append((end - start) * 1000.0)

        if timings_ms:
            mean_ms = float(np.mean(timings_ms))
            std_ms = float(np.std(timings_ms))
        else:
            mean_ms = float('nan')
            std_ms = float('nan')

        if torch.cuda.is_available():
            peak_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        else:
            peak_mb = float('nan')

        return {
            'round': int(round_num),
            'model_name': self.config.get('model_name', ''),
            'eval_max_length': max_len,
            'num_samples': int(len(texts)),
            'warmup': warmup,
            'timed_samples': int(len(timings_ms)),
            'mean_ms_per_sample': mean_ms,
            'std_ms_per_sample': std_ms,
            'peak_cuda_memory_mb': peak_mb,
            # FedProx parameters for comparison
            'fedprox_mu': float(self.config.get('fedprox_mu', 0.0)),
            'aggregation_method': 'FedProx' if self.config.get('fedprox_mu', 0.0) > 0 else 'FedAvg',
        }

    def _compute_temporal_metrics_from_df(self, df, round_num, k, *, granularity):
        """
        Computes early-stage detection metrics (TTD, coverage, FPR) when
        temporal and device information is available in the test set.

        - TTD (Time-to-Detection): tempo médio/mediano entre o início do ataque
          em um dispositivo e a primeira detecção.
        - Detection coverage: fração de dispositivos atacados para os quais
          houve detecção.
        - Benign FPR: taxa de falsos positivos em tráfego benigno.
        """
        required_cols = ['Timestamp_parsed', 'Src IP', 'Label', 'pred']
        if not all(col in df.columns for col in required_cols):
            return None

        df = df.dropna(subset=['Timestamp_parsed']).copy()

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
            'granularity': granularity,
            'mean_ttd_seconds': mean_ttd,
            'median_ttd_seconds': median_ttd,
            'detection_coverage': detection_coverage_ratio,
            'benign_fpr': benign_fpr,
            'num_attacked_devices': attacked_devices,
            # FedProx parameters for comparison
            'fedprox_mu': float(self.config.get('fedprox_mu', 0.0)),
            'aggregation_method': 'FedProx' if self.config.get('fedprox_mu', 0.0) > 0 else 'FedAvg',
        }

    def _compute_temporal_metrics(self, preds, round_num, k):
        required_cols = ['Timestamp', 'Src IP', 'Label']
        if not all(col in self.test_df.columns for col in required_cols):
            print("Temporal metrics skipped: required columns not found in test.csv.")
            return None

        df = self.test_df[['Timestamp', 'Src IP', 'Label']].copy()
        df['pred'] = preds.values if hasattr(preds, "values") else preds
        df['Timestamp_parsed'] = self._parse_timestamps(df['Timestamp'])
        return self._compute_temporal_metrics_from_df(df, round_num, k, granularity='flow')

    def _window_aggregate(self, series):
        agg = str(self.config.get('temporal_window_agg', 'mean')).lower()
        if series.empty:
            return float('nan')
        if agg == 'max':
            return float(series.max())
        if agg == 'min':
            return float(series.min())
        if agg == 'p90':
            return float(np.quantile(series.values, 0.90))
        if agg == 'p10':
            return float(np.quantile(series.values, 0.10))
        if agg == 'p25':
            return float(np.quantile(series.values, 0.25))
        return float(series.mean())

    def _build_windowed_df(self, base_df, score_series):
        """
        Builds a window-aggregated dataframe for temporal evaluation.

        base_df must contain: Timestamp, Src IP, Label (same row order as score_series).
        Returns a dataframe with: Timestamp_parsed, Src IP, Label, score.
        """
        window_seconds = int(self.config.get('temporal_window_seconds', 30))
        if window_seconds <= 0:
            return None

        df = base_df[['Timestamp', 'Src IP', 'Label']].copy()
        df['score'] = score_series.values
        df['Timestamp_parsed'] = self._parse_timestamps(df['Timestamp'])
        df = df.dropna(subset=['Timestamp_parsed'])

        # Group by device and time bin
        grouper = pd.Grouper(key='Timestamp_parsed', freq=f'{window_seconds}S')
        grouped = df.groupby(['Src IP', grouper], dropna=False)

        out = grouped.agg(
            Label=('Label', 'max'),
            score=('score', lambda s: self._window_aggregate(s)),
        ).reset_index()
        # Timestamp_parsed here is the window start as produced by the Grouper
        return out

    def evaluate(self):
        """
        Main evaluation loop. Iterates through saved models, calculates metrics,
        and saves results.
        """
        print("--- Starting Evaluation ---")
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'], **hf_from_pretrained_kwargs(self.config))
        tokenizer.pad_token = tokenizer.eos_token

        all_f1_results = []
        all_temporal_results = []
        all_bench_results = []
        print(f"Evaluating {self.config['num_rounds']} rounds...")

        rounds = list(range(1, int(self.config['num_rounds']) + 1))
        if self.config.get('include_round_0', False):
            rounds = [0] + rounds

        benchmark_rounds = set()
        if self.config.get('benchmark_inference', False):
            for r in self.config.get('benchmark_rounds', []):
                try:
                    benchmark_rounds.add(int(r))
                except Exception:
                    continue
            if benchmark_rounds:
                rounds = sorted(set(rounds).union(benchmark_rounds))

        for round_num in rounds:
            print(f"\n--- Evaluating Round {round_num} ---")
            model_path = os.path.join(self.results_dir, f'round_{round_num}', 'global_model')
            
            if not os.path.exists(model_path):
                print(f"Model for round {round_num} not found. Skipping.")
                continue

            print("Loading model...")
            # Load model for the current round
            base_model = AutoModelForCausalLM.from_pretrained(self.config['model_name'], **hf_from_pretrained_kwargs(self.config))
            if self.config['lora']:
                model = PeftModel.from_pretrained(base_model, model_path)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path, **hf_from_pretrained_kwargs(self.config))

            # Optional inference benchmark (deployability)
            if self.config.get('benchmark_inference', False) and int(round_num) in benchmark_rounds:
                n = int(self.config.get('benchmark_num_samples', 200))
                texts = self.test_df["Content"].astype(str).tolist()
                if n > 0 and len(texts) > n:
                    texts = texts[:n]
                bench = self._benchmark_inference(model, tokenizer, texts, round_num=round_num)
                if bench is not None:
                    all_bench_results.append(bench)

            print(f"Calculating accuracies... {model}")
            # Calculate accuracies
            acc_df = self._calculate_top_k_accuracy(model, tokenizer)
            print("Accuracies calculated:")
            acc_df['label'] = self.test_df['Label']
            print(acc_df.describe())

            threshold_mode = self.config.get('threshold_selection', 'f1_max')
            calib_acc_df = None
            calib_window_df_by_k = {}
            if threshold_mode == 'fpr_target':
                calib_df = self._load_calibration_data()
                calib_acc_df = self._calculate_top_k_accuracy_for_texts(
                    model,
                    tokenizer,
                    calib_df["Content"].astype(str).tolist(),
                    progress_label="calibration",
                )
                if str(self.config.get('temporal_eval_mode', 'flow')).lower() == 'window':
                    for k in self.config['top_k_values']:
                        calib_window_df_by_k[k] = self._build_windowed_df(calib_df, calib_acc_df[f'top{k}'])

            # Compute metrics for each k
            for k in self.config['top_k_values']:
                scores = acc_df[f'top{k}']

                temporal_mode = str(self.config.get('temporal_eval_mode', 'flow')).lower()
                if temporal_mode == 'window':
                    # Windowed evaluation (aggregate per Src IP and time bin)
                    window_df = self._build_windowed_df(self.test_df, scores)
                    if window_df is None or window_df.empty:
                        print("  Windowed evaluation skipped: unable to build windows (missing Timestamp/Src IP?).")
                        continue

                    if threshold_mode == 'fpr_target':
                        calib_window_df = calib_window_df_by_k.get(k)
                        th = self._select_threshold_fpr_target(
                            calib_window_df['score'].dropna().values if calib_window_df is not None else np.array([])
                        )
                    else:
                        th, _ = self._select_threshold_f1_max(window_df['score'].values, window_df['Label'].values)

                    window_df['pred'] = (window_df['score'] < th).astype(int)
                    f1 = float(f1_score(window_df['Label'], window_df['pred']))
                    precision = float(precision_score(window_df['Label'], window_df['pred']))
                    recall = float(recall_score(window_df['Label'], window_df['pred']))
                    benign_fpr = float((window_df.loc[window_df['Label'] == 0, 'pred'] == 1).mean()) if (window_df['Label'] == 0).any() else float('nan')

                    print(
                        f"  K={k} (window): F1={f1:.4f} | Threshold={th:.4f} ({threshold_mode})"
                        f" | Precision={precision:.4f}, Recall={recall:.4f}, BenignFPR={benign_fpr:.4f}"
                    )

                    all_f1_results.append({
                        'round': round_num,
                        'k': k,
                        'granularity': 'window',
                        'window_seconds': int(self.config.get('temporal_window_seconds', 30)),
                        'window_agg': str(self.config.get('temporal_window_agg', 'mean')),
                        'threshold_mode': threshold_mode,
                        'fpr_target': float(self.config.get('fpr_target', np.nan)) if threshold_mode == 'fpr_target' else np.nan,
                        'threshold': th,
                        'f1_score': f1,
                        'precision': precision,
                        'recall': recall,
                        'benign_fpr': benign_fpr,
                        'num_windows': int(len(window_df)),
                        # FedProx parameters for comparison
                        'fedprox_mu': float(self.config.get('fedprox_mu', 0.0)),
                        'aggregation_method': 'FedProx' if self.config.get('fedprox_mu', 0.0) > 0 else 'FedAvg',
                    })

                    if self.config.get('enable_temporal_metrics', True):
                        # Reuse temporal metric logic with window start timestamps
                        tmp = window_df.rename(columns={'Timestamp_parsed': 'Timestamp_parsed'}).copy()
                        temporal_metrics = self._compute_temporal_metrics_from_df(tmp, round_num, k, granularity='window')
                        if temporal_metrics is not None:
                            temporal_metrics['window_seconds'] = int(self.config.get('temporal_window_seconds', 30))
                            temporal_metrics['window_agg'] = str(self.config.get('temporal_window_agg', 'mean'))
                            all_temporal_results.append(temporal_metrics)
                else:
                    # Flow-level evaluation (current behavior)
                    if threshold_mode == 'fpr_target':
                        th = self._select_threshold_fpr_target(calib_acc_df[f'top{k}'].values)
                        preds = (scores < th).astype(int)
                        f1 = float(f1_score(acc_df['label'], preds))
                    else:
                        th, f1 = self._select_threshold_f1_max(scores.values, acc_df['label'].values)
                        preds = (scores < th).astype(int)

                    precision = float(precision_score(acc_df['label'], preds))
                    recall = float(recall_score(acc_df['label'], preds))
                    benign_fpr = float((preds[acc_df['label'] == 0] == 1).mean()) if (acc_df['label'] == 0).any() else float('nan')

                    print(
                        f"  K={k}: F1={f1:.4f} | Threshold={th:.4f} ({threshold_mode})"
                        f" | Precision={precision:.4f}, Recall={recall:.4f}, BenignFPR={benign_fpr:.4f}"
                    )

                    all_f1_results.append({
                        'round': round_num,
                        'k': k,
                        'granularity': 'flow',
                        'threshold_mode': threshold_mode,
                        'fpr_target': float(self.config.get('fpr_target', np.nan)) if threshold_mode == 'fpr_target' else np.nan,
                        'threshold': th,
                        'f1_score': f1,
                        'precision': precision,
                        'recall': recall,
                        'benign_fpr': benign_fpr,
                        # FedProx parameters for comparison
                        'fedprox_mu': float(self.config.get('fedprox_mu', 0.0)),
                        'aggregation_method': 'FedProx' if self.config.get('fedprox_mu', 0.0) > 0 else 'FedAvg',
                    })

                    if self.config.get('enable_temporal_metrics', True):
                        temporal_metrics = self._compute_temporal_metrics(preds, round_num, k)
                        if temporal_metrics is not None:
                            all_temporal_results.append(temporal_metrics)

        # Save final results
        f1_df = pd.DataFrame(all_f1_results)
        f1_results_path = self._output_csv_path('f1_scores')
        f1_df.to_csv(f1_results_path, index=False)
        print(f"\nEvaluation complete. F1 score results saved to {f1_results_path}")

        # Salva métricas temporais, se tiverem sido calculadas
        if all_temporal_results:
            temporal_df = pd.DataFrame(all_temporal_results)
            temporal_results_path = self._output_csv_path('temporal_metrics')
            temporal_df.to_csv(temporal_results_path, index=False)
            print(f"Temporal evaluation complete. Results saved to {temporal_results_path}")

        if all_bench_results:
            bench_df = pd.DataFrame(all_bench_results)
            bench_path = self._output_csv_path('inference_benchmark')
            bench_df.to_csv(bench_path, index=False)
            print(f"Inference benchmark complete. Results saved to {bench_path}")
