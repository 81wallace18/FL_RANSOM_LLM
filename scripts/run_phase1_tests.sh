#!/usr/bin/env bash
set -euo pipefail

# Quick validation: OrigLike FedAvg (LoRA r=32 vs r=16) + plots

python3 main.py --config configs/test/config_test_origlike_fedavg.yaml
python3 main.py --config configs/test/config_test_origlike_fedavg_lora16.yaml
python3 scripts/plot_results.py \
  --fedavg results/TEST_OrigLike_FedAvg_R10 --fedavg_label LoRA32 \
  --fedprox results/TEST_OrigLike_FedAvg_R10_LoRA16 --fedprox_label LoRA16 \
  --k 10 \
  --output_dir img/test_lora32_vs_lora16/
