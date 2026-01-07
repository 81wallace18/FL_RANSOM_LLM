#!/usr/bin/env bash
set -euo pipefail

# Phase 1 quick validation: FedAvg vs FedProx + plots

python3 main.py --config configs/test/config_test_fedavg.yaml
python3 main.py --config configs/test/config_test_fedprox.yaml
python3 scripts/plot_results.py \
  --fedavg results/TEST_FedAvg_Phase1 \
  --fedprox results/TEST_FedProx_Phase1 \
  --k 10 \
  --output_dir img/test/
