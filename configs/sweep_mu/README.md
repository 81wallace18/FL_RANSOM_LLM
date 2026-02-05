# Sweep de mu (FedProx) — LoRA16 / OrigLike

Objetivo: encontrar o `fedprox_mu` que melhora o **Top-10 F1 no round final** sem aumentar muito o FPR, mantendo o regime OrigLike.

Configs:
- `config_mu_baseline_fedavg.yaml` (mu=0.0)
- `config_mu_0005.yaml` (mu=0.0005)
- `config_mu_0010.yaml` (mu=0.0010)
- `config_mu_0020.yaml` (mu=0.0020)
- `config_mu_0010_warmup.yaml` (mu=0.0010 + warmup)

Notas:
- Todos usam `lora_rank: 16`, `initial_lr: 3e-4`, `client_frac: 0.2`, `num_rounds: 10`, `accuracy_method: original`.
- `use_parallel_training: false` para evitar perda de clientes (comparação mais justa).
- Se o cache estiver binned, rode primeiro uma vez com `force_reprocess_data: true` + `content_mode: raw` (qualquer config) ou apague `data/.../processed/tokenized`.

