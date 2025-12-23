# Tabela — Artigos, métricas e valores (extraído dos `docs/`)

Esta tabela compila os **valores numéricos explicitamente citados** nos arquivos do repositório. Use como base para a seção de comparação do artigo.

> Observação: alguns trabalhos são *supervisionados* (classificação) e outros não; portanto, nem toda comparação é “direta” (mesma tarefa/métrica). Aqui eu só listo o que está escrito nos textos.

## 1) Visão geral (headline metrics)

| Artigo (arquivo) | Setup (alto nível) | Métricas com valores reportados |
|---|---|---|
| Artigo base (`docs/Artigo base.md`) | FL + LoRA/PEFT + benign-only em logs HDFS | `F1 > 98%` (texto); “até `4000x`” redução em tamanho de mensagem (texto); comunicação por cliente/rodada (Tabela 2); inferência: `30.8908±0.3973 ms` e `1021.37 MB` (135M), `62.3438±0.8539 ms` e `1981.91 MB` (360M), `253.0354±7.1095 ms` e `7712.16 MB` (1.7B) (Tabela 3) |
| IDS + BERT+BBPE (`docs/artigos_referencia/Anomaly Based Intrusion Detection using Large.md`) | Classificação supervisionada multi-classe (UNSW-NB15, ToN-IoT, Edge-IIoTset) | ToN-IoT: `F1 0.9899→0.9938` (4 épocas) e `val loss 0.0191`; UNSW-NB15: `F1 0.8554` (época 5) e `val loss 0.3809`; Edge-IIoT: `F1 0.99999` e `val loss 0.0000441` (época 3); menção: Ransomware `97%` accuracy e MITM `99%` accuracy; também cita SecurityBERT `98.2%` accuracy (trabalho relacionado) |
| TL em syscalls (`docs/artigos_referencia/Transfer Learning in Pre-Trained Large Language.md`) | Classificação supervisionada em syscalls + análise de `context size` | Tabela de métricas por modelo: Accuracy/Precision/Recall/F1/Kappa/MCC (ver Seção 2 abaixo). Melhores: BigBird (Acc `0.8667`, F1 `0.8688`) e Longformer (Acc `0.8616`, F1 `0.8621`). Também cita “~`0.94` average F1” em janelas de 10s (trabalho relacionado). |
| Survey (`docs/artigos_referencia/Exploring llms for malware detection: Review, framework design, and countermeasure approaches..md`) | Survey/framework | Não reporta um benchmark único com números comparáveis; contém tabelas qualitativas e discussões de métricas/riscos. |

## 2) Transfer Learning em syscalls — tabela completa de métricas

Extraído de `docs/artigos_referencia/Transfer Learning in Pre-Trained Large Language.md` (tabela “Model / Context Size / Accuracy / Precision / Recall / F1-Score / Kappa / MCC”):

| Model | Context Size | Accuracy | Precision | Recall | F1-Score | Kappa | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| BERT | 512 | 0.6772 | 0.82 | 0.6596 | 0.6465 | 0.5504 | 0.6024 |
| DistilBERT | 512 | 0.6289 | 0.71 | 0.6181 | 0.5930 | 0.4786 | 0.5379 |
| GPT-2 | 1024 | 0.6944 | 0.7986 | 0.6865 | 0.6808 | 0.5792 | 0.6123 |
| BigBird | 4096 | 0.8667 | 0.8754 | 0.8668 | 0.8688 | 0.8298 | 0.8311 |
| Longformer | 4096 | 0.8616 | 0.8696 | 0.8614 | 0.8621 | 0.8232 | 0.8250 |
| Mistral | 8192 | 0.5817 | 0.6112 | 0.6462 | 0.6242 | 0.4754 | 0.4798 |

## 3) IDS com BERT+BBPE — acurácia por classe (tabela do artigo)

Extraído de `docs/artigos_referencia/Anomaly Based Intrusion Detection using Large.md` (TABLE III: “Sgl” = tokenizer treinado no dataset; “Cmb” = tokenizer treinado na combinação dos datasets):

| Class | ToN-IoT (Sgl) | ToN-IoT (Cmb) | UNSW-NB15 (Sgl) | UNSW-NB15 (Cmb) | Edge-IIoT (Sgl) | Edge-IIoT (Cmb) |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 0.99 | 1.00 | 0.96 | 0.98 | 0.99 | 1.00 |
| Backdoor | 0.99 | 1.00 | 0.00 | 0.17 | 0.97 | 1.00 |
| DDoS | 0.92 | 0.99 | - | - | 1.00 | 1.00 |
| DoS | 0.95 | 0.99 | 0.02 | 0.28 | - | - |
| MITM | 0.97 | 0.99 | - | - | 0.99 | 1.00 |
| Ransomware | 0.99 | 1.00 | - | - | 0.92 | 1.00 |
| Injection | 0.98 | 0.99 | - | - | 0.98 | 1.00 |
| Scanning | 0.99 | 0.99 | - | - | 0.98 | 1.00 |
| XSS | 0.87 | 1.00 | - | - | 0.99 | 1.00 |
| Exploits | - | - | 0.93 | 0.93 | - | - |
| Reconnaissance | - | - | 0.75 | 0.80 | - | - |
| Fuzzers | - | - | 0.65 | 0.75 | - | - |
| Analysis | - | - | 0.05 | 0.17 | - | - |
| Shellcode | - | - | 0.68 | 0.83 | - | - |
| Password | 0.99 | 1.00 | - | - | 1.00 | 1.00 |
| Port Scanning | - | - | - | - | 0.96 | 1.00 |
| SQL Injection | - | - | - | - | 0.99 | 1.00 |
| Vulnerability Scanner | - | - | - | - | 0.99 | 1.00 |

