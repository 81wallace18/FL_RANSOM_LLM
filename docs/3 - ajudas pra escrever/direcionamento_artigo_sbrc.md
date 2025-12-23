# Direcionamento do Artigo (SBRC A4) — Diferenciais e Escopo

Este arquivo consolida as decisões e o **direcionamento** do artigo para evitar reabrir as mesmas discussões. Ele parte do “artigo base” como inspiração, mas define quais **diferenciais** serão assumidos no artigo atual e onde isso está implementado no projeto.

## 1) Tese (mensagem central)

Propor e avaliar um fluxo de **detecção federada de ransomware em IoT/Edge** usando um **modelo de linguagem pequeno (SLM/LLM)** treinado **apenas em tráfego benigno**, com **fine-tuning eficiente (LoRA/PEFT)**, analisando o impacto de **distribuições Non-IID/heterogeneidade** e reportando métricas **operacionais temporais** além de F1.

## 2) Diferenciais assumidos (3 contribuições na Introdução)

### (C1) Cross-domain + treino só em benigno (LM como detector de anomalia)

**Contribuição:** adaptar o paradigma do artigo base (LM treinado em normal) para o domínio de **fluxos de rede tabulares** (IDS/IoT), convertendo-os em sequências textuais e treinando **somente com Label=0 (benigno)**.

**Onde está no código:**
- `src/data_processing/edge_ransomware_processor.py` (gera `Content` textual a partir de features do fluxo; split train/test; tokeniza apenas `Label==0`)
- `src/models/model_loader.py` (inicializa modelo + aplica LoRA e salva round 0)
- `src/federated_learning/client.py` / `src/federated_learning/server.py` (treino federado e agregação)

**Observação de validade (crítica):**
- Controle explícito de *data leakage*: **não** colocar `Attack Name` no `Content`.
  - O campo pode existir como metadado no CSV para análises, mas não pode “vazar” para a entrada do modelo.

### (C2) Estudo sistemático de Non-IID/heterogeneidade (além de IID)

**Contribuição:** avaliar o método sob **cenários não-IID** (mais próximos de IoT real), e não apenas IID, incluindo estratégias de distribuição e efeitos em desempenho/estabilidade.

**Onde está no código:**
- `src/federated_learning/server.py`
  - `data_distribution_strategy: iid`
  - `data_distribution_strategy: quantity_skew_dirichlet`
  - `data_distribution_strategy: hetero_device` (na prática: viés de quantidade por grupos)
- Seleção de clientes:
  - `client_selection_strategy: uniform` ou `data_size_proportional`
- Agregação:
  - `use_weighted_aggregation: true` (weighted FedAvg por nº de amostras)

**Limite atual (para não “prometer demais” no texto):**
- As estratégias atuais focam principalmente em **skew de quantidade**; não há, por enquanto, *label-skew/feature-skew* explícito por “tipos de dispositivo”.

### (C3) Avaliação operacional temporal (TTD/coverage/FPR), além de F1

**Contribuição:** além de F1 (via top-k accuracy + busca de limiar), reportar métricas operacionais que reflitam implantação:
- **TTD (Time-to-Detection)** por dispositivo (via `Src IP`)
- **Coverage** (fração de dispositivos atacados detectados)
- **Benign FPR** (taxa de falsos positivos em benigno)

**Onde está no código:**
- `src/evaluation/evaluator.py`
  - `_compute_temporal_metrics(...)` calcula `mean_ttd_seconds`, `median_ttd_seconds`, `detection_coverage`, `benign_fpr`.
- Requisito de dados:
  - `processed/test.csv` precisa conter `Timestamp` e `Src IP` (mantidos em `src/data_processing/edge_ransomware_processor.py`).

**Limite atual (para redigir corretamente):**
- A métrica temporal atual mede “tempo até detecção” ao ordenar flows por `Timestamp`; não implementa **janelamento deslizante/multiescala** (se o artigo usar o termo *early-stage detection*, deixar explícita a definição adotada).

## 3) Configuração do experimento (fonte de verdade)

Arquivo: `configs/config_edge_ransomware.yaml`
- Define: LoRA (rank/alpha/dropout), FL (nº rounds, nº clientes, fração, paralelismo), estratégia de distribuição, seleção, agregação e avaliação (F1 + temporal).

## 4) O que falta (somente para “fechar” o paper sem extrapolar)

Para sustentar as 3 contribuições com evidência, o artigo precisa ter **resultados** e comparações mínimas:

- Para (C1):
  - Relatar F1/precision/recall ao longo das rodadas (ou melhor rodada) e descrever como `Content` é construído.
  - Deixar explícito que o treino usa somente `Label==0`.

- Para (C2):
  - Rodar e comparar **IID vs quantity_skew_dirichlet vs hetero_device**, mantendo o resto fixo (mesma arquitetura, rounds, etc.).
  - Comparar **FedAvg vs weighted FedAvg** (`use_weighted_aggregation` on/off) sob pelo menos um cenário não-IID.

- Para (C3):
  - Reportar `temporal_metrics.csv` (TTD/coverage/FPR) e discutir implicações (trade-off: cobertura vs FPR; TTD vs F1).

## 5) Formulação dos “claims” (seguro para revisão)

Claims recomendados (defensáveis com o estado atual do código):
- O pipeline adapta LM+LoRA+FL para **detecção de ransomware** em **IoT/Edge** a partir de **fluxos tabulares convertidos em texto**, treinando somente em benigno.
- O estudo considera **distribuições não-IID** (skew de quantidade) e discute impacto em desempenho e estabilidade, incluindo seleção/agregação ponderadas.
- A avaliação inclui métricas **operacionais temporais** (TTD/coverage/FPR) além de F1.

Claims a evitar (até implementar):
- “Detecção precoce por janelas deslizantes” (não existe janelamento temporal hoje).
- “Agregação robusta além do FedAvg (FedProx/FedNova…)” (não implementado).

