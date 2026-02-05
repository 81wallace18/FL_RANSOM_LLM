# Diferenciais e Comparações com o Estado da Arte (guia para escrita)

Este documento serve como guia para **posicionamento** e **comparações** do artigo, usando o que já está implementado no projeto (sem prometer funcionalidades ausentes). Ele complementa `docs/direcionamento_artigo_sbrc.md`.

## 1) Quais são os diferenciais “defensáveis” do nosso trabalho (no estado atual do repo)

### D1) Detecção *benign-only* (LM treinado só em benigno) para ransomware em IoT/Edge

- **O que é**: Treinar o modelo apenas com amostras benignas (`Label==0`) e detectar ransomware por degradação da qualidade preditiva (top-k accuracy + limiar).
- **Por que é diferencial**: Muitos trabalhos em Edge-IIoTSet são **supervisionados** (multi-classe/ataque específico). A formulação benign-only aproxima o cenário de *zero-day/novelty* e reduz dependência de rotulagem.
- **Onde está**: `src/data_processing/edge_ransomware_processor.py` (filtra `Label==0` na tokenização) + `src/evaluation/evaluator_antigo.py` / `src/evaluation/evaluator.py` (top-k + threshold).

### D2) Aprendizado Federado + LoRA (eficiência e privacidade)

- **O que é**: Treino federado agregando apenas adaptadores LoRA (FedAvg/weighted FedAvg), mantendo dados locais.
- **Por que é diferencial**: Conecta privacidade + eficiência de comunicação (mensagens pequenas) a uma tarefa de segurança em IoT/Edge.
- **Onde está**: `src/federated_learning/server.py` (split/seleção/agregação) + `src/federated_learning/client.py` + `src/models/model_loader.py`.

### D3) Non-IID/heterogeneidade (no mínimo: skew de quantidade) + seleção/agregação ponderadas

- **O que é**: Experimentos com `iid`, `quantity_skew_dirichlet` e `hetero_device` e efeitos de `client_selection_strategy` e `use_weighted_aggregation`.
- **Por que é diferencial**: Mostra robustez sob distribuição mais realista que IID puro.
- **Limite atual**: As estratégias atuais simulam principalmente **skew de quantidade** (não feature/label-skew explícito por tipo de dispositivo).
- **Onde está**: `src/federated_learning/server.py` (split + seleção + weighted FedAvg).

### D4) Avaliação operacional temporal (TTD/coverage/FPR) além de F1

- **O que é**: Medir *Time-to-Detection* e cobertura por dispositivo (via `Src IP`) e FPR em benigno.
- **Por que é diferencial**: A literatura frequentemente reporta só F1/accuracy; métricas temporais conectam diretamente com ransomware (tempo = dano) e implantação.
- **Limite atual**: Métrica temporal atual opera por ordenação de `Timestamp` (não implementa janelamento deslizante/multiescala).
- **Onde está**: `src/evaluation/evaluator.py` (`_compute_temporal_metrics`) + `src/data_processing/edge_ransomware_processor.py` (mantém colunas `Timestamp`/`Src IP` no test).

### D5) Controle explícito de *data leakage* (e relato no artigo)

- **O que é**: Evitar inserir campos que “carregam o rótulo” (ex.: `Attack Name`) no `Content`.
- **Por que é diferencial**: É um aspecto de **validade experimental**; melhora credibilidade e reprodutibilidade (especialmente em datasets com rótulos textuais).
- **Onde está**: `src/data_processing/edge_ransomware_processor.py` (não usa `Attack Name` no `Content`).

## 2) Como comparar com o estado da arte (sem cair em comparação injusta)

### Regra de ouro: alinhar **tarefa** e **métrica**

Muitos papers de IDS em Edge-IIoTSet:
- fazem **classificação supervisionada** (multi-classe ou binária) e reportam *accuracy/F1*;
- usam features tabulares diretamente (ML clássico ou DL);
- não consideram FL e não medem custo de comunicação.

Nosso método:
- é **benign-only** (detecção por novelty) + threshold sobre score;
- é **federado** (privacidade) + LoRA (eficiência).

Portanto, “melhor” no seu artigo deve ser escrito como:
- **melhor trade-off operacional** (TTD/coverage/FPR sob FPR-alvo),
- **melhor custo-benefício** (F1 vs MB/rodada; F1 vs latência/memória),
- **mais realista** (Non-IID; threshold operacional; leakage control),
e **não** necessariamente “maior F1 absoluto” contra tarefas diferentes.

## 3) Artigos de referência (o que eles reforçam e como usar na escrita)

### 3.1) “Transfer Learning in Pre-Trained LLMs…” (syscalls)

O que eles enfatizam (e você pode citar para justificar seus diferenciais):
- Restrições operacionais e **tempo real/latência**.
- Trade-off de **context size** vs desempenho.
- Papel de **threshold** no módulo de decisão.
- Necessidade de **agregação temporal**/por janela para melhorar detecção.

Como comparar de forma justa:
- Não comparar F1 diretamente (dados/tarefa diferentes).
- Comparar como “categoria”: ambos mostram trade-offs operacionais e discutem decisão; você complementa com FL+LoRA + métricas temporais em IoT/Edge.

### 3.2) “Anomaly Based IDS using LLMs…” (BERT + BBPE + Edge-IIoT)

O que eles fazem:
- Transformação tabular→texto e foco em **tokenizer** (BBPE) e **classificação supervisionada multi-classe**.

Como comparar de forma justa:
- Se quiser baseline: implementar um baseline supervisionado binário (benign vs ransomware) com tabular/Transformer/BERT-classifier (escopo opcional).
- Caso contrário, compare em termos de **formulação** (benign-only vs supervisionado), **privacidade/FL** (ausente neles) e **métricas operacionais** (TTD/FPR) (geralmente ausentes).

### 3.3) “Exploring LLMs for malware detection…” (survey)

Como usar:
- Para motivar ameaças/risco, guiar escolhas de métricas (FPR, custo, deploy), e reforçar o argumento de “rigor e implantação”.

## 4) “Receitas” de comparação que funcionam bem em SBRC (e cabem no seu código)

### 4.1) Comparação por *FPR-alvo* (recomendado)

Em segurança, comparar em F1 puro pode ser enganoso. Uma comparação defensável é:
- Fixar um **FPR em benigno** (ex.: 1% ou 0,1%) e reportar:
  - `Recall` (ou TPR),
  - `F1`,
  - `TTD` e `coverage`.

Como escrever isso:
- “Para o mesmo custo de falso alarme (FPR), nossa abordagem reduz o tempo até detecção e aumenta a cobertura.”

Observação:
- Hoje o código escolhe limiar por busca de F1. Se você não mudar isso, ainda pode reportar FPR resultante; se implementar calibração por FPR, vira diferencial forte.

### 4.2) Trade-off “custo × segurança” (no estilo do artigo base)

Tabelas/figuras recomendadas:
- **Tabela**: MB/rodada (LoRA rank), #parâmetros treináveis, F1.
- **Tabela**: latência por amostra e memória (VRAM/RAM) na inferência.
- **Figura**: F1 vs comunicação total (ou MB/rodada) para ranks diferentes.

O que isso responde ao revisor:
- “Dá para implantar?” “Quanto custa?” “Qual trade-off?”

### 4.3) Non-IID vs IID (robustez)

Relatar no mínimo:
- `iid` vs `quantity_skew_dirichlet` vs `hetero_device`;
- com/sem `use_weighted_aggregation`;
- opcionalmente com `client_selection_strategy` (uniform vs data_size_proportional).

Como escrever:
- “Sob heterogeneidade de dados (Non-IID), weighted aggregation preserva desempenho/estabilidade do modelo global.”

## 5) Checklist de escrita (para não prometer o que não existe)

Evitar afirmar (até implementar):
- “detecção precoce por janelas deslizantes/multiescala” (hoje não há windowing no pipeline).
- “FedProx/FedNova/robust aggregation” (não implementado).
- “multi-classe” (não implementado; o cenário atual é binário benign/ransomware).

Afirmações seguras (com o estado atual):
- benign-only + LoRA + FL;
- Non-IID por skew de quantidade;
- métricas temporais (TTD/coverage/FPR) por timestamp ordenado;
- trade-off comunicação/inferência (se você medir e reportar, como o artigo base).

## 6) Sugestão de seção “Comparação com a literatura” (texto-modelo de estrutura)

- “Trabalhos supervisionados em Edge-IIoTSet reportam alta acurácia, porém dependem de rótulos e tipicamente operam em modo centralizado.”
- “Trabalhos com LLMs em segurança destacam custo, latência e trade-off de contexto; entretanto raramente analisam privacidade (FL) e custo de comunicação.”
- “Nosso trabalho complementa esses eixos ao avaliar benign-only + FL+LoRA sob Non-IID e ao reportar métricas operacionais (TTD/coverage/FPR) além de F1.”

## 7) Contribuições (para colar no artigo)

As principais contribuições deste trabalho são:

- **C1 — Detecção federada benign-only para ransomware em IoT/Edge**: propomos e avaliamos um pipeline que treina um SLM/LLM apenas com tráfego benigno (fluxos tabulares convertidos em texto) e detecta ransomware por degradação de top‑k accuracy, preservando privacidade via FL e reduzindo custo via LoRA/PEFT.
- **C2 — Robustez sob heterogeneidade (Non‑IID) com seleção/agregação ponderadas**: investigamos cenários IID e não‑IID (skew de quantidade) e analisamos o impacto de estratégias de seleção de clientes e agregação ponderada por volume de dados no desempenho do modelo global.
- **C3 — Avaliação operacional além de F1**: além de F1/precisão/recall, reportamos métricas temporais e de implantação (TTD, coverage e FPR em benigno) para refletir restrições práticas de detecção de ransomware em ambientes IoT/Edge.
- **C4 — Rigor experimental (leakage control)**: documentamos e mitigamos vazamento de rótulo no dataset (ex.: `Attack Name`), garantindo validade científica das métricas reportadas.

## 8) Com quais artigos comparar (e como comparar corretamente)

Esta seção amarra cada contribuição a comparações **justas** com: (i) o **artigo base** (SBRC) e (ii) os artigos em `docs/artigos_referencia/`.

### 8.1) Referências-alvo no repositório

- **Artigo base (inspirador / baseline conceitual)**: `docs/Artigo base.md`
  - Eixos fortes: LoRA/PEFT em FL; trade-off comunicação × desempenho; inferência (latência/memória).
- **IDS com Edge-IIoTSet (supervisionado + tokenizer)**: `docs/artigos_referencia/Anomaly Based Intrusion Detection using Large.md`
  - Eixos fortes: transformação tabular→texto; BBPE; multi-classe; compara datasets; reporta F1/accuracy.
- **Transfer learning + decisão/limiar + contexto + operação**: `docs/artigos_referencia/Transfer Learning in Pre-Trained Large Language.md`
  - Eixos fortes: trade-off context size × performance; restrições operacionais/tempo real; módulo de decisão com threshold; sugere agregação temporal.
- **Survey (motivação e métricas/risco)**: `docs/artigos_referencia/Exploring llms for malware detection: Review, framework design, and countermeasure approaches..md`
  - Eixos fortes: panorama; métricas; aspectos de risco e implantação; serve para motivação/ameaças à validade.

### 8.2) Como comparar sem “injustiça” (regra prática)

Antes de escrever “melhor que X”, alinhe:
- **Tarefa**: benign-only/novelty detection vs classificação supervisionada (binária/multi-classe).
- **Entrada**: texto “feature=value” vs tabular direto vs syscalls vs logs.
- **Cenário**: centralizado vs federado; IID vs Non-IID.
- **Métrica**: F1/accuracy vs FPR-alvo/TTD/coverage vs custo (MB/rodada, latência, memória).

Quando a tarefa não for equivalente, prefira frases do tipo:
- “nosso trabalho complementa X ao abordar privacidade/eficiência em FL”,
- “relatamos métricas operacionais (TTD/coverage/FPR) ausentes em X”,
- “para o mesmo custo de falso alarme (FPR), reduzimos TTD/aumentamos coverage”.

### 8.3) Comparações recomendadas por contribuição

#### C1 — Detecção federada benign-only para ransomware (IoT/Edge)

**Comparar com:** `docs/Artigo base.md`
- **O que é comparável diretamente:** a formulação benign-only + top-k/threshold e o uso de LoRA/PEFT em FL.
- **Como escrever a diferença:** o artigo base valida em logs HDFS; nosso trabalho valida em **fluxos de rede IoT/Edge** (tabular→texto) para ransomware.
- **Figuras/tabelas no “mesmo estilo” do base:** F1 vs rodadas; F1 vs K; e custo (comunicação).

**Comparar com:** `docs/artigos_referencia/Anomaly Based Intrusion Detection using Large.md`
- **Atenção:** eles tratam como **classificação supervisionada** (multi-classe) e enfatizam tokenizer (BBPE). Não é comparação direta de F1.
- **Como comparar corretamente (e ainda ficar forte):**
  - comparar **formulação** (benign-only vs supervisionado),
  - comparar **cenário** (federado/privacidade vs centralizado),
  - comparar **deployability** (comunicação/inferência), não “F1 absoluto”.
- **Se quiser comparação numérica justa (opcional):**
  - criar um baseline supervisionado **binário** (benign vs ransomware) com features tabulares e reportar F1/FPR/TTD lado a lado.

#### C2 — Robustez Non‑IID + seleção/agregação ponderadas

**Comparar com:** `docs/Artigo base.md`
- **Gancho claro:** o artigo base usa IID e cita Non‑IID como limitação/futuro; você pode posicionar como extensão prática.
- **Como comparar:** relatar diferenças IID vs Non‑IID (skew) e mostrar o efeito de `use_weighted_aggregation` e `client_selection_strategy`.
- **Como escrever “melhor”:**
  - “sob Non‑IID (skew de quantidade), a agregação ponderada preserva desempenho/estabilidade do modelo global”.

**Comparar com artigos de IDS/transfer learning (sem FL)**
- **Como comparar:** como “lacuna” (não tratam heterogeneidade de clientes/privacidade) e justificar por que FL é necessário em IoT/Edge.
- **Não prometer:** FedProx/FedNova/robust aggregation (não implementado atualmente).

#### C3 — Avaliação operacional temporal (TTD/coverage/FPR) + decisão

**Comparar com:** `docs/artigos_referencia/Transfer Learning in Pre-Trained Large Language.md`
- **Gancho direto no texto deles:** restrições operacionais/tempo real; trade-off de contexto; thresholds; sugestão de agregação temporal.
- **Como comparar corretamente:**
  - não comparar F1 diretamente (dados e tarefa diferentes),
  - usar como motivação: “literatura discute tempo real e decisão; nós medimos e reportamos TTD/coverage/FPR em IoT/Edge”.
- **Como escrever “melhor” de forma defensável (se você medir):**
  - fixar um **FPR em benigno** e comparar **TTD** e **coverage** (isso é argumento de operação, não de “benchmark de dataset”).

**Comparar com:** `docs/Artigo base.md`
- **Gancho:** o base mede F1 e custo e também mede inferência; você adiciona métricas temporais operacionais.
- **Como escrever a diferença:** “além de F1, reportamos TTD/coverage/FPR para refletir restrições de ransomware (tempo = dano)”.

#### C4 — Rigor experimental (leakage control)

**Comparar com:** qualquer trabalho em Edge-IIoTSet (incluindo `docs/artigos_referencia/Anomaly Based Intrusion Detection using Large.md`)
- **Como usar:** como seção curta de validade:
  - “campos textuais como `Attack Name` podem conter o rótulo e inflar métricas; removemos/evitamos esse vazamento”.
- **Como comparar sem soar acusatório:** “observamos que incluir `Attack Name` gera resultado artificial; por isso adotamos entrada sem campos correlacionados ao rótulo”.

### 8.4) Template de tabela “Comparação com a literatura” (para colar no artigo)

Use uma tabela simples (1 página SBRC costuma gostar de síntese) com colunas como:
- **Trabalho** | **Tarefa** (supervisionado vs benign-only) | **Cenário** (centralizado vs FL) | **Métricas** (F1/accuracy vs TTD/FPR) | **Custo** (comunicação/inferência) | **Nosso diferencial**

E preencha assim:
- `docs/Artigo base.md`: benign-only + FL+LoRA; custo (comunicação/inferência); (sem temporal/Non‑IID).
- `docs/artigos_referencia/Anomaly Based Intrusion Detection using Large.md`: supervisionado multi-classe; tokenizer; (sem FL/custo de comunicação; sem TTD).
- `docs/artigos_referencia/Transfer Learning in Pre-Trained Large Language.md`: supervisionado; trade-off contexto/latência; threshold/decisão; (sem FL).
- `docs/artigos_referencia/Exploring llms for malware detection: Review, framework design, and countermeasure approaches..md`: survey; métricas/risco; (sem experimento comparável).
