# Métricas presentes nos artigos de referência (para comparação)

Este documento lista, de forma objetiva, **quais métricas** aparecem nos artigos em `docs/artigos_referencia/` e **como** elas podem ser usadas na escrita do artigo (comparação numérica direta vs comparação de posicionamento).

> Importante: comparação numérica **direta** só é justa quando a **tarefa** e o **setup** são equivalentes (mesmo dataset/mesma definição de rótulo/mesma granularidade). Quando não forem, use a comparação para justificar *lacunas* (privacidade/eficiência/operacionalização) e não para “ganhar em F1”.

## 1) `Anomaly Based Intrusion Detection using Large.md` (BERT + BBPE, IDS)

**Tarefa/Setup**
- IDS em IoT; abordagem **supervisionada** (classificação), com BERT e um tokenizer BBPE treinado do zero.
- Avalia em múltiplos datasets (UNSW-NB15, ToN-IoT, Edge-IIoTset) e reporta desempenho multi-classe.

**Métricas explícitas no texto**
- `Validation Loss` (perda em validação; curva por épocas).
- `Weighted F1 Score` (F1 ponderado por suporte das classes; apropriado para desbalanceamento).
- `Accuracy` (acurácia global).
- `Class-specific Accuracy` (acurácia por classe; tabela/relato por tipo de ataque).

**Como usar para comparar no seu artigo**
- **Comparação numérica direta (só se alinhar tarefa)**:
  - Se você tiver um baseline supervisionado **binário** (benign vs ransomware) ou multi-classe no Edge-IIoTset, pode comparar `accuracy/F1` diretamente.
- **Comparação de posicionamento (recomendado com o estado atual)**:
  - Eles são centralizados/supervisionados; você compara em termos de:
    - benign-only/novelty detection vs supervisionado;
    - FL + privacidade + custo de comunicação (ausente lá);
    - métricas operacionais (FPR/TTD/coverage) (geralmente ausentes).

## 2) `Transfer Learning in Pre-Trained Large Language.md` (syscalls, malware)

**Tarefa/Setup**
- Classificação de malware baseada em sequências de system calls; **supervisionada** (multi-classe).
- Ênfase em **context size** e restrições operacionais/tempo real; discute thresholds e agregação temporal.

**Métricas explícitas no texto**
- Métricas clássicas de classificação: `Accuracy`, `Precision`, `Recall`, `F1-Score`.
- Métricas adicionais: `Kappa`, `MCC`.
- Relato de `TPR`/desempenho por classe e referência a `confusion matrix`.
- Discussão explícita de **trade-off**: `context size` × desempenho × custo/latência.
- Discussão explícita de **thresholding** (sensibilidade vs falsos positivos) e necessidade de **agregação temporal** para janelas.

**Como usar para comparar no seu artigo**
- **Comparação numérica direta**: geralmente **não** (dataset é syscalls; tarefa/dados diferentes).
- **Comparação de posicionamento (forte)**:
  - Citar como motivação para:
    - medir e reportar **latência/memória de inferência**;
    - analisar trade-off de **comprimento de sequência** (`max_length`) vs desempenho;
    - justificar thresholding operacional e agregação temporal (windowed evaluation).
- **Como “comparar melhor” sem injustiça**:
  - Use frases do tipo: “além de métricas de classificação, reportamos FPR/TTD/coverage e custo de comunicação em cenário federado”.

## 3) `Exploring llms for malware detection: Review, framework design, and countermeasure approaches..md` (survey)

**Tipo de trabalho**
- Survey + framework/riscos; não é um experimento direto com dataset específico comparável ao seu pipeline.

**Métricas no texto**
- Discute **métricas de avaliação** e princípios de avaliação/mitigação de risco, mas não é uma fonte “benchmark” para comparar F1 do seu método.

**Como usar para comparar no seu artigo**
- Para motivar **escolha de métricas** (ex.: FPR, custo, robustez) e estruturar discussão de riscos/ameaças.
- Para posicionar o artigo dentro do panorama de LLMs em malware/cibersegurança.

## 4) O que você consegue comparar “de verdade” hoje

### Comparação direta mais natural
- **Com o artigo base** (`docs/Artigo base.md`): mesma família de abordagem (LM benign-only + FL + LoRA) → comparação direta de:
  - `F1/precision/recall`;
  - custo de comunicação (MB/rodada/total);
  - latência/memória de inferência (estilo Tabela 3 do base).

### Comparação como diferencial (quando refs não reportam)
- `benign_fpr`, `TTD`, `coverage` e avaliação por **janelas temporais**:
  - Use como diferencial/operacionalização (não como “ganhei deles”).

## 5) Checklist de comparação para o texto

- Se o outro paper é **supervisionado** e o seu é **benign-only**:
  - compare formulação e custos; evite “melhor F1” sem alinhar tarefa.
- Se o outro paper não é **federado**:
  - compare privacidade/custo de comunicação (seu diferencial).
- Se o outro paper não reporta métricas operacionais:
  - compare por “lacuna”: você adiciona FPR/TTD/coverage.

