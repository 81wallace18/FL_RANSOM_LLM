# Justificativa das Contribuições com Base nos Artigos de Referência

Este documento mapeia como cada contribuição do trabalho se justifica em relação às lacunas identificadas nos artigos de referência.

## 1. Resumo das Contribuições e Lacunas

| Contribuição | O que falta nos artigos | Nosso diferencial |
|--------------|-------------------------|-------------------|
| **C1: Benign-only + FL** | Artigos são **supervisionados** e **centralizados** | Detecção por **anomalia** (benign-only) + **federado** (privacidade) |
| **C2: Non-IID** | Artigo base usa só IID (cita Non-IID como trabalho futuro) | **Implementa e avalia** Non-IID (hetero_device, Dirichlet, by_src_ip) |
| **C3: Métricas temporais** | Artigos reportam apenas F1/Accuracy | Reporta **TTD, coverage, FPR** (métricas operacionais) |
| **C4: Leakage control** | Artigos podem incluir `Attack Name` no texto | **Remove** campos que vazam rótulo, garantindo validade experimental |

## 2. Comparações por Artigo de Referência

### 2.1 vs. "Anomaly Based IDS using LLMs" (BERT + BBPE + Edge-IIoT)

**Referência:** `docs/artigos_referencia/Anomaly Based Intrusion Detection using Large.md`

**O que eles fazem:**
- Classificação **supervisionada multi-classe**
- Transformação tabular → texto com foco em **tokenizer (BBPE)**
- Avaliação **centralizada**
- Métricas: F1, Accuracy

**Lacunas que preenchemos:**
| Aspecto | Artigo de referência | Nosso trabalho |
|---------|---------------------|----------------|
| Formulação | Supervisionado (usa rótulos de ataque) | **Benign-only** (detecta por anomalia) |
| Cenário | Centralizado | **Federado** (preserva privacidade) |
| Comunicação | Não considera | **LoRA** (~4000x menos bytes) |
| Métricas | F1/Accuracy | **TTD, coverage, FPR** |

**Como comparar de forma justa:**
- ❌ Não comparar F1 diretamente (tarefa diferente)
- ✅ Comparar formulação, cenário e métricas operacionais

**Frase sugerida:**
> "Diferentemente de abordagens supervisionadas que dependem de rótulos de ataque, nossa formulação benign-only aproxima cenários de zero-day e opera de forma federada, preservando privacidade dos dados locais."

---

### 2.2 vs. "Transfer Learning in Pre-Trained LLMs" (syscalls)

**Referência:** `docs/artigos_referencia/Transfer Learning in Pre-Trained Large Language.md`

**O que eles fazem:**
- Classificação de malware via **syscalls**
- Análise de trade-off **context size vs performance**
- Discussão sobre **threshold** e decisão
- Sugestão de agregação temporal (como trabalho futuro)
- Métricas: F1 ~0.86 (BigBird/Longformer)

**Lacunas que preenchemos:**
| Aspecto | Artigo de referência | Nosso trabalho |
|---------|---------------------|----------------|
| Dados | Syscalls | **Fluxos de rede IoT/Edge** |
| Cenário | Centralizado | **Federado** |
| Métricas temporais | Sugere, não implementa | **Implementa TTD/coverage** |
| Eficiência | Full fine-tuning | **LoRA/PEFT** |

**Como comparar de forma justa:**
- ❌ Não comparar F1 diretamente (dados e tarefa diferentes)
- ✅ Usar como motivação para métricas operacionais

**Frase sugerida:**
> "Trabalhos recentes discutem restrições operacionais e trade-offs de decisão; nosso trabalho complementa ao medir e reportar TTD e coverage em ambiente IoT/Edge federado."

---

### 2.3 vs. Artigo Base (FL + LoRA + HDFS)

**Referência:** `docs/Artigo base.md`

**O que eles fazem:**
- Detecção **benign-only** em logs HDFS
- **FL + LoRA** com agregação FedAvg
- Trade-off comunicação × desempenho
- Métricas de inferência (latência/memória)
- F1 > 98%

**Lacunas que preenchemos:**
| Aspecto | Artigo base | Nosso trabalho |
|---------|-------------|----------------|
| Dataset | Logs HDFS | **Fluxos de rede ransomware (IoT/Edge)** |
| Distribuição | IID apenas | **IID + Non-IID** (hetero_device, Dirichlet) |
| Métricas | F1, custo comunicação | **+ TTD, coverage, FPR** |
| Agregação | FedAvg simples | **+ Weighted FedAvg** |

**Como comparar de forma justa:**
- ✅ Comparação direta possível (mesma formulação benign-only + FL + LoRA)
- ✅ Posicionar como extensão para domínio IoT/Edge + Non-IID + métricas operacionais

**Frase sugerida:**
> "Estendemos a formulação benign-only + FL + LoRA para detecção de ransomware em fluxos de rede IoT/Edge, investigando cenários Non-IID e reportando métricas operacionais temporais."

---

### 2.4 vs. "Exploring LLMs for malware detection" (Survey)

**Referência:** `docs/artigos_referencia/Exploring llms for malware detection: Review, framework design, and countermeasure approaches..md`

**Como usar:**
- Para **motivação** e contextualização de ameaças
- Para justificar escolha de **métricas** (FPR, custo, deployability)
- Para reforçar argumento de **rigor e implantação**

**Frase sugerida:**
> "Surveys recentes destacam a importância de métricas operacionais e custo de implantação; nosso trabalho responde a essa demanda ao reportar TTD, coverage e custo de comunicação."

---

## 3. Tabela Comparativa para o Artigo

| Trabalho | Tarefa | Cenário | Métricas | Custo | Nosso diferencial |
|----------|--------|---------|----------|-------|-------------------|
| Anomaly Based IDS (BERT+BBPE) | Supervisionado multi-classe | Centralizado | F1/Accuracy | Não reporta | FL + benign-only + TTD/FPR |
| Transfer Learning LLMs (syscalls) | Supervisionado | Centralizado | F1 ~0.86 | Latência | FL + LoRA + métricas temporais |
| Artigo Base (FL+LoRA+HDFS) | Benign-only | Federado (IID) | F1, comunicação | Bytes/rodada | Non-IID + TTD/coverage + IoT/Edge |
| Survey (Exploring LLMs) | N/A (revisão) | N/A | Discussão | Discussão | Validação experimental |

---

## 4. Frases-Chave para o Artigo

### Introdução/Motivação
> "Trabalhos supervisionados em Edge-IIoTSet reportam alta acurácia, porém dependem de rótulos e tipicamente operam em modo centralizado, não considerando privacidade nem heterogeneidade de dispositivos."

### Contribuições
> "Nosso trabalho complementa esses eixos ao avaliar detecção benign-only com FL+LoRA sob cenários Non-IID e ao reportar métricas operacionais (TTD, coverage, FPR) além de F1."

### Comparação
> "Para o mesmo custo de falso alarme (FPR), nossa abordagem reduz o tempo até detecção (TTD) e aumenta a cobertura de dispositivos atacados."

### Validade
> "Diferentemente de trabalhos anteriores, removemos campos que vazam informação do rótulo (como `Attack Name`), garantindo validade experimental das métricas reportadas."

---

## 5. Checklist de Comparação Justa

Antes de escrever "melhor que X", verificar:

- [ ] **Tarefa alinhada?** (benign-only vs supervisionado)
- [ ] **Dados comparáveis?** (fluxos de rede vs syscalls vs logs)
- [ ] **Cenário equivalente?** (federado vs centralizado; IID vs Non-IID)
- [ ] **Métrica justa?** (F1 vs FPR-alvo vs TTD)

Se não alinhado, usar frases como:
- "nosso trabalho **complementa** X ao abordar..."
- "**além de** F1, reportamos..."
- "para o **mesmo FPR**, reduzimos TTD..."
