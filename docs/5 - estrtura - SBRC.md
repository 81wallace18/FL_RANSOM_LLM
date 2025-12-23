## Artigo 1 — “Fine-Tuning Eficiente…” (Artigo base.md)

### 1) Macro-estrutura (o esqueleto SBRC)

Estrutura típica SBRC e bem “limpa”:

* **Resumo** + **Abstract** (1 parágrafo cada)
* **1. Introdução**
* **2. Trabalhos Relacionados**
* **3. Metodologia** (com 3.1 a 3.5)
* **4. Análise de Desempenho** (4.1 Cenário + 4.2 Resultados)
* **5. Conclusão**
* **6. Agradecimentos**
* **Disponibilidade de Artefatos**
* Em Introdução, há o parágrafo “mapa do artigo” (“O restante do artigo está organizado…”) 

---

### 2) Contagem por seção (parágrafos, palavras, linhas ~)

Linhas são **estimadas** (≈ palavras / 10,5), só pra você ter noção de “tamanho relativo” por seção.

| Seção                          | Parágrafos (texto) | Blocos (fig/tabela) | Palavras | Linhas ~ |
| ------------------------------ | -----------------: | ------------------: | -------: | -------: |
| Resumo                         |                  1 |                   0 |       77 |        7 |
| Abstract                       |                  1 |                   0 |       76 |        7 |
| 1. Introdução                  |                  8 |                   0 |      684 |       65 |
| 2. Trabalhos Relacionados      |                  8 |                   0 |      952 |       91 |
| 3.1. Visão Geral               |                  2 |                   1 |      383 |       36 |
| 3.2. Pré-processamento de Logs |                  2 |                   1 |      288 |       27 |
| 3.3. Aprendizado Federado      |                  2 |                   0 |      480 |       46 |
| 3.4. Fine-Tuning Eficiente     |                  3 |                   2 |      829 |       79 |
| 3.5. Tomada de Decisão         |                  2 |                   0 |      383 |       36 |
| 4.1. Cenário de Avaliação      |                  2 |                   0 |      646 |       62 |
| 4.2. Resultados                |                  5 |                   4 |     1189 |      113 |
| 5. Conclusão                   |                  2 |                   0 |      261 |       25 |
| 6. Agradecimentos              |                  1 |                   0 |       77 |        7 |
| Disponibilidade de Artefatos   |                  1 |                   0 |       57 |        5 |

---

### 3) Micro-estrutura por seção (como os parágrafos “funcionam”)

#### Resumo / Abstract (1 parágrafo cada)

Padrão SBRC bem clássico:

* Contexto do problema → método proposto → resultado quantitativo (ganho) → implicação.
* Aqui aparece o “gancho” de eficiência + cenário distribuído.

#### 1. Introdução (8 parágrafos de texto)

Padrão retórico (o que cada parágrafo costuma fazer, e esse artigo segue bem isso):

1. **Contexto amplo**: sistemas distribuídos / borda / complexidade + risco.
2. **Por que logs**: logs como “fonte universal” para detectar falhas/anomalias.
3. **Limites do ML clássico**: custo, rotulagem, generalização, etc.
4. **Entrada de LLM/SLM**: por que modelos de linguagem ajudam (semântica/seq).
5. **Problema central do artigo**: custo + comunicação + privacidade em cenário distribuído.
6. **Aposta técnica**: FL + PEFT (LoRA) para reduzir custo mantendo performance.
7. **Resumo do pipeline**: como treina, como agrega, como decide anomalia.
8. **Mapa do artigo** (“O restante do artigo está organizado…”) — isso é “obrigatório SBRC” porque ajuda o revisor a navegar. 

**Dica de SBRC:** se você estiver escrevendo o seu, tente manter a Introdução com **6–9 parágrafos** e sempre terminar com (i) contribuições (lista curta) e (ii) organização do artigo.

#### 2. Trabalhos Relacionados (8 parágrafos)

Estratégia muito típica:

* Parágrafo 1: panorama + crítica (“há soluções, mas são custosas e não servem para distribuído”).
* Parágrafos seguintes: **1 técnica por parágrafo** (LogBERT, LogFiT, LogLLM, LogGPT, etc.), sempre com 3 movimentos:

  1. o que é,
  2. como detecta,
  3. por que falha no seu cenário (custo, rótulo, flexibilidade, distribuído).
* Fecha apontando a **lacuna** e preparando a Metodologia.
  O “gancho” de “métodos relevantes + limitações” aparece bem explícito logo no início da seção. 

#### 3. Metodologia (3.1–3.5)

Aqui a escrita é “engenharia”: cada subseção responde uma pergunta.

* **3.1 Visão Geral**: apresenta o **fluxo inteiro** e já amarra com uma figura que mostra “passo a passo”.

  * O texto descreve o pipeline em **etapas numeradas (1 a 7)** (pré-processa, recebe modelo, ajusta local, envia adapters, agrega, devolve, decide anomalia). 
  * A Figura 1 é captionada como ilustração do passo a passo. 

* **3.3 Aprendizado Federado**: explica FedAvg e já antecipa o “problema do FL”: custo de comunicação. 

* **3.4 Fine-Tuning Eficiente**: aqui você vê o “núcleo” (LoRA).

  * O texto primeiro justifica o custo (treino + rede), depois apresenta LoRA como PEFT, depois conecta com “só adapters viajam”, e remete à figura. 

* **3.5 Tomada de Decisão**: fecha com o critério de detecção (top-K + limiar β) e como escolher hiperparâmetros. 

#### 4. Análise de Desempenho (4.1–4.2)

* **4.1 Cenário**: tudo que revisor quer ver: modelo escolhido (SmolLMs), LoRA rank, quantização, hardware, dataset, rounds, nº de clientes. 
* **4.2 Resultados**: escrita padrão “resultado científico”:

  1. define o que será avaliado (perdas, F1, custo),
  2. mostra figura de curva/perda,
  3. resume achado (“max F1”, “comunicação”),
  4. tabela final de performance.

---

### 4) Como o artigo introduz e explica Figuras e Tabelas

#### Figuras (padrão de uso)

O artigo segue o padrão SBRC correto:

* **antes da figura**: frase de promessa (“Figura X apresenta…”)
* **depois**: 2–5 linhas dizendo “o que observar” (tendência, comparação, implicação)

Exemplo perfeito: em 3.1, o texto fala “visão geral… apresentada na Figura 1” e imediatamente descreve as etapas numeradas. 

Principais figuras:

* **Figura 1**: pipeline completo (etapas 1–7). 
* **Figura 4**: “Máximo F1 atingido…” (serve como punchline de performance). 
* **Figura 5**: “Total de Comunicação” (punchline de custo de rede). 

#### Tabelas (padrão de uso)

* **Tabela 1 (Related Work)**: tabela comparativa funciona como “resumo crítico” do Related Work — depois de vários parágrafos 1-método-por-vez, você fecha com comparação lado a lado. 
* **Tabela 2 (custo do modelo transmitido)**: é a tabela que “materializa” o ganho de LoRA vs FFT, com ranks. 
* **Tabela 3 (performance final)**: tabela de consolidação (F1 e/ou métricas finais). 

---

### 5) Palavras mais frequentes por parte (Top termos)

A ideia aqui é: **o vocabulário dominante revela o foco** de cada seção.

* **Introdução**: modelos, sistemas, dados, anomalias, detecção, análise, logs, distribuídos
* **Trabalhos Relacionados**: modelos, detecção, anomalias, logs, treinamento, custo, abordagem, dados
* **3.1 Visão Geral**: clientes, etapas, modelo, servidor, parâmetros, treinamento, inferências
* **3.3 FL**: federado, clientes, servidor, comunicação, FedAvg, rodadas
* **3.4 LoRA/PEFT**: LoRA, parâmetros, custo, treinamento, adaptadores, comunicação
* **4.1 Cenário**: modelos, SmolLM, rank, quantização, dataset, clientes, rodadas, GPU
* **4.2 Resultados**: modelos, F1, comunicação, desempenho, rank, custo, resultados

Se você quiser, eu também extraio **bigramas** (ex.: “custo comunicação”, “aprendizado federado”, “dados locais”) porque isso ajuda MUITO a escrever seções coerentes.

---

---

## Artigo 2 — PDF “MENTORED… dataset / fluxo” (29837-…pdf)

### 1) Macro-estrutura

O próprio artigo declara a organização (isso é excelente para SBRC): Seção 2 related, Seção 3 fluxo, Seção 4 experimento/dataset, Seção 5 uso em IDS, Seção 6 conclusão. 

Estrutura:

* **1 Introdução**
* **2 Trabalhos Relacionados**
* **3 Fluxo para criação e análise de datasets…**
* **4 Estudo de caso e resultados experimentais** (4.1–4.3)
* **5 Uso do dataset em detecção de intrusão** (5.1–5.2)
* **6 Conclusão e trabalhos futuros**

---

### 2) Contagem por seção (blocos, palavras, linhas ~)

Aqui “blocos de texto” = parágrafos identificáveis no texto extraído (boa aproximação do tamanho relativo).

| Seção                                         | Blocos de texto | Blocos (fig/tabela) | Palavras | Linhas ~ |
| --------------------------------------------- | --------------: | ------------------: | -------: | -------: |
| Front matter (título/autores/resumo/abstract) |               5 |                   0 |      597 |       57 |
| 1. Introdução                                 |               3 |                   0 |      815 |       78 |
| 2. Trabalhos Relacionados                     |               5 |                   0 |     1042 |       99 |
| 3. Fluxo…                                     |               9 |                   1 |     1532 |      146 |
| 4. Estudo de caso…                            |               3 |                   0 |      538 |       51 |
| 5. Uso em IDS…                                |               3 |                   0 |      973 |       93 |
| 6. Conclusão…                                 |               3 |                   0 |     1051 |      100 |

---

### 3) Micro-estrutura por seção (o “jeito SBRC” que ele usa)

#### 1. Introdução

Ela faz 4 coisas bem típicas:

1. Problema: necessidade de datasets representativos/reprodutíveis em cibersegurança
2. Gap: dificuldade de reproduzir ataques + variações de topologia/heterogeneidade
3. Proposta: fluxo com MENTORED Testbed
4. Questão de pesquisa explícita + overview do artigo (muito forte)  

Além disso, ele define o que é o MENTORED (3 componentes: Portal, Master, Cluster RNP). 

#### 2. Trabalhos Relacionados

Estrutura “survey crítico”:

* explica como DDoS é gerado e por que é difícil reproduzir fielmente
* passa por testbeds relevantes (ex.: Deterlab) e aponta limitações (usabilidade, heterogeneidade IoT etc.) 
* fecha reforçando necessidade de um fluxo mais reprodutível/flexível

#### 3. Fluxo proposto

Essa é a seção “estrela”. Ela tem:

* introdução do fluxo
* lista numerada explicando etapas (definição, alocação, execução, coleta, processamento/análise) 
* uma figura que serve como “mapa mental” do fluxo (Figura 1). 

#### 4. Estudo de caso + resultados

Parte bem SBRC:

* 4.1 contextualiza o ataque (slowloris) e comportamento
* 4.2 define cenários/experimentos (Tabela 1)
* 4.3 descreve o que é coletado e como vira dataset

O artigo lista claramente **o que entra no dataset** (pcap, log Apache, csv por cliente, json de mapeamento, csv OpenArgus rotulado). 

#### 5. Uso do dataset em IDS

É “prova de utilidade”:

* define atributos (Proto, Sport, Dport… etc) e usa t-SNE pra visualizar separação de cenários. 
* traz gráficos de monitoramento e vazão como evidência do efeito do ataque:

  * **Figura 2**: monitoramento de respostas do servidor. 
  * **Figura 3**: vazão (throughput) durante cenários. 

---

### 4) Figuras e tabelas (como são “explicadas”)

#### Figura 1 (fluxo)

Ela é usada como:

* “ancora cognitiva”: o leitor entende o pipeline inteiro em 10s
* o texto então detalha passo a passo (lista numerada) 

#### Tabela 1 (cenários)

Ela define C1/C2, E1/E2, variações de participantes e parâmetros (clientes/atacantes, duração etc.). 

#### Tabela 2 (sumário do tráfego/requests)

Ela aparece como “sanity check” quantitativo do dataset: pacotes e respostas por cenário/experimento.  

---

### 5) Palavras mais frequentes por parte (indicativo)

No PDF, alguns termos acentuados ficam “quebrados” na extração, então trate isso como **indicativo**, mas já dá o retrato:

* **Introdução**: dataset(s), reprodutibilidade, testbed, cibersegurança, ataques, fluxo
* **Trabalhos Relacionados**: DDoS, testbed, experimentação, topologia, dispositivos, IoT
* **Fluxo**: experimento, alocação, execução, coleta, resultados, análise, MENTORED
* **Estudo de caso**: slowloris, cenários, tráfego, servidor, respostas
* **Uso em IDS**: OpenArgus, fluxos, atributos, t-SNE, detecção, classificação

---

# Template SBRC (estrutura + conteúdo obrigatório)

## Título

**Objetivo:** dizer *o que você fez + em que contexto + com que objetivo*.

**Padrões bons:**

* “[Técnica] para [problema] em [contexto]”
* “Uma Abordagem [X] para [Y] com [restrição Z] (privacidade/custo/edge)”

**Evite:** título muito genérico (“Uma análise de…”) sem método.

---

## Resumo (1 parágrafo | ~120–200 palavras)

**O que vai aqui (ordem recomendada):**

1. **Contexto** (1 frase): área + por que importa
2. **Problema** (1 frase): dor clara
3. **Lacuna** (1 frase): por que soluções atuais não bastam
4. **Proposta** (1–2 frases): o que você propõe (método/pipeline)
5. **Avaliação** (1–2 frases): dataset/cenário + métricas
6. **Resultados** (1 frase): números principais (ex.: +F1, -custo, -comunicação)
7. **Contribuição** (1 frase): o que o trabalho entrega

**Palavras comuns/esperadas:** *propõe, apresenta, avaliamos, resultados, desempenho, custo, comunicação, privacidade, experimento, dataset, métrica (F1/AUC/accuracy)*.

---

## Palavras-chave (3 a 5)

**Regra prática:** 2 do **método** + 2 do **domínio** + 1 do **contexto**
Ex.: “aprendizado federado”, “LoRA/PEFT”, “detecção de intrusão”, “IoT/Edge”, “logs/tráfego”.

---

# 1. Introdução (6–9 parágrafos | ~1–1,5 página)

**Objetivo:** convencer *por que* o problema é importante e *o que* você contribui.

### Estrutura recomendada (parágrafo a parágrafo)

**P1 — Contexto macro**

* Área + cenário real (IoT/Edge, sistemas distribuídos, redes, SOC, etc.)
* “Crescimento / complexidade / criticidade”

**P2 — Problema concreto**

* Defina o problema em 1–2 frases bem claras
* Ex.: “Detectar X sob restrições Y”

**P3 — Por que é difícil**

* 2–4 limitações: custo, rótulos, drift, privacidade, comunicação, heterogeneidade, dados não-IID

**P4 — Estado atual (alto nível)**

* 1–2 frases sobre o que existe (sem entrar em detalhes ainda)

**P5 — Lacuna**

* “Apesar de…, ainda falta…” (o revisor procura isso)

**P6 — Sua proposta (a 1ª vez que você “declara”)**

* Nomeie o método/pipeline
* Diga os blocos: entrada → processamento → treino → decisão

**P7 — Contribuições (bullet points)**

* 3 a 5 bullets, começando com verbos:

  * “Propomos…”
  * “Projetamos…”
  * “Avaliamos…”
  * “Disponibilizamos…” (se tiver artefatos)

**P8 — Organização do artigo**

* “O restante do artigo está organizado…” (padrão SBRC)

**Palavras comuns/esperadas:** *motivação, desafio, lacuna, abordagem, proposta, contribuições, eficiência, escalabilidade, privacidade, custo, comunicação, robustez, cenário real*.

**Checklist do revisor (Introdução)**

* [ ] Problema definido em 1 frase
* [ ] Lacuna explícita
* [ ] Contribuições em bullets
* [ ] Mapa do artigo

---

# 2. Trabalhos Relacionados (5–10 parágrafos | ~1–1,5 página)

**Objetivo:** provar que você conhece o estado da arte e *posicionar sua lacuna*.

### Estrutura recomendada

**P1 — Agrupamento por categorias**

* Ex.: (i) métodos clássicos, (ii) deep learning, (iii) LLM/SLM, (iv) federado/privacidade

**P2–P(n-1) — 1 parágrafo por grupo**
Cada parágrafo deve ter:

1. **o que é** (1 frase)
2. **como funciona** (1–2 frases)
3. **limitação no seu cenário** (1–2 frases) — isso é crucial

**Pn — Síntese + ponte**

* “Portanto, este trabalho foca em…”

**Se usar tabela comparativa (opcional e forte):**

* Uma tabela “Método vs Requisitos” (privacidade, custo, rotulagem, distribuído, etc.)
* A tabela fecha a seção e “mata” a dúvida do revisor.

**Palavras comuns/esperadas:** *estado da arte, abordagens, limitações, custo computacional, rotulagem, generalização, privacidade, comunicação, comparação*.

---

# 3. Metodologia / Abordagem Proposta (subseções | ~2–3 páginas)

**Objetivo:** permitir que alguém replique: entradas, passos, algoritmos, hiperparâmetros e decisão.

## 3.1 Visão Geral do Pipeline (2–3 parágrafos + 1 figura recomendada)

**P1 — O que entra e sai**

* Input: [tipo de dado] → Output: [classe/score/anomalia]

**P2 — Passos do pipeline (numerado 1…N)**

* Pré-processamento → representação → treino local → agregação → inferência → decisão

**P3 — Onde está a “novidade”**

* 2–4 linhas destacando o diferencial

**Figura obrigatória (muito recomendada):**

* Um diagrama com blocos e setas mostrando o fluxo ponta-a-ponta.

**Palavras comuns:** *pipeline, fluxo, etapas, arquitetura, cliente/servidor, treinamento, inferência, decisão*.

---

## 3.2 Dados e Pré-processamento (2–4 parágrafos + 1 tabela opcional)

**O que precisa aparecer:**

* fonte do dado, campos/atributos, janelamento/segmentação, normalização, tokenização (se texto), balanceamento/filtragem
* como você constrói o “Content”/representação final

**Tabela opcional ótima:**

* “Antes vs Depois” (nº instâncias, features, classes, tamanho, etc.)

**Palavras comuns:** *dataset, pré-processamento, limpeza, janelas, transformação, representação, rótulos*.

---

## 3.3 Modelo / Treinamento (3–6 parágrafos)

**Inclua:**

* qual modelo, por quê (1 parágrafo)
* função de perda / objetivo
* hiperparâmetros principais (batch, lr, epochs/rounds, rank etc.)
* restrições (memória, quantização, tempo)

**Se for FL/PEFT (exemplo de estrutura):**

* (i) Treino local
* (ii) O que é enviado (gradientes? pesos? adapters?)
* (iii) Agregação (FedAvg etc.)
* (iv) Custo de comunicação (como você mede)

**Palavras comuns:** *modelo, ajuste fino, treinamento, hiperparâmetros, convergência, custo, comunicação*.

---

## 3.4 Critério de Decisão (2–4 parágrafos)

**O que precisa ficar cristalino:**

* qual é o score (ex.: perplexidade/perda/threshold/classificador)
* como define limiar (β, percentil, validação)
* como trata falso positivo/negativo
* como lida com classes/normal vs ataque/anomalia

**Palavras comuns:** *limiar, score, decisão, detecção, falso positivo, sensibilidade, threshold*.

---

# 4. Configuração Experimental (2–4 parágrafos + 1 tabela recomendada)

**Objetivo:** ser “reprodutível”.

**Obrigatório:**

* Hardware/ambiente (GPU/CPU/RAM)
* Dataset(s): versão, splits, balanceamento
* Protocolo: nº clientes, rounds, IID vs non-IID, seeds
* Baselines comparadas (pelo menos 1–3)
* Métricas (F1, AUC, precision/recall, tempo, bytes transmitidos)

**Tabela recomendada: “Setup do experimento”**

* dataset, #clientes, #rounds, lr, batch, rank, quantização, baselines, métricas

**Palavras comuns:** *cenário, configuração, protocolo, métricas, baseline, reprodutibilidade*.

---

# 5. Resultados e Discussão (4–8 parágrafos + figuras/tabelas)

**Objetivo:** responder à pergunta: “funciona, e vale a pena?”

### Estrutura recomendada

**P1 — O que será respondido**

* Declare 2–4 perguntas:

  * “Qual o desempenho?”
  * “Qual o custo de comunicação?”
  * “Efeito do rank/hiperparâmetros?”
  * “Robustez em non-IID?”

**P2–P4 — Resultado principal (desempenho)**

* 1 figura ou tabela com métricas
* explique *o que observar* (tendência/diferença)

**P5 — Custo (tempo/comunicação/memória)**

* figura/tabela de bytes transmitidos, tempo/round, etc.

**P6 — Ablation / Sensibilidade (opcional mas forte)**

* rank, nº clientes, non-IID, quantização, threshold

**P7 — Ameaças à validade**

* limitações, vieses, generalização, restrições do dataset

**P8 — Síntese**

* 2–4 linhas: “o que aprendemos”

**Palavras comuns:** *ganho, comparação, melhoria, redução, custo, comunicação, estabilidade, variação, trade-off*.

---

# 6. Conclusão e Trabalhos Futuros (2–4 parágrafos | ~0,5 página)

**P1 — Reafirme problema + proposta**
**P2 — Principais achados (com números)**
**P3 — Limitações**
**P4 — Próximos passos**

* novos datasets, cenários reais, ataques diferentes, escalabilidade, deploy

**Palavras comuns:** *concluímos, demonstramos, resultados, limitações, futuro, extensão*.

---

# Agradecimentos (1 parágrafo)

Financiamento, laboratório, bolsa, infraestrutura.

---

# Disponibilidade de Artefatos (1 parágrafo, se aplicável)

* link/repositório, scripts, seeds, instruções, licença
* checklist mínimo: “como rodar”, “como reproduzir figuras/tabelas”

---

## Regras práticas para Figuras e Tabelas (padrão SBRC que revisor gosta)

### Como introduzir uma figura (mini-template)

**Antes da figura (1 frase):**

* “A Figura X apresenta [o que é] para evidenciar [o que o leitor deve ver].”

**Depois da figura (2–5 linhas):**

* (i) tendência principal
* (ii) comparação com baseline
* (iii) implicação (“isso sugere que…”)
* (iv) ressalva (se necessário)

### Como introduzir uma tabela (mini-template)

**Antes (1 frase):**

* “A Tabela Y resume [o que] sob [condições].”

**Depois (2–4 linhas):**

* destaque 2–3 células importantes (melhor/worst/trade-off)
* diga o “takeaway”

---

## Banco rápido de palavras/expressões por seção (pra manter o texto “no trilho”)

* **Introdução:** motivação, desafio, lacuna, cenário, restrição, proposta, contribuições, objetivo
* **Related Work:** abordagens, estado da arte, comparação, limitações, custo, privacidade, escalabilidade
* **Metodologia:** pipeline, arquitetura, etapas, representação, treinamento, agregação, decisão, hiperparâmetros
* **Experimentos:** protocolo, configuração, baseline, métrica, reprodutibilidade, cenário, split
* **Resultados:** melhoria, trade-off, redução, estabilidade, variação, significância, análise, discussão
* **Conclusão:** demonstramos, resultados, limitações, próximos passos, generalização

---