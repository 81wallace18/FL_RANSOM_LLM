## Cenário do Artigo: Detecção Federada de Ransomware com Modelos de Linguagem

Este documento resume a relação entre o artigo base, o cenário que estamos construindo e as modificações feitas no código.

---

## 1. Artigo Base

**Ideia central**

- Problema: detecção de anomalias em logs de sistemas distribuídos, preservando privacidade e reduzindo custo de comunicação/treino.
- Abordagem:
  - Treinar um **modelo de linguagem** apenas em logs **normais**.
  - Em sessões anômalas (falhas/ataques), a acurácia de predição do próximo token cai; essa queda é usada como sinal de anomalia.

**Solução técnica**

- **Aprendizado Federado (FL)**:
  - Vários clientes treinam localmente e enviam apenas atualizações de modelo (não enviam logs crus).
  - O servidor agrega as atualizações (FedAvg) e produz um modelo global por rodada.
- **Fine-tuning eficiente (LoRA/PEFT)**:
  - Em vez de ajustar todos os parâmetros do LLM, ajusta-se apenas adapters de baixa dimensão (LoRA).
  - Isso reduz:
    - Número de parâmetros treináveis.
    - Tamanho das mensagens enviadas pelos clientes.
    - Custo computacional por rodada.
- **Detecção de anomalias**:
  - Para cada sessão de log, calcula-se a **top-k accuracy** na tarefa de predição do próximo token.
  - Varre-se um limiar de accuracy; abaixo do limiar = anomalia, acima = normal.
  - Escolhe-se o limiar que maximiza a F1 (método de avaliação "antigo").

**Cenário experimental original**

- Dataset de logs HDFS (sistema de arquivos distribuído).
- Múltiplos tamanhos de modelo (135M, 360M, 1.7B) e diferentes ranks LoRA.
- Demonstra trade-off entre:
  - Performance (F1, precisão, recall).
  - Custo de comunicação (tamanho das mensagens) e custo de treino (FLOPs).

---

## 2. Cenário do Novo Artigo

**Problema alvo**

- Detecção federada de **ransomware** em **logs de IDS / fluxos de rede IoT**, preservando:
  - Privacidade dos dados (logs não saem dos clientes).
  - Eficiência de comunicação e treino (uso de LoRA).

**Dataset utilizado**

- Família: **CIC-BCCC-NRC-TabularIoTAttacks-2024**.
- Subconjunto específico: **CIC-BCCC-NRC-Edge-IIoTSet-2022**.
- Arquivos usados:
  - `Benign Traffic.csv` &rarr; sessões normais (**Label = 0**).
  - `Ransomware.csv` &rarr; sessões com tráfego de **ransomware** (**Label = 1**).
- Apenas esses dois arquivos são usados; outros ataques (Brute Force, Scan etc.) foram removidos do cenário atual para manter o foco em benigno vs ransomware.

**Pré-processamento e representação textual**

- Implementado em `src/data_processing/edge_ransomware_processor.py` (`EdgeRansomwareProcessor`).
- Para cada linha (fluxo) dos CSVs:
  - Constrói-se um texto (`Content`) a partir de campos como:
    - `Flow ID`
    - `Src IP`, `Src Port`
    - `Dst IP`, `Dst Port`
    - `Protocol`
    - `Flow Duration`
    - `Total Fwd Packet`, `Total Bwd packets`
    - `Total Length of Fwd Packet`, `Total Length of Bwd Packet`
  - **Importante**: o campo `Attack Name` **não** é incluído no texto, para evitar vazamento de rótulo (data leakage).
- O dataset combinado (benigno + ransomware) é baralhado e dividido em:
  - `train.csv` (80%)
  - `test.csv` (20%)
  - Ambos com colunas `Content` e `Label`.

**Sanitização**

- Ainda no `EdgeRansomwareProcessor`, a função `preprocess_and_sanitize` aplica regex para:
  - Mascarar apenas endereços IP (`\b(?:\d{1,3}\.){3}\d{1,3}\b`), substituindo-os por um token genérico (`IP_ADDR`).
- Valores numéricos (duração, contagem de pacotes, bytes/s, variância de tamanho etc.) **são mantidos**, pois carregam a informação estatística essencial para distinguir tráfego benigno de ransomware.

**Tokenização**

- Apenas as linhas com `Label == 0` (tráfego benigno) do `train.csv` são usadas para treinar o modelo.
- É criado um `Dataset` com coluna `text = Content`.
- Tokenização:
  - Modelo: `HuggingFaceTB/SmolLM-135M`.
  - `pad_token = eos_token`.
  - `max_length = 1024` (comprimento máximo usado na tokenização atual).
  - Os `input_ids` são copiados para `labels` (treino de linguagem autoregressivo).
- O dataset tokenizado é salvo em `data/ids_ransomware/edge_ransomware/processed/tokenized/`.

---

## 3. Metodologia de Treino Federado e Fine-Tuning

**Configuração de experimento**

- Arquivo: `configs/config_edge_ransomware.yaml`.
- Parâmetros principais:
  - `simulation_name: "EdgeRansomware_SmolLM135M_LoRA_R8"`.
  - `dataset_name: "edge_ransomware"`.
  - `data_base_path: "./data/ids_ransomware/"`.
  - Modelo: `model_name: "HuggingFaceTB/SmolLM-135M"`.
  - LoRA:
    - `lora: True`
    - `lora_rank: 8`
    - `lora_alpha_multiplier` e `lora_dropout` definidos conforme o artigo base.
  - Aprendizado Federado:
    - `num_rounds: 50`
    - `num_clients: 50`
    - `client_frac: 0.1` (5 clientes por rodada).
    - `use_parallel_training: false` (modo sequencial estável em uma GPU).
  - Treino do cliente:
    - `max_steps: 5`
    - `batch_size: 2`

**Inicialização do modelo global**

- Implementado em `src/models/model_loader.py`:
  - Carrega o LLM pré-treinado (`SmolLM-135M`) da Hugging Face.
  - Aplica LoRA (`LoraConfig`) ao modelo:
    - Apenas ~0,34% dos parâmetros tornam-se treináveis (~460k de 135M).
  - Salva o modelo (com adapters) como **round_0/global_model**:
    - `results/EdgeRansomware_SmolLM135M_LoRA_R8/round_0/global_model/`.

**Treino local do cliente (fine-tuning eficiente)**

- Implementado em `src/federated_learning/client.py` (`ClientTrainer`):
  - Para cada rodada e cliente selecionado:
    - Carrega o modelo global da rodada anterior:
      ```python
      model = AutoModelForCausalLM.from_pretrained(
          self.model_name,
          torch_dtype=torch.float16,
          device_map=device_map,
      )
      model = PeftModel.from_pretrained(model, model_path, is_trainable=True, device_map=device_map)
      ```
    - Carrega o shard tokenizado do cliente (`load_from_disk`).
    - Cria um `Trainer` com:
      - `max_steps = 5`
      - `batch_size = 2`
      - Otimizador 8-bit (`paged_adamw_8bit`), se configurado.
    - Executa `trainer.train()`.
    - Extrai apenas os parâmetros `lora_*` do `state_dict` para CPU e devolve ao servidor.

**Agregação no servidor (FedAvg dos adapters)**

- Implementado em `src/federated_learning/server.py` (`FederatedServer`):
  - Divide o dataset tokenizado em shards para os 50 clientes.
  - Em cada rodada:
    - Seleciona aleatoriamente `client_frac * num_clients` clientes.
    - Para cada cliente selecionado:
      - Chama `ClientTrainer.train`, recebe os adapters LoRA no CPU.
    - Faz `FedAvg` dos adapters:
      - Para cada parâmetro `lora_*`, calcula a média sobre os clientes.
    - Atualiza o modelo global com os adapters agregados.
    - Salva o novo modelo global em `round_N/global_model`.

---

## 4. Avaliação e Métrica

**Dataset de teste**

- Usa-se o `test.csv` gerado pelo `EdgeRansomwareProcessor`, que contém:
  - Sessões **benignas**.
  - Sessões de **ransomware**.
- O avaliador "antigo" (`src/evaluation/evaluator_antigo.py`) faz:
  - Balanceamento do teste:
    - Seleciona um subconjunto equilibrado, por exemplo:
      - 130 sessões anômalas (ransomware).
      - 1000 sessões normais.

**Cálculo da métrica (método antigo)**

- Para cada rodada e para cada valor de `k` em `top_k_values = [1, 3, 5, 10]`:
  - Calcula a **top-k accuracy** por sessão (sem o "shift" de tokens, como no script original).
  - Varre um conjunto de limiares (`f1_threshold_steps`) entre 0 e 1.
  - Para cada limiar, define:
    - `accuracy < threshold` &rarr; predição de **anomalia** (1).
    - `accuracy >= threshold` &rarr; **normal** (0).
  - Calcula F1, precisão e recall para cada limiar.
  - Escolhe o limiar que maximiza a F1 para aquele `k` e rodada.
- Ao final, salva um CSV:
  - `results/EdgeRansomware_SmolLM135M_LoRA_R8/f1_scores_antigo.csv`
  - Com colunas como:
    - `round`, `k`, `f1_score`, `threshold`, `precision`, `recall`.

---

## 5. Considerações Sobre o F1 e Data Leakage

- Em uma primeira versão do processor, o campo `Attack Name` foi incluído no texto (`Content`):
  ```python
  f"attack {row.get('Attack Name', '')}",
  ```
  - No Edge-IIoTSet, `Attack Name` é o **rótulo humano** ("Benign Traffic", "Ransomware").
  - Isso introduz **data leakage**: o modelo tem acesso direto ao rótulo no próprio texto.
  - Com isso, o F1 chegou a 1.0 de forma artificial.
- Correção aplicada:
  - Remoção de `Attack Name` do `Content` em `_build_text_from_row`.
  - Reprocessamento dos dados (`force_reprocess_data: True` temporariamente).
  - Reexecução do FL e avaliação.
- Resultado:
  - A detecção torna-se mais difícil (como esperado), mas cientificamente válida.
  - O modelo passa a basear-se em padrões de tráfego (duração, bytes, direção, etc.), não no nome do ataque.

---

## 6. Resumo para o Artigo

Em resumo, o novo artigo:

- **Reutiliza a metodologia do artigo base**:
  - LLM treinado apenas em tráfego/logs normais.
  - Aprendizado Federado para preservar privacidade.
  - Fine-tuning eficiente com LoRA para reduzir custo de comunicação.
  - Detecção baseada em top-k accuracy e busca de limiar que maximiza F1.

- **Adapta o cenário para ransomware em IoT/Edge**:
  - Usa subset público do **CIC-BCCC-NRC-Edge-IIoTSet-2022**:
    - `Benign Traffic.csv` vs `Ransomware.csv`.
  - Constrói sessões textuais a partir de fluxos tabulares.
  - Remove campos que causam vazamento de rótulo.

- **Mostra empiricamente**:
  - Que um modelo relativamente pequeno (SmolLM-135M) com LoRA, treinado federadamente em dados normais, consegue distinguir sessões de ransomware de tráfego benigno com F1 alta em um dataset público.
  - Que isso é feito com uma fração dos parâmetros treinados e do custo de comunicação, alinhado com o foco em eficiência e privacidade.
