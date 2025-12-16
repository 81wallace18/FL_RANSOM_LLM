# Contribuições do Artigo (estilo SBRC)

As principais contribuições deste artigo são:

1. **Detecção federada de ransomware em IoT/Edge com treinamento benign-only**: propomos um fluxo de detecção que ajusta um modelo de linguagem pequeno (SLM/LLM) **apenas com tráfego benigno** (Label=0), a partir de **fluxos tabulares convertidos em texto**, e detecta ransomware por degradação da qualidade preditiva (top‑k accuracy + limiar).

2. **Eficiência e privacidade via FL + LoRA/PEFT**: investigamos a viabilidade do treinamento colaborativo preservando privacidade (dados permanecem locais), comunicando apenas **adaptadores LoRA** agregados no servidor (FedAvg/weighted FedAvg), permitindo reduzir o custo de comunicação e de ajuste do modelo em cenários com recursos limitados.

3. **Análise sob heterogeneidade (Non‑IID) e impacto de seleção/agregação**: avaliamos a abordagem em cenários IID e **não‑IID** (incluindo skew de quantidade e particionamento por dispositivo), analisando o impacto de estratégias de seleção de clientes e de agregação ponderada por volume de dados local no desempenho do modelo global.

4. **Avaliação operacional além de F1**: além de F1/precisão/recall, reportamos métricas alinhadas a implantação em IoT/Edge, incluindo **FPR em benigno** e métricas temporais como **TTD (Time‑to‑Detection)** e **coverage**, bem como avaliação por **janelas temporais** quando aplicável.

5. **Rigor experimental e mitigação de *data leakage***: documentamos e mitigamos vazamento de rótulo em campos textuais do dataset (ex.: `Attack Name`), assegurando validade das métricas e evitando resultados artificialmente inflados.

