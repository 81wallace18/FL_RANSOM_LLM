# FL-TFlow: Federated LoRA Tuning of SLMs for Edge Ransomware Detection

A privacy-preserving framework for ransomware detection in IoT/Edge environments using Federated Learning with Parameter-Efficient Fine-Tuning (LoRA) of Small Language Models.

## Overview

**FL-TFlow** addresses the challenge of detecting ransomware in distributed IoT/Edge networks where:
- Raw data cannot be centralized due to privacy constraints
- Devices have limited computational resources
- Labeled attack data is scarce or unavailable

The framework uses a **benign-only training strategy**: by learning exclusively from normal network traffic patterns, the model detects ransomware through degradation in token-prediction scores, enabling zero-day attack detection without prior exposure to attack samples.

## Key Features

- **Privacy-Preserving**: Raw network flows never leave client devices; only lightweight LoRA adapter updates are shared
- **Communication-Efficient**: LoRA reduces parameter exchange by orders of magnitude compared to full model fine-tuning
- **Benign-Only Detection**: No labeled attack data required for training
- **Flow-to-Text Serialization**: Converts numerical network flow records into textual sequences for language model processing

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FL-TFlow Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Client 1   │    │   Client 2   │    │   Client N   │      │
│  │ ┌──────────┐ │    │ ┌──────────┐ │    │ ┌──────────┐ │      │
│  │ │ Benign   │ │    │ │ Benign   │ │    │ │ Benign   │ │      │
│  │ │ Flows    │ │    │ │ Flows    │ │    │ │ Flows    │ │      │
│  │ └────┬─────┘ │    │ └────┬─────┘ │    │ └────┬─────┘ │      │
│  │      ▼       │    │      ▼       │    │      ▼       │      │
│  │ Flow-to-Text │    │ Flow-to-Text │    │ Flow-to-Text │      │
│  │      ▼       │    │      ▼       │    │      ▼       │      │
│  │ Local Train  │    │ Local Train  │    │ Local Train  │      │
│  │ (LoRA+FedProx│    │ (LoRA+FedProx│    │ (LoRA+FedProx│      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             ▼                                   │
│                 ┌───────────────────────┐                       │
│                 │    Federated Server   │                       │
│                 │  ┌─────────────────┐  │                       │
│                 │  │ FedAvg Aggreg.  │  │                       │
│                 │  │ Global SLM+LoRA │  │                       │
│                 │  └─────────────────┘  │                       │
│                 └───────────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Methodology

### 1. Flow-to-Text Serialization

Network flows are converted into structured text sequences:

```
Input (Raw Flow):
┌────────────────────────────────────────────────────┐
│ Protocol: TCP, Duration: 0.5s, Fwd_Pkts: 10, ...   │
└────────────────────────────────────────────────────┘
                         ▼
Output (Text Content):
┌────────────────────────────────────────────────────┐
│ "protocol TCP flow_duration 0.5 fwd_packets 10 ..." │
└────────────────────────────────────────────────────┘
```

Label-carrying fields (e.g., `Attack_Name`) are excluded to prevent data leakage.

### 2. Benign-Only Training

The SLM learns the "grammar" of normal network behavior through next-token prediction. During inference:
- **Normal flows**: High prediction accuracy (tokens are predictable)
- **Ransomware flows**: Low prediction accuracy (anomalous patterns)

### 3. Scoring and Detection

For each flow, a top-k token-prediction score is computed:

```
score(x) = (1/L) * Σ I[x_t ∈ Top-k predictions]

If score < threshold β → Ransomware (Anomaly)
If score ≥ threshold β → Benign
```

## Project Structure

```
FL-TFlow/
├── configs/
│   └── config.yaml              # Experiment configuration
├── src/
│   ├── data_processing/
│   │   ├── base_processor.py    # Abstract processor interface
│   │   └── ransomlog_processor.py
│   ├── federated_learning/
│   │   ├── server.py            # FL server (FedAvg aggregation)
│   │   └── client.py            # Client training logic
│   ├── evaluation/
│   │   └── evaluator.py         # Top-k accuracy & F1 evaluation
│   └── models/
│       └── model_loader.py      # Model initialization with LoRA
├── main.py                      # Main entry point
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/81wallace18/FL_RANSOM_LLM.git
cd FL_RANSOM_LLM

# Create virtual environment
uv venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -r requirements.txt
```

## Usage

### 1. Prepare Data

Place your dataset in `data/<dataset_name>/raw/`. The framework expects network flow data that can be converted to the Flow-to-Text format.

### 2. Configure Experiment

Edit `configs/config.yaml`:

```yaml
# General
simulation_name: "experiment_01"
dataset_name: "ransomlog"

# Model
model_name: "HuggingFaceTB/SmolLM-135M"
lora: True
lora_rank: 16

# Federated Learning
num_rounds: 30
num_clients: 50
client_frac: 0.2

# Training
max_steps: 50
batch_size: 4
initial_lr: 0.001
```

### 3. Run Training

```bash
python main.py --config configs/config.yaml
```

The pipeline will:
1. Process and tokenize the data
2. Initialize the global SLM with LoRA
3. Run federated training for the specified rounds
4. Evaluate and save results to `results/<simulation_name>/`

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_name` | HuggingFace model identifier | SmolLM-135M |
| `lora_rank` | LoRA adapter rank | 16 |
| `num_rounds` | FL communication rounds | 30 |
| `num_clients` | Total number of clients | 50 |
| `client_frac` | Fraction of clients per round | 0.2 |
| `max_steps` | Local training steps per client | 50 |
| `top_k_values` | Values of k for evaluation | [1, 3, 5, 10] |

## Results

Experimental results on the Edge-IIoTSet dataset demonstrate:

| Metric | FL-TFlow | Baseline |
|--------|----------|----------|
| Flow F1-Score | **0.937** | 0.026 |
| Precision | 0.997 | 0.084 |
| Recall | 0.884 | 0.016 |
| Benign FPR | **0.012%** | 0.667% |

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{cruz2025fltflow,
  title={FL-TFlow: Benign-Only Federated LoRA Tuning of SLMs for Edge Ransomware Detection},
  author={Cruz, Wallace P. and Pinto, Jose and Marques, Thiago and Veiga, Rafael and Santos, Hugo and Bastos, Lucas and Talasso, Gabriel and Costa, Allan and Ros{\'a}rio, Denis and Cerqueira, Eduardo},
  booktitle={SBRC 2025},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Federal University of Pará (UFPA)
- Federal University of South and Southeast of Pará (UNIFESPA)
- Federal Rural University of the Amazon (UFRA)
- State University of Pará (UEPA)
- University of Campinas (UNICAMP)
