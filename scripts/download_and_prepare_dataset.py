#!/usr/bin/env python3
"""
================================================================================
Download e Preparacao do Dataset Edge-IIoTset para FL-TFlow
================================================================================

Este script baixa o dataset Edge-IIoTset do Kaggle e converte os arquivos PCAP
para o formato de fluxos de rede (CSV) compativel com o projeto FL-TFlow.

USO:
    python scripts/download_and_prepare_dataset.py

REQUISITOS:
    - Python 3.8+
    - Kaggle API configurada (~/.config/kaggle/kaggle.json)
    - Dependencias: kaggle, nfstream, pandas

SAIDA:
    - data/ids_ransomware/edge_ransomware_new/raw/Ransomware.csv
    - data/ids_ransomware/edge_ransomware_new/raw/Benign%20Traffic.csv

CITACAO:
    Ferrag, M.A., Friha, O., Hamouda, D., Maglaras, L., Janicke, H. (2022).
    "Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of
    IoT and IIoT Applications for Centralized and Federated Learning".
    IEEE Access. DOI: 10.1109/ACCESS.2022.3165809

================================================================================
"""

import os
import sys
import zipfile
from pathlib import Path
from datetime import datetime

# Diretorios
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DOWNLOAD_DIR = DATA_DIR / "edge_iiotset_full"
OUTPUT_DIR = DATA_DIR / "ids_ransomware" / "edge_ransomware_new" / "raw"

# Kaggle dataset
KAGGLE_DATASET = "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot"


def log(msg: str):
    """Log com timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def step_1_download_kaggle():
    """
    ETAPA 1: Download do dataset Edge-IIoTset via Kaggle API.

    O dataset contem:
    - Attack traffic/ : arquivos PCAP de diversos ataques (incluindo Ransomware)
    - Normal traffic/ : arquivos PCAP de sensores IoT (trafego benigno)
    """
    log("=" * 60)
    log("ETAPA 1: Download do Edge-IIoTset (Kaggle)")
    log("=" * 60)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Verificar se ja foi baixado
    zip_file = DOWNLOAD_DIR / "edgeiiotset-cyber-security-dataset-of-iot-iiot.zip"
    extract_marker = DOWNLOAD_DIR / "Edge-IIoTset dataset"

    if extract_marker.exists():
        log("Dataset ja existe. Pulando download.")
        return True

    log(f"Dataset: {KAGGLE_DATASET}")
    log(f"Destino: {DOWNLOAD_DIR}")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        log("Baixando... (pode demorar alguns minutos)")
        api.dataset_download_files(KAGGLE_DATASET, path=str(DOWNLOAD_DIR), unzip=False)

        log("Extraindo ZIP...")
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(DOWNLOAD_DIR)

        log("Download concluido!")
        return True

    except Exception as e:
        log(f"ERRO: {e}")
        log("Verifique se o Kaggle API esta configurado corretamente.")
        return False


def step_2_convert_pcap_to_flows():
    """
    ETAPA 2: Conversao dos arquivos PCAP para formato de fluxos CSV.

    Usa NFStream para extrair fluxos de rede dos PCAPs.
    NFStream gera features compativeis com CICFlowMeter.

    Arquivos processados:
    - Attack traffic/Ransomware attack.pcap -> Ransomware.csv
    - Normal traffic/*/[sensor].pcap -> Benign%20Traffic.csv
    """
    log("")
    log("=" * 60)
    log("ETAPA 2: Conversao PCAP -> CSV (NFStream)")
    log("=" * 60)

    try:
        from nfstream import NFStreamer
        import pandas as pd
    except ImportError:
        log("ERRO: nfstream nao instalado. Execute: pip install nfstream")
        return False

    dataset_dir = DOWNLOAD_DIR / "Edge-IIoTset dataset"
    attack_dir = dataset_dir / "Attack traffic"
    normal_dir = dataset_dir / "Normal traffic"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Processar Ransomware ---
    log("")
    log("Processando Ransomware...")
    ransomware_pcap = attack_dir / "Ransomware attack.pcap"

    if ransomware_pcap.exists():
        log(f"  Arquivo: {ransomware_pcap.name} ({ransomware_pcap.stat().st_size / 1024 / 1024:.1f} MB)")

        df = _pcap_to_cicflowmeter_format(ransomware_pcap, "Ransomware", label=1)

        if df is not None:
            output_path = OUTPUT_DIR / "Ransomware.csv"
            df.to_csv(output_path, index=False)
            log(f"  -> {len(df)} fluxos salvos em {output_path.name}")
    else:
        log(f"  ERRO: Arquivo nao encontrado: {ransomware_pcap}")
        return False

    # --- Processar trafego benigno (todos os sensores) ---
    log("")
    log("Processando trafego benigno (sensores IoT)...")

    benign_dfs = []
    sensor_dirs = [
        "Distance", "Flame_Sensor", "Heart_Rate", "IR_Receiver", "Modbus",
        "phValue", "Soil_Moisture", "Sound_Sensor", "Temperature_and_Humidity", "Water_Level"
    ]

    for sensor in sensor_dirs:
        sensor_dir = normal_dir / sensor
        pcap_file = sensor_dir / f"{sensor}.pcap"

        if pcap_file.exists():
            log(f"  {sensor}...")
            df = _pcap_to_cicflowmeter_format(pcap_file, "Benign Traffic", label=0)
            if df is not None:
                benign_dfs.append(df)
                log(f"    -> {len(df)} fluxos")

    if benign_dfs:
        benign_df = pd.concat(benign_dfs, ignore_index=True)
        output_path = OUTPUT_DIR / "Benign%20Traffic.csv"
        benign_df.to_csv(output_path, index=False)
        log(f"  Total: {len(benign_df)} fluxos salvos em {output_path.name}")

    return True


def _pcap_to_cicflowmeter_format(pcap_path: Path, attack_name: str, label: int):
    """
    Converte um PCAP para DataFrame no formato CICFlowMeter.

    NFStream extrai features de fluxo bidirecionais que sao mapeadas
    para o formato esperado pelo CICFlowMeter.
    """
    from nfstream import NFStreamer
    import pandas as pd

    try:
        streamer = NFStreamer(source=str(pcap_path), statistical_analysis=True)
        df = streamer.to_pandas()

        if len(df) == 0:
            return None

        # Evitar divisao por zero
        duration_sec = df['bidirectional_duration_ms'] / 1000.0
        duration_sec = duration_sec.replace(0, 0.001)

        # Mapear NFStream -> CICFlowMeter
        result = pd.DataFrame({
            # Identificadores
            'Flow ID': (df['src_ip'].astype(str) + '-' + df['dst_ip'].astype(str) + '-' +
                       df['src_port'].astype(str) + '-' + df['dst_port'].astype(str) + '-' +
                       df['protocol'].astype(str)),
            'Src IP': df['src_ip'],
            'Src Port': df['src_port'],
            'Dst IP': df['dst_ip'],
            'Dst Port': df['dst_port'],
            'Protocol': df['protocol'],
            'Timestamp': pd.to_datetime(df['bidirectional_first_seen_ms'], unit='ms').dt.strftime('%d/%m/%Y %I:%M:%S %p'),

            # Duracao (microsegundos)
            'Flow Duration': df['bidirectional_duration_ms'] * 1000,

            # Contagem de pacotes
            'Total Fwd Packet': df['src2dst_packets'],
            'Total Bwd packets': df['dst2src_packets'],

            # Bytes
            'Total Length of Fwd Packet': df['src2dst_bytes'],
            'Total Length of Bwd Packet': df['dst2src_bytes'],

            # Estatisticas de tamanho de pacote (forward)
            'Fwd Packet Length Max': df['src2dst_max_ps'],
            'Fwd Packet Length Min': df['src2dst_min_ps'],
            'Fwd Packet Length Mean': df['src2dst_mean_ps'],
            'Fwd Packet Length Std': df['src2dst_stddev_ps'],

            # Estatisticas de tamanho de pacote (backward)
            'Bwd Packet Length Max': df['dst2src_max_ps'],
            'Bwd Packet Length Min': df['dst2src_min_ps'],
            'Bwd Packet Length Mean': df['dst2src_mean_ps'],
            'Bwd Packet Length Std': df['dst2src_stddev_ps'],

            # Taxas
            'Flow Bytes/s': df['bidirectional_bytes'] / duration_sec,
            'Flow Packets/s': df['bidirectional_packets'] / duration_sec,

            # Inter-arrival time (IAT) - bidirectional
            'Flow IAT Mean': df['bidirectional_mean_piat_ms'] * 1000,
            'Flow IAT Std': df['bidirectional_stddev_piat_ms'] * 1000,
            'Flow IAT Max': df['bidirectional_max_piat_ms'] * 1000,
            'Flow IAT Min': df['bidirectional_min_piat_ms'] * 1000,

            # IAT forward
            'Fwd IAT Total': df['src2dst_duration_ms'] * 1000,
            'Fwd IAT Mean': df['src2dst_mean_piat_ms'] * 1000,
            'Fwd IAT Std': df['src2dst_stddev_piat_ms'] * 1000,
            'Fwd IAT Max': df['src2dst_max_piat_ms'] * 1000,
            'Fwd IAT Min': df['src2dst_min_piat_ms'] * 1000,

            # IAT backward
            'Bwd IAT Total': df['dst2src_duration_ms'] * 1000,
            'Bwd IAT Mean': df['dst2src_mean_piat_ms'] * 1000,
            'Bwd IAT Std': df['dst2src_stddev_piat_ms'] * 1000,
            'Bwd IAT Max': df['dst2src_max_piat_ms'] * 1000,
            'Bwd IAT Min': df['dst2src_min_piat_ms'] * 1000,

            # Flags PSH/URG
            'Fwd PSH Flags': df['src2dst_psh_packets'],
            'Bwd PSH Flags': df['dst2src_psh_packets'],
            'Fwd URG Flags': df['src2dst_urg_packets'],
            'Bwd URG Flags': df['dst2src_urg_packets'],

            # Header length (estimado)
            'Fwd Header Length': df['src2dst_packets'] * 20,
            'Bwd Header Length': df['dst2src_packets'] * 20,

            # Taxas de pacotes
            'Fwd Packets/s': df['src2dst_packets'] / duration_sec,
            'Bwd Packets/s': df['dst2src_packets'] / duration_sec,

            # Estatisticas de tamanho (bidirectional)
            'Packet Length Min': df['bidirectional_min_ps'],
            'Packet Length Max': df['bidirectional_max_ps'],
            'Packet Length Mean': df['bidirectional_mean_ps'],
            'Packet Length Std': df['bidirectional_stddev_ps'],
            'Packet Length Variance': df['bidirectional_stddev_ps'] ** 2,

            # TCP Flags
            'FIN Flag Count': df['bidirectional_fin_packets'],
            'SYN Flag Count': df['bidirectional_syn_packets'],
            'RST Flag Count': df['bidirectional_rst_packets'],
            'PSH Flag Count': df['bidirectional_psh_packets'],
            'ACK Flag Count': df['bidirectional_ack_packets'],
            'URG Flag Count': df['bidirectional_urg_packets'],
            'CWR Flag Count': df['bidirectional_cwr_packets'],
            'ECE Flag Count': df['bidirectional_ece_packets'],

            # Ratios
            'Down/Up Ratio': df['dst2src_packets'] / df['src2dst_packets'].replace(0, 1),
            'Average Packet Size': df['bidirectional_bytes'] / df['bidirectional_packets'].replace(0, 1),

            # Segment size
            'Fwd Segment Size Avg': df['src2dst_mean_ps'],
            'Bwd Segment Size Avg': df['dst2src_mean_ps'],

            # Bulk (nao disponivel no NFStream, usar 0)
            'Fwd Bytes/Bulk Avg': 0,
            'Fwd Packet/Bulk Avg': 0,
            'Fwd Bulk Rate Avg': 0,
            'Bwd Bytes/Bulk Avg': 0,
            'Bwd Packet/Bulk Avg': 0,
            'Bwd Bulk Rate Avg': 0,

            # Subflow
            'Subflow Fwd Packets': df['src2dst_packets'],
            'Subflow Fwd Bytes': df['src2dst_bytes'],
            'Subflow Bwd Packets': df['dst2src_packets'],
            'Subflow Bwd Bytes': df['dst2src_bytes'],

            # Init window (nao disponivel)
            'FWD Init Win Bytes': 0,
            'Bwd Init Win Bytes': 0,

            # Active data packets
            'Fwd Act Data Pkts': df['src2dst_packets'],
            'Fwd Seg Size Min': df['src2dst_min_ps'],

            # Active/Idle (nao disponivel)
            'Active Mean': 0,
            'Active Std': 0,
            'Active Max': 0,
            'Active Min': 0,
            'Idle Mean': 0,
            'Idle Std': 0,
            'Idle Max': 0,
            'Idle Min': 0,

            # Labels
            'Attack Name': attack_name,
            'Label': label,
        })

        return result.fillna(0)

    except Exception as e:
        print(f"    ERRO: {e}")
        return None


def step_3_summary():
    """
    ETAPA 3: Resumo do dataset gerado.
    """
    import pandas as pd

    log("")
    log("=" * 60)
    log("RESUMO")
    log("=" * 60)

    ransomware_path = OUTPUT_DIR / "Ransomware.csv"
    benign_path = OUTPUT_DIR / "Benign%20Traffic.csv"

    r_df = pd.read_csv(ransomware_path)
    b_df = pd.read_csv(benign_path, low_memory=False)

    total = len(r_df) + len(b_df)

    log(f"Ransomware:  {len(r_df):>10,} fluxos ({100*len(r_df)/total:.2f}%)")
    log(f"Benign:      {len(b_df):>10,} fluxos ({100*len(b_df)/total:.2f}%)")
    log(f"Total:       {total:>10,} fluxos")
    log("")
    log(f"Arquivos salvos em: {OUTPUT_DIR}")
    log("")
    log("Pronto para usar com: dataset_name: 'edge_ransomware_new'")


def main():
    """Pipeline completo."""
    print("""
================================================================================
    DOWNLOAD E PREPARACAO DO DATASET EDGE-IIoTSET
================================================================================
    """)

    if not step_1_download_kaggle():
        sys.exit(1)

    if not step_2_convert_pcap_to_flows():
        sys.exit(1)

    step_3_summary()

    log("")
    log("Concluido com sucesso!")


if __name__ == "__main__":
    main()
