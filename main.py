import yaml
import argparse
import os
from src.data_processing.ransomlog_processor import RansomLogProcessor
from src.data_processing.hdfs_processor import HDFSProcessor
from src.data_processing.edge_ransomware_processor import EdgeRansomwareProcessor
from src.federated_learning.server import FederatedServer
from src.evaluation.evaluator_antigo import Evaluator

def main(config_path):
    """
    Main function to orchestrate the federated learning pipeline.
    """
    # 1. Load Configuration
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
        
    print("Configuration loaded successfully.")
    print(f"Starting simulation: {config.get('simulation_name', 'N/A')}")

    # 2. Execute Data Processing Pipeline
    # This factory pattern dynamically selects the correct processor based on the config.
    print("\n--- Initializing Data Processing ---")
    if config['dataset_name'] == 'ransomlog':
        processor = RansomLogProcessor(config)
    elif config['dataset_name'] == 'edge_ransomware':
        processor = EdgeRansomwareProcessor(config)
    elif config['dataset_name'] == 'hdfs':
        processor = HDFSProcessor(config)
    else:
        raise ValueError(f"Dataset '{config['dataset_name']}' not supported in the current implementation.")
    
    processor.run()

    # 3. Execute Federated Training
    print("\n--- Starting Federated Training ---")
    server = FederatedServer(config)
    server.run_federated_training()
    print("--- Federated Training Complete ---")

    # 4. Execute Evaluation
    print("\n--- Starting Evaluation ---")
    evaluator = Evaluator(config)
    evaluator.evaluate()
    print("--- Evaluation Complete ---")

import multiprocessing as mp

if __name__ == "__main__":
    # Define o método de início de multiprocessamento como 'spawn'.
    # Isso é crucial para evitar erros de inicialização da CUDA em processos filhos.
    # Deve ser chamado dentro deste bloco if e antes de qualquer código de paralelismo.
    try:
        mp.set_start_method('spawn', force=True)
        print("Método de início de multiprocessamento configurado para 'spawn'.")
    except RuntimeError:
        # Pode já ter sido definido, o que não é um problema.
        pass

    parser = argparse.ArgumentParser(description="Run Federated Learning for Anomaly Detection.")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to the YAML configuration file.')
    args = parser.parse_args()
    
    # Ensure the script's working directory is the project root
    # This makes path handling in config.yaml consistent
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    main(args.config)
