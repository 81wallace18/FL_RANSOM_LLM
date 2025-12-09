import os
import random
import math
import json
import torch
from datasets import load_from_disk
import concurrent.futures

from src.models.model_loader import initialize_global_model
from .client import ClientTrainer

def train_client_process(args):
    """
    Função wrapper para treinar um cliente em um processo separado.
    Isso é necessário para o paralelismo com ProcessPoolExecutor.
    """
    client_id, config, round_num, learning_rate, gpu_id = args
    
    # Define qual GPU este processo deve usar
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    
    print(f"Iniciando treinamento para o cliente {client_id} na GPU {gpu_id}")
    # Passa o gpu_id para o ClientTrainer
    client_trainer = ClientTrainer(client_id, config, gpu_id)
    cpu_weights = client_trainer.train(round_num, learning_rate)
    return cpu_weights

class FederatedServer:
    """
    Orchestrates the federated learning process, including client selection,
    training, and model aggregation.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model, self.tokenizer = initialize_global_model(config)
        self.global_model.to(self.device)

        # Mantém metadados sobre o tamanho de cada shard de cliente para
        # permitir seleção adaptativa e agregação ponderada.
        self.client_sample_counts = {}

    def _split_data_for_clients(self):
        """
        Splits the main tokenized dataset into shards for each client.
        Supports IID and Non-IID strategies and records per-client sample counts.
        This is a one-time setup operation per simulation.
        """
        client_data_base_path = os.path.join(
            self.config['results_path'],
            self.config['simulation_name'],
            'client_data'
        )
        metadata_path = os.path.join(client_data_base_path, "client_data_metadata.json")

        # Avoid re-splitting if already done
        if os.path.exists(client_data_base_path) and os.listdir(client_data_base_path):
            print("Client data shards already exist. Skipping split.")
            # Tenta carregar metadados previamente salvos para uso em seleção/agragação
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        data = json.load(f)
                    # Converte chaves para int
                    self.client_sample_counts = {int(k): int(v) for k, v in data.items()}
                    print("Loaded client sample counts metadata.")
                except Exception as e:
                    print(f"Warning: Failed to load client metadata: {e}")
            else:
                # Best-effort: infere contagens carregando os datasets de cada cliente
                print("Client metadata not found. Inferring sample counts from disk...")
                self.client_sample_counts = {}
                for client_id in range(self.config['num_clients']):
                    client_path = os.path.join(client_data_base_path, f'client_{client_id}')
                    if os.path.exists(client_path):
                        try:
                            client_dataset = load_from_disk(client_path)
                            self.client_sample_counts[client_id] = len(client_dataset)
                        except Exception as e:
                            print(f"  Warning: Could not load data for client {client_id}: {e}")
            return

        os.makedirs(client_data_base_path, exist_ok=True)
        
        tokenized_dataset_path = os.path.join(
            self.config['data_base_path'],
            self.config['dataset_name'],
            'processed',
            'tokenized'
        )
        dataset = load_from_disk(tokenized_dataset_path)['train']

        num_clients = self.config['num_clients']
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        # Cria splits conforme a estratégia configurada
        strategy = self.config.get('data_distribution_strategy', 'iid')
        client_splits = {i: [] for i in range(num_clients)}

        if strategy == 'iid':
            # Distribuição IID simples (round-robin após embaralhar)
            for i in range(num_clients):
                client_splits[i] = indices[i::num_clients]
        elif strategy == 'quantity_skew_dirichlet':
            try:
                import numpy as np
            except ImportError:
                print("Warning: numpy not available, falling back to IID split.")
                for i in range(num_clients):
                    client_splits[i] = indices[i::num_clients]
            else:
                alpha = float(self.config.get('non_iid_alpha', 0.5))
                dirichlet = np.random.dirichlet([alpha] * num_clients)
                counts = (dirichlet * len(indices)).astype(int)
                # Ajusta para garantir que a soma dos counts seja exatamente o total
                diff = len(indices) - int(counts.sum())
                step = 1 if diff > 0 else -1
                for i in range(abs(diff)):
                    counts[i % num_clients] += step

                cursor = 0
                for client_id, count in enumerate(counts):
                    client_splits[client_id] = indices[cursor:cursor + int(count)]
                    cursor += int(count)
        elif strategy == 'hetero_device':
            # Simula grupos de dispositivos com diferentes capacidades.
            # Pequeno (~20%), médio (~40%), grande (~40%) com pesos distintos.
            client_ids = list(range(num_clients))
            small_end = max(1, int(0.2 * num_clients))
            medium_end = max(small_end + 1, int(0.6 * num_clients))

            size_weights = {}
            for cid in client_ids[:small_end]:
                size_weights[cid] = 1.0   # dispositivos leves
            for cid in client_ids[small_end:medium_end]:
                size_weights[cid] = 3.0   # dispositivos intermediários
            for cid in client_ids[medium_end:]:
                size_weights[cid] = 6.0   # gateways/edge poderosos

            total_weight = sum(size_weights.values())
            clients = list(size_weights.keys())
            weights = [size_weights[cid] / total_weight for cid in clients]

            for idx in indices:
                chosen = random.choices(clients, weights=weights, k=1)[0]
                client_splits[chosen].append(idx)
        else:
            print(f"Warning: Unknown data_distribution_strategy='{strategy}', falling back to IID.")
            for i in range(num_clients):
                client_splits[i] = indices[i::num_clients]

        # Salva shards e registra contagem de amostras por cliente
        self.client_sample_counts = {}
        for client_id, client_indices in client_splits.items():
            client_dataset = dataset.select(client_indices)
            self.client_sample_counts[client_id] = len(client_dataset)
            client_dataset.save_to_disk(os.path.join(client_data_base_path, f'client_{client_id}'))
        
        try:
            with open(metadata_path, "w") as f:
                json.dump(self.client_sample_counts, f)
        except Exception as e:
            print(f"Warning: Failed to save client metadata: {e}")

        print(f"Data successfully split for {num_clients} clients using strategy '{strategy}'.")

    def _get_learning_rate(self, current_round):
        """Calculates the learning rate for the current round based on the schedule."""
        if self.config['lr_scheduler_type'] == 'cosine':
            initial_lr = self.config['initial_lr']
            min_lr = self.config['min_lr']
            total_rounds = self.config['num_rounds']
            
            return min_lr + 0.5 * (initial_lr - min_lr) * \
                   (1 + math.cos(math.pi * current_round / total_rounds))
        else: # constant
            return self.config['initial_lr']

    def _get_adapters(self, model):
        """Extracts LoRA adapter weights from a model."""
        return {name: param.data.clone() for name, param in model.named_parameters() if "lora_" in name}

    def _set_adapters(self, model, aggregated_adapters):
        """Updates the model with aggregated LoRA adapter weights."""
        for name, param in model.named_parameters():
            if name in aggregated_adapters:
                param.data.copy_(aggregated_adapters[name].to(self.device))

    def _aggregate_models(self, client_weights_list, client_ids=None):
        """
        Aggregates client model weights (adapters or full) using FedAvg.
        The aggregation is done on the CPU.
        """
        # The keys are the same for all clients, so we can take them from the first one.
        if not client_weights_list:
            print("Warning: Client weights list is empty. Skipping aggregation.")
            return

        aggregated_weights = {}
        weight_keys = client_weights_list[0].keys()

        # Configuração opcional: agregação ponderada pelo número de amostras por cliente
        use_weighted_agg = self.config.get('use_weighted_aggregation', False)
        sample_weights = None
        if use_weighted_agg and client_ids is not None and self.client_sample_counts:
            counts = [self.client_sample_counts.get(cid, 0) for cid in client_ids]
            total = sum(counts)
            if total > 0:
                sample_weights = [c / total for c in counts]
            else:
                print("Warning: Sample counts sum to zero. Falling back to unweighted aggregation.")
                sample_weights = None

        for key in weight_keys:
            # Stack all tensors for the current key from all clients and average them.
            # The tensors are on the CPU, so this uses RAM, not VRAM.
            if sample_weights is not None:
                stacked = torch.stack(
                    [sample_weights[i] * client_weights_list[i][key] for i in range(len(client_weights_list))],
                    dim=0
                )
                aggregated_weights[key] = torch.sum(stacked, dim=0)
            else:
                aggregated_weights[key] = torch.mean(
                    torch.stack([weights[key] for weights in client_weights_list]), dim=0
                )

        if self.config['lora']:
            self._set_adapters(self.global_model, aggregated_weights)
        else:
            # For full fine-tuning, update the entire model state dict
            # Move weights to GPU before loading them into the model
            gpu_aggregated_weights = {k: v.to(self.device) for k, v in aggregated_weights.items()}
            self.global_model.load_state_dict(gpu_aggregated_weights)

    def _select_clients_for_round(self, round_num):
        """
        Selects clients for the current round according to the configured strategy.
        Supports uniform random selection and selection proportional to local data size.
        """
        num_clients = self.config['num_clients']
        num_selected_clients = int(num_clients * self.config['client_frac'])
        all_clients = list(range(num_clients))

        strategy = self.config.get('client_selection_strategy', 'uniform')

        if strategy == 'data_size_proportional' and self.client_sample_counts:
            counts = [self.client_sample_counts.get(cid, 0) for cid in all_clients]
            total = sum(counts)
            if total <= 0:
                print("Warning: Invalid or zero client sample counts. Falling back to uniform selection.")
                return random.sample(all_clients, num_selected_clients)

            probabilities = [c / total for c in counts]
            selected_clients = set()
            # Garante seleção sem reposição usando amostragem ponderada
            while len(selected_clients) < num_selected_clients:
                chosen = random.choices(all_clients, weights=probabilities, k=1)[0]
                selected_clients.add(chosen)
            selected_clients_ids = list(selected_clients)
            print(f"Client selection strategy 'data_size_proportional' selected: {selected_clients_ids}")
            return selected_clients_ids

        # Estratégia padrão: seleção uniforme aleatória
        selected_clients_ids = random.sample(all_clients, num_selected_clients)
        print(f"Client selection strategy 'uniform' selected: {selected_clients_ids}")
        return selected_clients_ids

    def run_federated_training(self):
        """The main federated training loop."""
        self._split_data_for_clients()

        # --- Lógica Condicional para Paralelismo ---
        if self.config.get('use_parallel_training', False):
            self._run_parallel_training()
        else:
            self._run_sequential_training()

    def _run_sequential_training(self):
        """Executes training sequentially in the main process."""
        print("--- Running in Sequential Mode ---")
        for round_num in range(1, self.config['num_rounds'] + 1):
            print(f"\n===== Starting Round {round_num}/{self.config['num_rounds']} =====")

            selected_clients_ids = self._select_clients_for_round(round_num)

            client_weights_list = []
            successful_client_ids = []
            current_lr = self._get_learning_rate(round_num)
            for client_id in selected_clients_ids:
                client_trainer = ClientTrainer(client_id, self.config)
                cpu_weights = client_trainer.train(round_num, current_lr)
                if cpu_weights:
                    client_weights_list.append(cpu_weights)
                    successful_client_ids.append(client_id)
            
            print("Aggregating client models...")
            self._aggregate_models(client_weights_list, successful_client_ids)
            
            round_model_path = os.path.join(
                self.config['results_path'], self.config['simulation_name'],
                f'round_{round_num}', 'global_model'
            )
            os.makedirs(round_model_path, exist_ok=True)
            self.global_model.save_pretrained(round_model_path)
            print(f"New global model for round {round_num} saved to: {round_model_path}")

    def _run_parallel_training(self):
        """Executes training in parallel across multiple GPUs."""
        print("--- Running in Parallel Mode ---")
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("AVISO: Nenhuma GPU encontrada. Voltando para o modo sequencial.")
            self._run_sequential_training()
            return
        
        print(f"Encontradas {num_gpus} GPUs. Distribuindo clientes entre elas.")

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        for round_num in range(1, self.config['num_rounds'] + 1):
            print(f"\n===== Starting Round {round_num}/{self.config['num_rounds']} =====")

            selected_clients_ids = self._select_clients_for_round(round_num)

            client_weights_list = []
            successful_client_ids = []
            current_lr = self._get_learning_rate(round_num)

            # Mover o modelo global para a CPU para liberar VRAM para os clientes
            self.global_model.to('cpu')
            torch.cuda.empty_cache()

            client_chunks = list(chunks(selected_clients_ids, num_gpus))
            for i, client_chunk in enumerate(client_chunks):
                print(f"  --- Processing client batch {i+1}/{len(client_chunks)} ---")
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
                    future_to_client = {}
                    for j, client_id in enumerate(client_chunk):
                        gpu_id = j % num_gpus
                        args = (client_id, self.config, round_num, current_lr, gpu_id)
                        future = executor.submit(train_client_process, args)
                        future_to_client[future] = client_id

                    for future in concurrent.futures.as_completed(future_to_client):
                        client_id = future_to_client[future]
                        try:
                            cpu_weights = future.result()
                            if cpu_weights:
                                client_weights_list.append(cpu_weights)
                                successful_client_ids.append(client_id)
                        except Exception as e:
                            print(f"Erro ao treinar cliente: {e}")
            
            # Mover o modelo de volta para a GPU para agregação
            self.global_model.to(self.device)

            print("Aggregating client models...")
            self._aggregate_models(client_weights_list, successful_client_ids)
            
            round_model_path = os.path.join(
                self.config['results_path'], self.config['simulation_name'],
                f'round_{round_num}', 'global_model'
            )
            os.makedirs(round_model_path, exist_ok=True)
            self.global_model.save_pretrained(round_model_path)
            print(f"New global model for round {round_num} saved to: {round_model_path}")
