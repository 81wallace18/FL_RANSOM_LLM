import os
import random
import math
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

    def _split_data_for_clients(self):
        """
        Splits the main tokenized dataset into shards for each client.
        This is a one-time setup operation.
        """
        client_data_base_path = os.path.join(
            self.config['results_path'],
            self.config['simulation_name'],
            'client_data'
        )
        
        # Avoid re-splitting if already done
        if os.path.exists(client_data_base_path):
            print("Client data shards already exist. Skipping split.")
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
        random.shuffle(indices) # Shuffle for IID distribution

        for i in range(num_clients):
            client_indices = indices[i::num_clients]
            client_dataset = dataset.select(client_indices)
            client_dataset.save_to_disk(os.path.join(client_data_base_path, f'client_{i}'))
        
        print(f"Data successfully split for {num_clients} clients.")

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

    def _select_clients(self, round_num):
        """
        Selects clients for training using adaptive strategies.
        Uses different strategies based on the round to improve convergence.
        """
        num_clients = self.config['num_clients']
        num_selected = int(num_clients * self.config['client_frac'])

        # Round 1: Random selection for diversity
        if round_num == 1:
            selected = random.sample(range(num_clients), num_selected)
            print(f"Round {round_num}: Random selection of {num_selected} clients")
            return selected

        # Early rounds (2-10): Stratified sampling based on previous performance
        elif round_num <= 10:
            # Track client performance if not already done
            if not hasattr(self, 'client_performance'):
                self.client_performance = {i: [] for i in range(num_clients)}
                self.client_diversity = {i: set() for i in range(num_clients)}

            # Select clients with mixed strategy
            selected = []

            # 50% clients with lowest recent performance (need more training)
            if round_num > 2:
                avg_performance = []
                for client_id in range(num_clients):
                    if self.client_performance[client_id]:
                        avg_perf = np.mean(self.client_performance[client_id][-3:])  # Last 3 rounds
                        avg_performance.append((client_id, avg_perf))
                    else:
                        avg_performance.append((client_id, 0.0))  # New clients get priority

                # Sort by performance (lower is better - more room for improvement)
                avg_performance.sort(key=lambda x: x[1])
                num_low_perf = num_selected // 2
                selected.extend([cid for cid, _ in avg_performance[:num_low_perf]])

            # 50% random for diversity
            remaining = [i for i in range(num_clients) if i not in selected]
            num_random = num_selected - len(selected)
            if num_random > 0:
                selected.extend(random.sample(remaining, num_random))

            print(f"Round {round_num}: Strategic selection ({num_selected} clients)")
            return selected

        # Middle rounds (11-30): Focus on diversity and performance
        elif round_num <= 30:
            selected = []

            # Select clients from different performance quartiles
            if hasattr(self, 'client_performance'):
                performance_scores = []
                for client_id in range(num_clients):
                    if self.client_performance[client_id]:
                        scores = self.client_performance[client_id]
                        recent_scores = scores[-5:] if len(scores) >= 5 else scores
                        perf = np.mean(recent_scores)
                        variability = np.std(recent_scores) if len(recent_scores) > 1 else 0
                        performance_scores.append((client_id, perf, variability))
                    else:
                        performance_scores.append((client_id, 0.0, 0.0))

                # Sort by performance
                performance_scores.sort(key=lambda x: x[1])
                total = len(performance_scores)

                # Select from each quartile
                quartile_size = total // 4
                for i in range(0, total, quartile_size):
                    quartile = performance_scores[i:i+quartile_size]
                    num_from_quartile = max(1, num_selected // 4)
                    # Prefer clients with higher variability (more potential for improvement)
                    quartile.sort(key=lambda x: x[2], reverse=True)
                    selected.extend([cid for cid, _, _ in quartile[:num_from_quartile]])
            else:
                selected = random.sample(range(num_clients), num_selected)

            # Ensure we have exactly num_selected clients
            if len(selected) > num_selected:
                selected = selected[:num_selected]
            elif len(selected) < num_selected:
                remaining = [i for i in range(num_clients) if i not in selected]
                selected.extend(random.sample(remaining, num_selected - len(selected)))

            print(f"Round {round_num}: Diverse selection ({num_selected} clients)")
            return selected

        # Late rounds (31+): Focus on best performing clients
        else:
            selected = []

            if hasattr(self, 'client_performance'):
                # Select top performing clients
                performance_scores = []
                for client_id in range(num_clients):
                    if self.client_performance[client_id]:
                        # Weight recent performance more heavily
                        recent_scores = self.client_performance[client_id][-10:]
                        if len(recent_scores) >= 3:
                            # Exponential weighting for recent rounds
                            weights = np.exp(np.linspace(-1, 0, len(recent_scores)))
                            weighted_perf = np.average(recent_scores, weights=weights)
                        else:
                            weighted_perf = np.mean(recent_scores)
                        performance_scores.append((client_id, weighted_perf))
                    else:
                        performance_scores.append((client_id, 0.0))

                # Sort by weighted performance (higher is better for convergence)
                performance_scores.sort(key=lambda x: x[1], reverse=True)

                # Select top 70% best performers
                num_top = int(num_selected * 0.7)
                selected.extend([cid for cid, _ in performance_scores[:num_top]])

                # Add 30% random for exploration
                remaining = [i for i in range(num_clients) if i not in selected]
                num_random = num_selected - len(selected)
                if num_random > 0:
                    selected.extend(random.sample(remaining, num_random))
            else:
                selected = random.sample(range(num_clients), num_selected)

            print(f"Round {round_num}: Performance-focused selection ({num_selected} clients)")
            return selected

    def _estimate_client_loss(self, client_weights_list):
        """
        Estimates average client loss based on weight updates.
        Used for adaptive client selection.
        """
        if not client_weights_list:
            return 1.0

        # Calculate L2 norm of weight updates as proxy for loss
        total_norm = 0
        total_params = 0

        for client_weights in client_weights_list:
            for param_name, param_tensor in client_weights.items():
                if "lora_" in param_name:  # Only consider LoRA parameters
                    norm = torch.norm(param_tensor).item()
                    total_norm += norm
                    total_params += param_tensor.numel()

        avg_norm = total_norm / max(total_params / 1000, 1)  # Normalize per 1000 parameters
        return min(max(avg_norm, 0.1), 2.0)  # Clamp between 0.1 and 2.0

    def _aggregate_models(self, client_weights_list):
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

        for key in weight_keys:
            # Stack all tensors for the current key from all clients and average them.
            # The tensors are on the CPU, so this uses RAM, not VRAM.
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

            selected_clients_ids = self._select_clients(round_num)
            print(f"Clients selected for this round: {selected_clients_ids}")

            client_weights_list = []
            current_lr = self._get_learning_rate(round_num)
            for client_id in selected_clients_ids:
                client_trainer = ClientTrainer(client_id, self.config)
                cpu_weights = client_trainer.train(round_num, current_lr)
                if cpu_weights:
                    client_weights_list.append(cpu_weights)
            
            print("Aggregating client models...")
            self._aggregate_models(client_weights_list)

            # Evaluate client performance for adaptive selection
            if hasattr(self, 'client_performance'):
                # Simulate client performance based on convergence
                avg_loss = self._estimate_client_loss(client_weights_list)
                for client_id in selected_clients_ids:
                    # Add some randomness to simulate real-world variation
                    performance = avg_loss + np.random.normal(0, 0.05)
                    self.client_performance[client_id].append(performance)
                    # Keep only last 10 rounds of performance
                    if len(self.client_performance[client_id]) > 10:
                        self.client_performance[client_id] = self.client_performance[client_id][-10:]

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

            selected_clients_ids = self._select_clients(round_num)
            print(f"Clients selected for this round: {selected_clients_ids}")

            client_weights_list = []
            current_lr = self._get_learning_rate(round_num)

            # Mover o modelo global para a CPU para liberar VRAM para os clientes
            self.global_model.to('cpu')
            torch.cuda.empty_cache()

            client_chunks = list(chunks(selected_clients_ids, num_gpus))
            for i, client_chunk in enumerate(client_chunks):
                print(f"  --- Processing client batch {i+1}/{len(client_chunks)} ---")
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
                    tasks = []
                    for j, client_id in enumerate(client_chunk):
                        gpu_id = j % num_gpus
                        args = (client_id, self.config, round_num, current_lr, gpu_id)
                        tasks.append(executor.submit(train_client_process, args))

                    for future in concurrent.futures.as_completed(tasks):
                        try:
                            cpu_weights = future.result()
                            if cpu_weights:
                                client_weights_list.append(cpu_weights)
                        except Exception as e:
                            print(f"Erro ao treinar cliente: {e}")
            
            # Mover o modelo de volta para a GPU para agregação
            self.global_model.to(self.device)

            print("Aggregating client models...")
            self._aggregate_models(client_weights_list)

            # Evaluate client performance for adaptive selection
            if hasattr(self, 'client_performance'):
                # Simulate client performance based on convergence
                avg_loss = self._estimate_client_loss(client_weights_list)
                for client_id in selected_clients_ids:
                    # Add some randomness to simulate real-world variation
                    performance = avg_loss + np.random.normal(0, 0.05)
                    self.client_performance[client_id].append(performance)
                    # Keep only last 10 rounds of performance
                    if len(self.client_performance[client_id]) > 10:
                        self.client_performance[client_id] = self.client_performance[client_id][-10:]

            round_model_path = os.path.join(
                self.config['results_path'], self.config['simulation_name'],
                f'round_{round_num}', 'global_model'
            )
            os.makedirs(round_model_path, exist_ok=True)
            self.global_model.save_pretrained(round_model_path)
            print(f"New global model for round {round_num} saved to: {round_model_path}")
