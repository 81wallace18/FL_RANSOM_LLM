import os
import random
import math
import torch
from datasets import load_from_disk

from src.models.model_loader import initialize_global_model
from .client import ClientTrainer

class FederatedServer:
    """
    Orchestrates the federated learning process, including client selection,
    training, and model aggregation.
    """
    def __init__(self, config):
        self.config = config
        self.global_model, self.tokenizer = initialize_global_model(config)

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
                param.data.copy_(aggregated_adapters[name])

    def _aggregate_models(self, client_models):
        """Aggregates client models using Federated Averaging (FedAvg)."""
        if self.config['lora']:
            all_adapters = [self._get_adapters(model) for model in client_models]
            aggregated_adapters = {}
            
            for key in all_adapters[0].keys():
                aggregated_adapters[key] = torch.mean(
                    torch.stack([adapters[key] for adapters in all_adapters]), dim=0
                )
            
            self._set_adapters(self.global_model, aggregated_adapters)
        else:
            # Aggregate the entire model (full fine-tuning)
            global_state_dict = self.global_model.state_dict()
            for key in global_state_dict.keys():
                global_state_dict[key] = torch.mean(
                    torch.stack([model.state_dict()[key].data for model in client_models]), dim=0
                )
            self.global_model.load_state_dict(global_state_dict)

    def run_federated_training(self):
        """The main federated training loop."""
        # 1. Setup: Split data for clients
        self._split_data_for_clients()

        # 2. Training Loop
        for round_num in range(1, self.config['num_rounds'] + 1):
            print(f"\n===== Starting Round {round_num}/{self.config['num_rounds']} =====")
            
            # Select clients for this round
            num_selected_clients = int(self.config['num_clients'] * self.config['client_frac'])
            selected_clients_ids = random.sample(range(self.config['num_clients']), num_selected_clients)
            print(f"Clients selected for this round: {selected_clients_ids}")

            # Train clients in parallel (conceptually)
            client_models = []
            current_lr = self._get_learning_rate(round_num)
            for client_id in selected_clients_ids:
                client_trainer = ClientTrainer(client_id, self.config)
                client_model = client_trainer.train(round_num, current_lr)
                client_models.append(client_model)
            
            # Aggregate models
            print("Aggregating client models...")
            self._aggregate_models(client_models)
            
            # Save new global model
            round_model_path = os.path.join(
                self.config['results_path'],
                self.config['simulation_name'],
                f'round_{round_num}',
                'global_model'
            )
            os.makedirs(round_model_path, exist_ok=True)
            self.global_model.save_pretrained(round_model_path)
            print(f"New global model for round {round_num} saved to: {round_model_path}")
