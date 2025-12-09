#!/usr/bin/env python3
"""
Exemplo de como usar o sistema aprimorado de detecção de ransomware.

Este script demonstra como executar o pipeline completo de treinamento
federado com todas as melhorias implementadas para aumentar o F1 score.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import main
from src.data_processing.ransomlog_processor import RansomLogProcessor
from src.evaluation.evaluator import Evaluator
import yaml

def demonstrate_improvements():
    """
    Demonstra as melhorias implementadas no sistema.
    """
    print("=" * 60)
    print("MELHORIAS IMPLEMENTADAS PARA AUMENTAR O F1 SCORE")
    print("=" * 60)

    print("\n1. DADOS REAIS E PROCESSAMENTO INTELIGENTE:")
    print("   - Suporte para múltiplos formatos de log (CSV, JSON, .log)")
    print("   - Criação de sessões por ID de processo ou janela de tempo")
    print("   - Sintetização enhanced quando dados reais não disponíveis")
    print("   - Sanitização inteligente que preserva estrutura relevante")

    print("\n2. OTIMIZAÇÃO DE TOKENIZAÇÃO:")
    print("   - Comprimento dinâmico baseado na distribuição dos dados")
    print("   - Análise de percentis para evitar padding excessivo")
    print("   - Preservação de metadados para análise")

    print("\n3. HIPERPARÂMETROS MELHORADOS:")
    print("   - max_steps: 10 → 50 (mais treinamento)")
    print("   - batch_size: 32 → 16 (melhor generalização)")
    print("   - gradient_accumulation_steps: 4 (batch size efetivo de 64)")
    print("   - learning_rate: 0.001 → 0.0001 (treinamento mais estável)")
    print("   - lora_rank: 8 → 16 (maior capacidade)")
    print("   - lora_alpha_multiplier: 2 → 4 (melhor estabilidade)")

    print("\n4. BALANCEAMENTO ESTRATIFICADO:")
    print("   - Detecção automática de desbalanceamento")
    print("   - Amostragem balanceada quando razão > 3:1")
    print("   - Preservação de distribuição original para referência")

    print("\n5. THRESHOLD ADAPTATIVO:")
    print("   - Busca em duas fases (grossa e fina)")
    print("   - Validação cruzada estratificada")
    print("   - Seleção robusta de threshold ótimo")

    print("\n6. SELEÇÃO INTELIGENTE DE CLIENTES:")
    print("   - Round 1: Aleatório para diversidade")
    print("   - Rounds 2-10: Foco em clientes com baixo desempenho")
    print("   - Rounds 11-30: Seleção diversificada por quartis")
    print("   - Rounds 31+: Foco nos melhores performers")

    print("\n7. MÉTRICAS ADICIONAIS:")
    print("   - Accuracy e AUC além de Precision/Recall/F1")
    print("   - Controle de zero_division para evitar divisão por zero")
    print("   - Tracking de performance dos clientes")

    print("\n8. OTIMIZAÇÕES DE MEMÓRIA:")
    print("   - Processamento em batches menores")
    print("   - Limpeza explícita de VRAM")
    print("   - Mapeamento explícito de GPU para paralelismo")

def run_improved_training():
    """
    Executa o treinamento com as novas configurações.
    """
    print("\n" + "=" * 60)
    print("INICIANDO TREINAMENTO MELHORADO")
    print("=" * 60)

    # Verificar configuração
    config_path = "configs/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("\nConfiguração atual:")
    print(f"  Modelo: {config['model_name']}")
    print(f"  LoRA rank: {config['lora_rank']}")
    print(f"  Max steps: {config['max_steps']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Gradient accumulation: {config.get('gradient_accumulation_steps', 1)}")
    print(f"  Learning rate: {config['initial_lr']}")

    # Executar treinamento principal
    try:
        print("\nIniciando pipeline federado...")
        main()
        print("\nTreinamento concluído com sucesso!")
    except Exception as e:
        print(f"\nErro durante treinamento: {e}")
        print("Verifique se os dados estão corretamente configurados.")

def analyze_results():
    """
    Analisa os resultados obtidos.
    """
    print("\n" + "=" * 60)
    print("ANÁLISE DOS RESULTADOS")
    print("=" * 60)

    config_path = "configs/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    results_path = os.path.join(config['results_path'], config['simulation_name'])
    f1_scores_path = os.path.join(results_path, 'f1_scores.csv')

    if os.path.exists(f1_scores_path):
        import pandas as pd
        df = pd.read_csv(f1_scores_path)

        print("\nEstatísticas de F1 Score por round:")
        for round_num in df['round'].unique():
            round_data = df[df['round'] == round_num]
            best_f1 = round_data['f1_score'].max()
            best_k = round_data.loc[round_data['f1_score'].idxmax(), 'k']
            print(f"  Round {round_num}: Melhor F1={best_f1:.4f} (k={best_k})")

        print(f"\nMelhor F1 Score geral: {df['f1_score'].max():.4f}")
        print(f"Média F1 Score: {df['f1_score'].mean():.4f}")
        print(f"Desvio padrão: {df['f1_score'].std():.4f}")

        # Mostrar tendências
        best_per_round = df.groupby('round')['f1_score'].max()
        if len(best_per_round) > 1:
            improvement = best_per_round.iloc[-1] - best_per_round.iloc[0]
            print(f"\nMelhoria do primeiro ao último round: {improvement:+.4f}")
    else:
        print("\nResultados não encontrados. Execute o treinamento primeiro.")

if __name__ == "__main__":
    print("SISTEMA MELHORADO DE DETECÇÃO DE RANSOMWARE")
    print("============================================")

    while True:
        print("\nOpções:")
        print("1. Demonstrar melhorias implementadas")
        print("2. Executar treinamento melhorado")
        print("3. Analisar resultados existentes")
        print("4. Sair")

        choice = input("\nEscolha uma opção (1-4): ").strip()

        if choice == '1':
            demonstrate_improvements()
        elif choice == '2':
            run_improved_training()
        elif choice == '3':
            analyze_results()
        elif choice == '4':
            print("Saindo...")
            break
        else:
            print("Opção inválida. Tente novamente.")