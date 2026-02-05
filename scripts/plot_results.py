#!/usr/bin/env python3
"""
Script para plotar resultados comparativos FedAvg vs FedProx.
Gera gráficos para o artigo SBRC.

Uso:
    python scripts/plot_results.py --results_dir results/

Ou especificando experimentos:
    python scripts/plot_results.py \
        --fedprox results/EdgeRansomware_SmolLM135M_FedProx \
        --fedavg results/EdgeRansomware_SmolLM135M_FedAvg_Baseline
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Estilo para publicação
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Cores consistentes
COLORS = {
    'FedAvg': '#1f77b4',
    'FedProx': '#ff7f0e',
    'FedProx_0.001': '#2ca02c',
    'FedProx_0.01': '#ff7f0e',
    'FedProx_0.1': '#d62728',
}


def load_experiment_data(exp_dir: str) -> dict:
    """Carrega todos os CSVs de um diretório de experimento."""
    data = {}

    csv_files = [
        'f1_scores.csv',
        'f1_scores_antigo.csv',
        'temporal_metrics.csv',
        'communication_metrics.csv',
        'inference_benchmark.csv',
    ]

    for csv_file in csv_files:
        path = os.path.join(exp_dir, csv_file)
        if os.path.exists(path):
            data[csv_file.replace('.csv', '')] = pd.read_csv(path)
            print(f"  Loaded: {csv_file}")

    return data


def find_experiments(results_dir: str) -> list:
    """Encontra todos os experimentos no diretório de resultados."""
    experiments = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            f1_path = os.path.join(item_path, 'f1_scores.csv')
            if os.path.exists(f1_path):
                experiments.append(item_path)
    return sorted(experiments)


def plot_f1_convergence(dfs: list, labels: list, output_path: str, k: int = 1):
    """
    Plota curva de convergência do F1-Score por rodada.

    Args:
        dfs: Lista de DataFrames com resultados
        labels: Lista de labels para cada experimento
        output_path: Caminho para salvar o gráfico
        k: Valor de k para top-k accuracy
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for df, label in zip(dfs, labels):
        subset = df[df['k'] == k].copy()
        if subset.empty:
            continue

        subset = subset.sort_values('round')
        color = COLORS.get(label, None)

        ax.plot(
            subset['round'],
            subset['f1_score'],
            marker='o',
            markersize=4.5,
            linewidth=2.5,
            label=label,
            color=color,
        )

        # Annotate last + best points for readability
        try:
            last = subset.iloc[-1]
            best = subset.loc[subset['f1_score'].idxmax()]
            ax.scatter([best['round']], [best['f1_score']], marker='*', s=140, color=color, zorder=5)
            ax.annotate(
                f"{last['f1_score']:.3f}",
                (last['round'], last['f1_score']),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
                color=color or '#333333',
            )
        except Exception:
            pass

    ax.set_xlabel('Rodada de Treinamento')
    ax.set_ylabel('F1-Score')
    ax.set_title(f'Convergência do F1-Score (Top-{k})')
    ax.legend(loc='lower right', frameon=True, framealpha=0.95)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_metrics_comparison(dfs: list, labels: list, output_path: str, k: int = 1):
    """
    Plota comparação de métricas (F1, Precision, Recall) no último round.
    Para muitos experimentos, prefira os gráficos `final_f1_ranking.png` e `final_fpr_ranking.png`.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['f1_score', 'precision', 'recall']
    metric_labels = ['F1-Score', 'Precision', 'Recall']

    x = np.arange(len(metrics))
    width = 0.8 / len(dfs)

    for i, (df, label) in enumerate(zip(dfs, labels)):
        subset = df[df['k'] == k].copy()
        if subset.empty:
            continue

        # Pega último round
        last_round = subset['round'].max()
        final = subset[subset['round'] == last_round].iloc[0]

        values = [final[m] for m in metrics]
        color = COLORS.get(label, None)

        offset = (i - len(dfs) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color)

        # Adiciona valores nas barras
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}',
                ha='center',
                va='bottom',
                fontsize=9,
            )

    ax.set_ylabel('Score')
    ax.set_title(f'Comparação de Métricas - Round Final (Top-{k})')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc='upper right', frameon=True, framealpha=0.95)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def _extract_last_row(df: pd.DataFrame, k: int) -> pd.Series | None:
    subset = df[df['k'] == k].copy()
    if subset.empty:
        return None
    last_round = subset['round'].max()
    return subset[subset['round'] == last_round].iloc[0]


def plot_final_ranking(
    dfs: list,
    labels: list,
    output_path: str,
    *,
    k: int = 10,
    metric: str = "f1_score",
    title: str | None = None,
    ylabel: str | None = None,
    logy: bool = False,
):
    """
    Bar chart with experiments ranked by the chosen metric on the last round.
    Useful when comparing many experiments (e.g., sweeps).
    """
    rows = []
    for df, label in zip(dfs, labels):
        last = _extract_last_row(df, k)
        if last is None or metric not in last:
            continue
        rows.append((label, float(last[metric]), float(last.get("precision", np.nan)), float(last.get("recall", np.nan)), float(last.get("benign_fpr", np.nan))))

    if not rows:
        print(f"Sem dados para ranking ({metric})")
        return

    rows.sort(key=lambda x: x[1], reverse=True)
    labels_sorted = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [COLORS.get(l, '#333333') for l in labels_sorted]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels_sorted, values, color=colors)
    ax.set_title(title or f"Ranking (Round Final) — Top-{k} {metric}")
    ax.set_ylabel(ylabel or metric)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=25)

    if logy:
        ax.set_yscale("log")

    # annotate bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * (1.02 if not logy else 1.2),
            f"{val:.3f}" if metric != "benign_fpr" else f"{val:.4f}",
            ha='center',
            va='bottom',
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def save_summary_csv(dfs: list, labels: list, output_path: str, k: int = 10):
    """Saves a compact CSV summary (last + best) for quick inspection and paper tables."""
    rows = []
    for df, label in zip(dfs, labels):
        subset = df[df['k'] == k].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("round")
        last = subset.iloc[-1]
        # idxmax returns an index label (not positional), so use `.loc` (or reset_index).
        # Also guard against all-NaN columns.
        f1_series = pd.to_numeric(subset["f1_score"], errors="coerce")
        if f1_series.isna().all():
            continue
        best = subset.loc[f1_series.idxmax()]
        rows.append({
            "label": label,
            "k": k,
            "last_round": int(last["round"]),
            "last_f1": float(last["f1_score"]),
            "last_precision": float(last.get("precision", np.nan)),
            "last_recall": float(last.get("recall", np.nan)),
            "last_benign_fpr": float(last.get("benign_fpr", np.nan)),
            "last_threshold": float(last.get("threshold", np.nan)),
            "best_round": int(best["round"]),
            "best_f1": float(best["f1_score"]),
            "best_precision": float(best.get("precision", np.nan)),
            "best_recall": float(best.get("recall", np.nan)),
            "best_benign_fpr": float(best.get("benign_fpr", np.nan)),
        })

    if not rows:
        print("Sem dados para summary.csv")
        return

    out_df = pd.DataFrame(rows).sort_values(["last_f1"], ascending=False)
    out_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def plot_convergence_speed(dfs: list, labels: list, output_path: str, k: int = 1, threshold: float = 0.8):
    """
    Plota análise de velocidade de convergência.
    Mostra quantas rodadas para atingir threshold de F1.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    convergence_rounds = []

    for df, label in zip(dfs, labels):
        subset = df[df['k'] == k].sort_values('round')
        if subset.empty:
            convergence_rounds.append((label, None))
            continue

        # Encontra primeiro round que atinge threshold
        above_threshold = subset[subset['f1_score'] >= threshold]
        if not above_threshold.empty:
            first_round = above_threshold['round'].min()
        else:
            first_round = None

        convergence_rounds.append((label, first_round))

    # Plot
    valid_data = [(l, r) for l, r in convergence_rounds if r is not None]
    if not valid_data:
        print(f"Nenhum experimento atingiu F1 >= {threshold}")
        return

    labels_plot = [l for l, _ in valid_data]
    rounds_plot = [r for _, r in valid_data]
    colors = [COLORS.get(l, '#333333') for l in labels_plot]

    bars = ax.bar(labels_plot, rounds_plot, color=colors)

    for bar, val in zip(bars, rounds_plot):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{int(val)}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
        )

    ax.set_ylabel('Rodadas para Convergência')
    ax.set_title(f'Velocidade de Convergência (F1 >= {threshold})')
    ax.grid(True, alpha=0.3, axis='y')

    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_communication_cost(dfs: list, labels: list, output_path: str):
    """
    Plota custo de comunicação acumulado por rodada.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for df, label in zip(dfs, labels):
        if df.empty:
            continue

        df = df.sort_values('round')
        cumulative_mb = df['bytes_total'].cumsum() / (1024 * 1024)
        color = COLORS.get(label, None)

        ax.plot(
            df['round'],
            cumulative_mb,
            marker='s',
            markersize=4,
            linewidth=2,
            label=label,
            color=color,
        )

    ax.set_xlabel('Rodada')
    ax.set_ylabel('Comunicação Acumulada (MB)')
    ax.set_title('Custo de Comunicação')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_temporal_metrics(dfs: list, labels: list, output_path: str, k: int = 1):
    """
    Plota métricas temporais (TTD, Coverage) no último round.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # TTD
    ax1 = axes[0]
    ttd_data = []
    for df, label in zip(dfs, labels):
        if df.empty:
            continue
        subset = df[df['k'] == k]
        if subset.empty:
            continue
        last_round = subset['round'].max()
        final = subset[subset['round'] == last_round].iloc[0]
        ttd_data.append((label, final.get('mean_ttd_seconds', np.nan)))

    if ttd_data:
        labels_ttd = [l for l, _ in ttd_data]
        values_ttd = [v for _, v in ttd_data]
        colors = [COLORS.get(l, '#333333') for l in labels_ttd]

        bars = ax1.bar(labels_ttd, values_ttd, color=colors)
        for bar, val in zip(bars, values_ttd):
            if not np.isnan(val):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}s', ha='center', va='bottom', fontsize=10)

        ax1.set_ylabel('Tempo (segundos)')
        ax1.set_title('Time-to-Detection (TTD) Médio')
        ax1.grid(True, alpha=0.3, axis='y')

    # Detection Coverage
    ax2 = axes[1]
    coverage_data = []
    for df, label in zip(dfs, labels):
        if df.empty:
            continue
        subset = df[df['k'] == k]
        if subset.empty:
            continue
        last_round = subset['round'].max()
        final = subset[subset['round'] == last_round].iloc[0]
        coverage_data.append((label, final.get('detection_coverage', np.nan) * 100))

    if coverage_data:
        labels_cov = [l for l, _ in coverage_data]
        values_cov = [v for _, v in coverage_data]
        colors = [COLORS.get(l, '#333333') for l in labels_cov]

        bars = ax2.bar(labels_cov, values_cov, color=colors)
        for bar, val in zip(bars, values_cov):
            if not np.isnan(val):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

        ax2.set_ylabel('Cobertura (%)')
        ax2.set_title('Cobertura de Detecção')
        ax2.set_ylim(0, 110)
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_all_k_comparison(dfs: list, labels: list, output_path: str):
    """
    Plota F1-Score para todos os valores de K no último round.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for df, label in zip(dfs, labels):
        if df.empty:
            continue

        last_round = df['round'].max()
        subset = df[df['round'] == last_round].sort_values('k')

        color = COLORS.get(label, None)
        ax.plot(
            subset['k'],
            subset['f1_score'],
            marker='o',
            markersize=6,
            linewidth=2,
            label=label,
            color=color,
        )

    ax.set_xlabel('K (Top-K)')
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score por Top-K (Round Final)')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def generate_latex_table(dfs: list, labels: list, output_path: str, k: int = 1):
    """
    Gera tabela LaTeX com resultados para o artigo.
    """
    rows = []

    for df, label in zip(dfs, labels):
        if df.empty:
            continue

        subset = df[df['k'] == k]
        if subset.empty:
            continue

        last_round = subset['round'].max()
        final = subset[subset['round'] == last_round].iloc[0]

        rows.append({
            'Método': label,
            'F1-Score': f"{final['f1_score']:.4f}",
            'Precision': f"{final['precision']:.4f}",
            'Recall': f"{final['recall']:.4f}",
            'FPR Benigno': f"{final.get('benign_fpr', 0):.4f}",
        })

    if not rows:
        print("Sem dados para gerar tabela")
        return

    table_df = pd.DataFrame(rows)

    # Gera LaTeX
    latex = table_df.to_latex(index=False, escape=False)

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"Saved: {output_path}")

    # Também salva como CSV
    csv_path = output_path.replace('.tex', '.csv')
    table_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot resultados FedAvg vs FedProx')
    parser.add_argument('--results_dir', type=str, default='results/',
                        help='Diretório base com resultados')
    parser.add_argument('--fedprox', type=str, default=None,
                        help='Diretório específico do experimento FedProx')
    parser.add_argument('--fedavg', type=str, default=None,
                        help='Diretório específico do experimento FedAvg')
    parser.add_argument('--fedprox_label', type=str, default=None,
                        help='Label customizada para o experimento passado em --fedprox')
    parser.add_argument('--fedavg_label', type=str, default=None,
                        help='Label customizada para o experimento passado em --fedavg')
    parser.add_argument('--output_dir', type=str, default='img/',
                        help='Diretório para salvar gráficos')
    parser.add_argument('--k', type=int, default=1,
                        help='Valor de K para análises Top-K')
    parser.add_argument('--title', type=str, default=None,
                        help='Título opcional para os gráficos principais')

    args = parser.parse_args()

    # Pequena higiene para evitar erro ao colar comandos com vírgula no final.
    # Ex.: "--output_dir img/test/,".
    for attr in ("results_dir", "fedprox", "fedavg", "fedprox_label", "fedavg_label", "output_dir"):
        val = getattr(args, attr, None)
        if isinstance(val, str):
            setattr(args, attr, val.rstrip(","))

    # Cria diretório de saída
    os.makedirs(args.output_dir, exist_ok=True)

    # Carrega dados
    experiments = []
    labels = []

    if args.fedprox and args.fedavg:
        # Usa experimentos especificados
        for path, default_label, override_label in [
            (args.fedavg, 'FedAvg', args.fedavg_label),
            (args.fedprox, 'FedProx', args.fedprox_label),
        ]:
            if os.path.exists(path):
                print(f"\nLoading: {path}")
                data = load_experiment_data(path)
                if 'f1_scores' in data:
                    experiments.append(data)
                    if override_label:
                        label = override_label
                    else:
                        # Tenta pegar label do CSV; se duplicar, usa o basename do diretório.
                        if 'aggregation_method' in data['f1_scores'].columns:
                            label = data['f1_scores']['aggregation_method'].iloc[0]
                        else:
                            label = default_label
                        if label in labels:
                            label = os.path.basename(path)
                    labels.append(label)
    else:
        # Auto-detecta experimentos
        print(f"Buscando experimentos em: {args.results_dir}")
        exp_dirs = find_experiments(args.results_dir)

        for exp_dir in exp_dirs:
            print(f"\nLoading: {exp_dir}")
            data = load_experiment_data(exp_dir)
            if 'f1_scores' in data:
                experiments.append(data)
                # Tenta pegar label do CSV
                if 'aggregation_method' in data['f1_scores'].columns:
                    label = data['f1_scores']['aggregation_method'].iloc[0]
                    mu = data['f1_scores'].get('fedprox_mu', pd.Series([0])).iloc[0]
                    if mu > 0:
                        label = f"FedProx_{mu}"
                else:
                    label = os.path.basename(exp_dir)
                # Se vários experimentos tiverem o mesmo método (ex.: todos "FedAvg"),
                # desambigua pelo nome da pasta para evitar legenda duplicada.
                if label in labels:
                    label = os.path.basename(exp_dir)
                labels.append(label)

    if not experiments:
        print("Nenhum experimento encontrado!")
        return

    print(f"\n{'='*50}")
    print(f"Experimentos carregados: {labels}")
    print(f"{'='*50}\n")

    # Extrai DataFrames de f1_scores
    f1_dfs = [exp.get('f1_scores', pd.DataFrame()) for exp in experiments]
    comm_dfs = [exp.get('communication_metrics', pd.DataFrame()) for exp in experiments]
    temporal_dfs = [exp.get('temporal_metrics', pd.DataFrame()) for exp in experiments]

    # Gera gráficos
    print("Gerando gráficos...\n")

    # 1. Convergência F1
    plot_f1_convergence(
        f1_dfs, labels,
        os.path.join(args.output_dir, 'f1_convergence.png'),
        k=args.k
    )

    # 2. Comparação de métricas
    plot_metrics_comparison(
        f1_dfs, labels,
        os.path.join(args.output_dir, 'metrics_comparison.png'),
        k=args.k
    )

    # Rankings (melhor para muitos experimentos)
    plot_final_ranking(
        f1_dfs, labels,
        os.path.join(args.output_dir, 'final_f1_ranking.png'),
        k=args.k,
        metric="f1_score",
        title=args.title or f"Ranking de F1 (Round Final) — Top-{args.k}",
        ylabel="F1 (round final)",
    )
    plot_final_ranking(
        f1_dfs, labels,
        os.path.join(args.output_dir, 'final_fpr_ranking.png'),
        k=args.k,
        metric="benign_fpr",
        title=args.title or f"Ranking de FPR Benigno (Round Final) — Top-{args.k}",
        ylabel="Benign FPR (round final)",
        logy=True,
    )

    save_summary_csv(
        f1_dfs, labels,
        os.path.join(args.output_dir, 'summary.csv'),
        k=args.k,
    )

    # 3. Velocidade de convergência
    plot_convergence_speed(
        f1_dfs, labels,
        os.path.join(args.output_dir, 'convergence_speed.png'),
        k=args.k,
        threshold=0.8
    )

    # 4. F1 por Top-K
    plot_all_k_comparison(
        f1_dfs, labels,
        os.path.join(args.output_dir, 'f1_by_topk.png')
    )

    # 5. Custo de comunicação
    if any(not df.empty for df in comm_dfs):
        plot_communication_cost(
            comm_dfs, labels,
            os.path.join(args.output_dir, 'communication_cost.png')
        )

    # 6. Métricas temporais
    if any(not df.empty for df in temporal_dfs):
        plot_temporal_metrics(
            temporal_dfs, labels,
            os.path.join(args.output_dir, 'temporal_metrics.png'),
            k=args.k
        )

    # 7. Tabela LaTeX
    generate_latex_table(
        f1_dfs, labels,
        os.path.join(args.output_dir, 'results_table.tex'),
        k=args.k
    )

    print(f"\nTodos os gráficos salvos em: {args.output_dir}")


if __name__ == '__main__':
    main()
