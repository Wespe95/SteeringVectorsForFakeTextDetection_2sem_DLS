"""
Visalization Functions for computed steering vectors
This module contains functions for visualizing like Magnitude Analysis
    cosine similarity between sequential layers and heat-Map components
    UMAP, T-SNE and PCA, also projection analysis
"""

import os
import torch
import matplotlib.pyplot as plt
from typing import Dict
import numpy as np
import pandas as pd
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.spatial.distance import pdist
import pickle
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def plot_steering_magnitudes(steering_vectors: Dict[str, torch.Tensor],
                           save_plots: bool, plot_dir: str) -> None:
    """Визуализирует величины.
    Steering Vector Magnitude Analysis (https://arxiv.org/abs/2407.12404):
    согласно этой статье бóльшие величины в определенных слоях показывают, что эти слои активнее реагируют на тестируемый концепт
    """

    fig = plt.figure(figsize=(12, 8))

    # Layer-wise magnitudes
    layer_keys = [k for k in steering_vectors.keys() if k.startswith('layer_')] # and not k.endswith('_normalized')
    layer_nums = [int(k.split('_')[1]) for k in layer_keys]
    magnitudes = [torch.norm(steering_vectors[k]).item() for k in layer_keys]

    plt.plot(layer_nums, magnitudes, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Layer Index')
    plt.ylabel('Steering Vector Magnitude')
    plt.title('Steering Vector Magnitude by Layer')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(plot_dir, 'steering_magnitudes.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_layerwise_analysis(steering_vectors: Dict[str, torch.Tensor],
                          save_plots: bool, plot_dir: str,
                          target_dtype: torch.dtype) -> None:
    """
    Анализ векторов по слоям:
    - Косинусное сходство между последовательными слоя
    - heat-Map компонентов: визуализация первых 50 измерений общего вектора (среднее всех слоев)
    """

    layer_keys = [k for k in steering_vectors.keys() if k.startswith('layer_') and not k.startswith('overall')]
    if len(layer_keys) < 2:
        return

    # Calculate layer-to-layer similarities
    layer_nums = sorted([int(k.split('_')[1]) for k in layer_keys])
    similarities = []

    for i in range(len(layer_nums) - 1):
        vec1 = steering_vectors[f'layer_{layer_nums[i]}']
        vec2 = steering_vectors[f'layer_{layer_nums[i+1]}']

        # Cosine similarity
        cos_sim = torch.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
        similarities.append(cos_sim)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Layer similarities
    ax1.plot(layer_nums[:-1], similarities, 'g-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Cosine Similarity with Next Layer')
    ax1.set_title('Layer-to-Layer Steering Vector Similarity')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Heatmap of steering vector components (sample)
    if 'overall' in steering_vectors:
        overall_vec = steering_vectors['overall'].cpu().to(target_dtype).numpy()
        # Show first 50 dimensions for readability
        n_dims = min(50, len(overall_vec))

        im = ax2.imshow(overall_vec[:n_dims].reshape(1, -1), cmap='RdBu_r', aspect='auto')
        ax2.set_xlabel('Dimension')
        ax2.set_title(f'Overall Steering Vector Components (First {n_dims} dims)')
        ax2.set_yticks([])
        plt.colorbar(im, ax=ax2)

    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(plot_dir, 'layerwise_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_activation_spaces(activations: Dict[str, Dict[int, torch.Tensor]],
                         save_plots: bool, plot_dir: str, target_dtype: torch.dtype, random_state: int = 42) -> None:
    """Визуализирует пространства активации с помощью UMAP, t-SNE и PCA."""

    # Select a representative layer (middle layer)
    sample_key = list(activations.keys())[0]
    available_layers = list(activations[sample_key].keys())
    #middle_layer = available_layers[len(available_layers) // 2]

    #print(f"Visualizing activation space for layer {middle_layer}")

    color_map = {'train_real': 'blue', 'train_fake': 'red', 'dev_real': 'lightblue', 'dev_fake': 'lightcoral'}

    for layer_idx in available_layers:
        #print(F"""Visualizing activation space for layer {layer_idx}""")

        # Collect data for visualization
        all_activations = []
        labels = []
        colors = []


        for data_type, acts in activations.items():
            if layer_idx in acts:
                layer_acts = acts[layer_idx].cpu().to(target_dtype).numpy()
                all_activations.append(layer_acts)
                labels.extend([data_type] * len(layer_acts))
                colors.extend([color_map[data_type]] * len(layer_acts))

        if not all_activations:
            print("No activations found for visualization")
            return

        # Combine all activations
        X = np.vstack(all_activations)
        #print(f"Total samples for visualization: {X.shape[0]}, Dimensions: {X.shape[1]}")

        # Subsample if too many points (for performance)
        if X.shape[0] > 2000:
            indices = np.random.choice(X.shape[0], 2000, replace=False)
            X = X[indices]
            labels = [labels[i] for i in indices]
            colors = [colors[i] for i in indices]
            #print(f"Subsampled to {len(X)} points for visualization")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Activation Space - Layer {layer_idx}', fontsize=16)

        # PCA
        #print(f"Computing PCA for layer {layer_idx}")
        pca = PCA(n_components=2, random_state=random_state)
        X_pca = pca.fit_transform(X)

        for data_type in set(labels):
            mask = [l == data_type for l in labels]
            axes[0, 0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                            c=color_map[data_type], label=data_type, alpha=0.6, s=20)

        axes[0, 0].set_title(f'PCA - Layer {layer_idx}\nExplained Variance: {pca.explained_variance_ratio_.sum():.3f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # t-SNE
        #print(f"Computing t-SNE for layer {layer_idx}")
        tsne = TSNE(n_components=2, random_state=random_state, perplexity=min(30, X.shape[0]//4))
        X_tsne = tsne.fit_transform(X)

        for data_type in set(labels):
            mask = [l == data_type for l in labels]
            axes[0, 1].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                            c=color_map[data_type], label=data_type, alpha=0.6, s=20)

        axes[0, 1].set_title(f't-SNE - Layer {layer_idx}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # UMAP
        #print(f"Computing UMAP for layer {layer_idx}")
        reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=min(15, X.shape[0]//4))
        X_umap = reducer.fit_transform(X)

        for data_type in set(labels):
            mask = [l == data_type for l in labels]
            axes[1, 0].scatter(X_umap[mask, 0], X_umap[mask, 1],
                            c=color_map[data_type], label=data_type, alpha=0.6, s=20)

        axes[1, 0].set_title(f'UMAP - Layer {layer_idx}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Distance distributions
        train_real_mask = [l == 'train_real' for l in labels]
        train_fake_mask = [l == 'train_fake' for l in labels]

        if any(train_real_mask) and any(train_fake_mask):
            real_points = X_umap[train_real_mask]
            fake_points = X_umap[train_fake_mask]

            # Calculate pairwise distances within each group
            real_distances = pdist(real_points)
            fake_distances = pdist(fake_points)

            axes[1, 1].hist(real_distances, bins=30, alpha=0.6, label='Real-Real distances', color='blue')
            axes[1, 1].hist(fake_distances, bins=30, alpha=0.6, label='Fake-Fake distances', color='red')
            axes[1, 1].set_xlabel('Distance')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Intra-class Distance Distributions')
            axes[1, 1].legend()

        plt.tight_layout()
        if save_plots:
            filename = f'activation_space_layer_{layer_idx}.png'
            filepath = os.path.join(plot_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            #print(f"Saved plot for layer {layer_idx} to {filepath}")
    plt.show()

    plt.close(fig)
    print(f"Computed activation space for all {len(available_layers)} layers")


def plot_activation_spaces_combined(activations: Dict[str, Dict[int, torch.Tensor]], 
                                    save_plots: bool, plot_dir: str, target_dtype: torch.dtype, random_state: int = 42) -> None:
    """
    Визуализация активаций всех слоев в одном окне.
    """

    sample_key = list(activations.keys())[0]
    available_layers = list(activations[sample_key].keys())

    print(F"Creating combined visualization for {len(available_layers)} layers")

    # Grid layout
    n_layers = len(available_layers)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    color_map = {'train_real': 'blue', 'train_fake': 'red', 'dev_real': 'lightblue', 'dev_fake': 'lightcoral'}

    for method in ['PCA', 't-SNE', 'UMAP']:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        fig.suptitle(f'{method} visualization - all layers', fontsize=16)

        if n_layers == 1:
            axes = [axes]
        elif n_rows ==1:
            axes = axes.reshape(1, -1)

        for idx, layer_idx in enumerate(available_layers):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]

            all_activations = []
            labels = []

            for data_type, acts in activations.items():
                if layer_idx in acts:
                    layer_acts = acts[layer_idx].cpu().to(target_dtype).numpy()
                    all_activations.append(layer_acts)
                    labels.extend([data_type] * len(layer_acts))
            
            if not all_activations:
                continue

            X=np.vstack(all_activations)
            if X.shape[0] > 1000:
                indices = np.random.choice(X.shape[0], size=1000, replace=False)
                X = X[indices]
                labels = [labels[i] for i in indices]

            if method == 'PCA':
                reduced = PCA(n_components=2, random_state=random_state)
                X_reduced = reduced.fit_transform(X)
                variance_ratio = F"\Variance ratio: {reduced.explained_variance_ratio_.sum():.3f}"
            elif method == 't-SNE':
                reduced = TSNE(n_components=2, random_state=random_state, perplexity= min(30, X.shape[0]//4))
                X_reduced = reduced.fit_transform(X)
                variance_ratio = ""
            elif method == 'UMAP':
                reduced = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=min(15, X.shape[0]//4))
                variance_ratio = ""

            for data_type in set(labels):
                mask = [l == data_type for l in labels]
                ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1], label=data_type, color=color_map[data_type], alpha=0.6, s=10)
            
            ax.set_title(f'{method}, Layer {layer_idx} - {variance_ratio if variance_ratio else ""}', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Delete empty axes for the rest of the grid
        for idx in range(n_layers, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        plt.tight_layout()
        if save_plots:
            filename = f'activation_space_all_layers_{method.lower()}.png'
            filepath = os.path.join(plot_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved combined visualization for {method} to {filepath}")
        plt.show()
        plt.close(fig)


def plot_steering_similarities(steering_vectors: Dict[str, torch.Tensor],
                             save_plots: bool, plot_dir: str) -> None:
    """
    Косинусное сходство между Steering-векторами всех пар слоев.
    - Похожие смежные слои могут показывают на постепенное развитие концепции.
    - Низкое сходство между слоями может указывать на функциональную специализацию определенного слоя, тем самым он не реагирует на оаределенный концепт
    - Очень высокие значения могут показывать на рудундантную информацию в слоях
    """

    # Get layer steering vectors
    layer_keys = [k for k in steering_vectors.keys() if k.startswith('layer_') and not k.startswith('overall')]
    if len(layer_keys) < 2:
        return

    # Calculate similarity matrix
    n_layers = len(layer_keys)
    similarity_matrix = np.zeros((n_layers, n_layers))
    layer_nums = sorted([int(k.split('_')[1]) for k in layer_keys])

    for i, layer1 in enumerate(layer_nums):
        for j, layer2 in enumerate(layer_nums):
            vec1 = steering_vectors[f'layer_{layer1}']
            vec2 = steering_vectors[f'layer_{layer2}']
            sim = torch.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
            similarity_matrix[i, j] = sim

    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix,
                xticklabels=[f'Layer {l}' for l in layer_nums],
                yticklabels=[f'Layer {l}' for l in layer_nums],
                annot=True, cmap='coolwarm',annot_kws={"size": 8},
                fmt='.2f',
                center=0,
                square=True, linewidths=0.5)
    plt.title('Cosine Similarity Between Layer Steering Vectors')
    plt.tight_layout()

    if save_plots:
        plt.savefig(os.path.join(plot_dir, 'steering_similarities.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def analyze_projections(activations: Dict[str, Dict[int, torch.Tensor]],
                       steering_vectors: Dict[str, torch.Tensor],
                       save_plots: bool, plot_dir: str,
                       target_dtype: torch.dtype) -> None:
    """
    Анализ активаций.
    """

    if 'overall' not in steering_vectors:
        print("No normalized steering vector found for projection analysis")
        return

    steering_vec = steering_vectors['overall']

    # Calculate projections for each dataset
    projections = {}

    for data_type, acts in activations.items():
        data_projections = []

        # the same layer as used for overall steering vector calculation
        # (typically averaging across layers, so we'll use a representative layer)
        available_layers = list(acts.keys())
        middle_layer = available_layers[len(available_layers) // 2]

        if middle_layer in acts:
            layer_acts = acts[middle_layer]  # [n_samples, hidden_dim]

            # Project onto steering vector
            projections_vals = torch.matmul(layer_acts, steering_vec).cpu().to(target_dtype).numpy()
            projections[data_type] = projections_vals

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Histogram of projections
    colors = {'train_real': 'blue', 'train_fake': 'red', 'dev_real': 'lightblue', 'dev_fake': 'lightcoral'}

    for data_type, proj_vals in projections.items():
        ax1.hist(proj_vals, bins=30, alpha=0.6, label=data_type, color=colors[data_type])

    ax1.set_xlabel('Projection onto Steering Vector')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Activations Projected onto Steering Vector')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot comparison
    box_data = []
    box_labels = []
    for data_type, proj_vals in projections.items():
        box_data.append(proj_vals)
        box_labels.append(data_type)

    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, data_type in zip(bp['boxes'], box_labels):
        patch.set_facecolor(colors[data_type])

    ax2.set_ylabel('Projection Value')
    ax2.set_title('Projection Values by Dataset')
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(plot_dir, 'projection_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    # Print statistics
    print("\nProjection Statistics:")
    for data_type, proj_vals in projections.items():
        print(f"{data_type}: Mean={np.mean(proj_vals):.4f}, Std={np.std(proj_vals):.4f}")


def visualize_steering_vectors(
    target_dtype: torch.dtype,
    results_path: str = "../steering_data/llama_steering_vectors/complete_results.pkl",
    save_plots: bool = True,
    plot_dir: str = "./visualisation_data_from_steeging_vectors/cache",
    random_state: int = 42
) -> None:
    """
    Создает визуализации для steering векторов и активаций.
    """

    # Create plot directory
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)

    # Load results
    print("Loading results...")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    steering_vectors = results['steering_vectors']
    metadata = results['metadata']

    # Load activations for visualization
    activations_dir = os.path.dirname(results_path)
    train_real_path = os.path.join(activations_dir, "train_real_activations.pkl")
    train_fake_path = os.path.join(activations_dir, "train_fake_activations.pkl")
    dev_real_path = os.path.join(activations_dir, "dev_real_activations.pkl")
    dev_fake_path = os.path.join(activations_dir, "dev_fake_activations.pkl")

    activations = {}
    for name, path in [("train_real", train_real_path), ("train_fake", train_fake_path),
                       ("dev_real", dev_real_path), ("dev_fake", dev_fake_path)]:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                activations[name] = pickle.load(f)

    print(f"Loaded {len(activations)} activation sets")

    # 1. Steering Vector Magnitude Analysis
    plot_steering_magnitudes(steering_vectors, save_plots, plot_dir)

    # 2. Layer-wise Steering Vector Analysis
    plot_layerwise_analysis(steering_vectors, save_plots, plot_dir, target_dtype)

    # 3. Activation Space Visualization (UMAP/t-SNE/PCA)
    if activations:
        plot_activation_spaces(activations, save_plots, plot_dir, target_dtype, random_state)

    if activations:
        plot_activation_spaces_combined(activations, save_plots, plot_dir, target_dtype, random_state)

    # 4. Steering Vector Similarity Analysis
    plot_steering_similarities(steering_vectors, save_plots, plot_dir)

    # 5. Projection Analysis
    if activations:
        analyze_projections(activations, steering_vectors, save_plots, plot_dir, target_dtype)

    print(f"Visualization completed! Plots saved to: {plot_dir}")


