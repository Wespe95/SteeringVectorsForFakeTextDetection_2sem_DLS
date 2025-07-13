"""
This module contains functions for calculating the projections of steering vectors 
and preform there for classification task with Machine Learning.
"""

import os
import pickle
from typing import Any, Dict, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt

#############################
# Загрузка данных
#############################
def load_steering_data(
    save_dir: str = "./kaggle/working/llama_steering_vectors",
    domain_tag: Optional[str] = None,
    alt_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Загрузка созданных ранее стиринг векторов и активаций
    """
    train_dir = save_dir
    dev_dir = alt_dir if domain_tag == "from_another_domain" and alt_dir else save_dir

    # 2. Train_data
    print(f"Lade Trainingsdaten aus {train_dir}")
    train_results_file = os.path.join(train_dir, "complete_results.pkl")
    with open(train_results_file, 'rb') as f:
        train_complete_results = pickle.load(f)

    train_real_file = os.path.join(train_dir, "train_real_activations.pkl")
    train_fake_file = os.path.join(train_dir, "train_fake_activations.pkl")
    with open(train_real_file, 'rb') as f:
        train_real_activations = pickle.load(f)
    with open(train_fake_file, 'rb') as f:
        train_fake_activations = pickle.load(f)

    data_train = {
        'steering_vectors': train_complete_results['steering_vectors'],
        'metadata': train_complete_results['metadata'],
        'train_real_activations': train_real_activations,
        'train_fake_activations': train_fake_activations
    }

    # 3. DEv_Data
    print(f"Lade Entwicklungsdaten aus {dev_dir}")
    dev_results_file = os.path.join(dev_dir, "complete_results.pkl")
    with open(dev_results_file, 'rb') as f:
        dev_complete_results = pickle.load(f)

    dev_real_file = os.path.join(dev_dir, "dev_real_activations.pkl")
    dev_fake_file = os.path.join(dev_dir, "dev_fake_activations.pkl")

    with open(dev_real_file, 'rb') as f:
        dev_real_activations = pickle.load(f)
    with open(dev_fake_file, 'rb') as f:
        dev_fake_activations = pickle.load(f)

    data_dev = {
        'steering_vectors': train_complete_results['steering_vectors'], # Steering Vectors from the train domain
        'metadata': dev_complete_results['metadata'],
        'dev_real_activations': dev_real_activations,
        'dev_fake_activations': dev_fake_activations
    }

    # 4. Ausgabe der Statistiken
    train_meta = data_train['metadata']
    dev_meta = data_dev['metadata']
    print(f"TRAIN-DATA: {train_meta['train_real_count']} real, {train_meta['train_fake_count']} fake Samples")
    print(f"DEV-DATA: {dev_meta['dev_real_count']} real, {dev_meta['dev_fake_count']} fake Samples")

    return data_train, data_dev

##########################
# Вычисление проекций
##########################
def calculate_projections(
    data_train: Dict[str, Any],
    data_dev: Dict[str, Any],
    save_dir: str = "./steering_vectors_dotProduct"
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
    """
    Calculates dot product projections for training and development data separately.

    - Steering vectors from data_train
    - Activations from data_train (train_real_activations, train_fake_activations) and data_dev (dev_real_activations, dev_fake_activations)

    Returns two dictionaries:
    (layer_projections_train, layer_projections_dev)
    """

    print(" Calculate projections for train and dev data...")

    steering_vectors = data_train['steering_vectors']

    # Train-Data
    train_real_activations = data_train.get('train_real_activations')
    train_fake_activations = data_train.get('train_fake_activations')

    # Dev-Data
    dev_real_activations = data_dev.get('dev_real_activations')
    dev_fake_activations = data_dev.get('dev_fake_activations')

    projections_dir = os.path.join(save_dir, "projections")
    os.makedirs(projections_dir, exist_ok=True)

    layer_projections_train = {}
    layer_projections_dev = {}

    # Nur Layer-Steering-Vektoren (ohne 'overall')
    layer_steering_vectors = {k: v for k, v in steering_vectors.items()
                              if k.startswith('layer_') and k != 'overall'}

    print(f"Verarbeite {len(layer_steering_vectors)} Layer-Steering-Vektoren")

    for i, layer_name in enumerate(layer_steering_vectors.keys()):
        steering_vec = steering_vectors[layer_name]

        # Convert steering vector to NumPy array
        if hasattr(steering_vec, 'detach'):
            steering_vec_np = steering_vec.detach().cpu().numpy()
        else:
            steering_vec_np = np.asarray(steering_vec)

        # --- Train_Data-Projections ---
        train_real_proj = []
        train_fake_proj = []
        if train_real_activations is not None and train_fake_activations is not None:
            for activation in train_real_activations[i]:
                if hasattr(activation, 'detach'):
                    activation_np = activation.detach().cpu().numpy()
                else:
                    activation_np = np.asarray(activation)
                proj = np.dot(activation_np.flatten(), steering_vec_np.flatten())
                train_real_proj.append(proj)

            for activation in train_fake_activations[i]:
                if hasattr(activation, 'detach'):
                    activation_np = activation.detach().cpu().numpy()
                else:
                    activation_np = np.asarray(activation)
                proj = np.dot(activation_np.flatten(), steering_vec_np.flatten())
                train_fake_proj.append(proj)

            layer_projections_train[layer_name] = {
                'real': np.array(train_real_proj),
                'fake': np.array(train_fake_proj)
            }

            # Save training projections per layer
            train_layer_file = os.path.join(projections_dir, f"{layer_name}_train_projections.pkl")
            with open(train_layer_file, 'wb') as f:
                pickle.dump(layer_projections_train[layer_name], f)
        else:
            print(f"Warnung: training activations for layer {layer_name} not found. ")

        # --- Dev_Data-Projections ---
        dev_real_proj = []
        dev_fake_proj = []
        if dev_real_activations is not None and dev_fake_activations is not None:
            for activation in dev_real_activations[i]:
                if hasattr(activation, 'detach'):
                    activation_np = activation.detach().cpu().numpy()
                else:
                    activation_np = np.asarray(activation)
                proj = np.dot(activation_np.flatten(), steering_vec_np.flatten())
                dev_real_proj.append(proj)

            for activation in dev_fake_activations[i]:
                if hasattr(activation, 'detach'):
                    activation_np = activation.detach().cpu().numpy()
                else:
                    activation_np = np.asarray(activation)
                proj = np.dot(activation_np.flatten(), steering_vec_np.flatten())
                dev_fake_proj.append(proj)

            layer_projections_dev[layer_name] = {
                'real': np.array(dev_real_proj),
                'fake': np.array(dev_fake_proj)
            }

            # Save development projections per layer
            dev_layer_file = os.path.join(projections_dir, f"{layer_name}_dev_projections.pkl")
            with open(dev_layer_file, 'wb') as f:
                pickle.dump(layer_projections_dev[layer_name], f)
        else:
            print(f"Warnung: Development activations for layer {layer_name} not found. ")

    # Total save
    with open(os.path.join(projections_dir, "all_train_projections.pkl"), 'wb') as f:
        pickle.dump(layer_projections_train, f)

    with open(os.path.join(projections_dir, "all_dev_projections.pkl"), 'wb') as f:
        pickle.dump(layer_projections_dev, f)

    print(f"Training- and Development projections saved in {projections_dir}")

    return layer_projections_train, layer_projections_dev


#################################
# Создание данных для обучения ML модели
#################################
def create_classification_dataset(layer_projections: Dict[str, Dict[str, np.ndarray]],
                                layer_name: str,
                                real_label: int = 0,
                                fake_label: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Создает набор данных для детекции из проекций слоя.
    """
    real_proj = layer_projections[layer_name]['real']
    fake_proj = layer_projections[layer_name]['fake']

    # Combine projections and create labels
    X = np.concatenate([real_proj, fake_proj]).reshape(-1, 1)
    y = np.concatenate([
        np.full(len(real_proj), real_label),
        np.full(len(fake_proj), fake_label)
    ])

    return X, y

###################################
# Вычисление корролляций между проекциями и лейблами
###################################
def calculate_correlation(layer_projections: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, float]:
    """
    Значения корреляции между проекциями и метками для каждого слоя.
    """
    correlations = {}

    for layer_name in layer_projections.keys():
        X, y = create_classification_dataset(layer_projections, layer_name)
        correlation, p_value = pearsonr(X.flatten(), y)
        correlations[layer_name] = {
            'correlation': correlation,
            'p_value': p_value,
            'abs_correlation': abs(correlation)
        }

    return correlations


#################################
# Тренировка 3-х классификаторовЮ логистической регрессии, Random Forest, SVM
#################################
def train_classifiers(X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_dev: np.ndarray,
                        y_dev: np.ndarray,
                        random_state: int = 42
                    ) -> Dict[str, Dict[str, Any]]:
    """
    Обучает различные классификаторы машинного обучения: логистической регрессии, Random Forest, SVM

    """

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)

    # Define classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'SVM': SVC(kernel='rbf', random_state=random_state)
    }

    results = {}

    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)

        # Vorhersagen auf Trainings- und Entwicklungsdaten
        y_train_pred = clf.predict(X_train_scaled)
        y_dev_pred = clf.predict(X_dev_scaled)

        # Metriken berechnen
        results[name] = {
            'model': clf,
            'scaler': scaler,
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'dev_accuracy': accuracy_score(y_dev, y_dev_pred),
            'train_precision': precision_score(y_train, y_train_pred, average='weighted'),
            'dev_precision': precision_score(y_dev, y_dev_pred, average='weighted'),
            'train_recall': recall_score(y_train, y_train_pred, average='weighted'),
            'dev_recall': recall_score(y_dev, y_dev_pred, average='weighted'),
            'train_f1': f1_score(y_train, y_train_pred, average='weighted'),
            'dev_f1': f1_score(y_dev, y_dev_pred, average='weighted'),
            'y_train_true': y_train,
            'y_train_pred': y_train_pred,
            'y_dev_true': y_dev,
            'y_dev_pred': y_dev_pred
        }

    return results


################################
# Оценка всех слоеа по отдельности от каждого классификатора
################################
def evaluate_all_layers(
                        layer_projections_train: Dict[str, Dict[str, np.ndarray]],
                        layer_projections_dev: Dict[str, Dict[str, np.ndarray]],
                        save_dir: str = "./steering_vectors_dotProduct") -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], dict]:
    """
    Оценка всех слоев по отдельности для train и dev множеств.
    """

    print("Evaluating all layers (train & dev)...")

    # Calculate correlations separately
    correlations_train = calculate_correlation(layer_projections_train)
    correlations_dev = calculate_correlation(layer_projections_dev)

    results_list = []
    classifier_data = {}

    best_model_info = {
        'accuracy': 0,
        'model': None,
        'scaler': None,
        'layer': None,
        'classifier_name': None,
        'metrics': {}
    }

    for layer_name in layer_projections_train.keys():
        # Train-Data
        X_train, y_train = create_classification_dataset(layer_projections_train, layer_name)
        # Dev-Data
        X_dev, y_dev = create_classification_dataset(layer_projections_dev, layer_name)

        classifier_results = train_classifiers(X_train, y_train, X_dev, y_dev)

        for clf_name, clf_result in classifier_results.items():
            result_row = {
                'layer': layer_name,
                'classifier': clf_name,
                'train_accuracy': clf_result['train_accuracy'],
                'dev_accuracy': clf_result['dev_accuracy'],
                'train_precision': clf_result['train_precision'],
                'dev_precision': clf_result['dev_precision'],
                'train_recall': clf_result['train_recall'],
                'dev_recall': clf_result['dev_recall'],
                'train_f1': clf_result['train_f1'],
                'dev_f1': clf_result['dev_f1'],
                'train_correlation': correlations_train[layer_name]['correlation'],
                'dev_correlation': correlations_dev[layer_name]['correlation'],
                'train_abs_correlation': correlations_train[layer_name]['abs_correlation'],
                'dev_abs_correlation': correlations_dev[layer_name]['abs_correlation'],
                'train_p_value': correlations_train[layer_name]['p_value'],
                'dev_p_value': correlations_dev[layer_name]['p_value'],
            }
            results_list.append(result_row)

            if clf_result['dev_accuracy'] > best_model_info['accuracy']:
                best_model_info.update({
                    'accuracy': clf_result['dev_accuracy'],
                    'model': clf_result['model'],
                    'scaler': clf_result['scaler'],
                    'layer': layer_name,
                    'classifier_name': clf_name,
                    'metrics': result_row
                })

            # For DataFrames per Classifier
            classifier_row = {k: v for k, v in result_row.items() if k != 'classifier'}
            if clf_name not in classifier_data:
                classifier_data[clf_name] = []
            classifier_data[clf_name].append(classifier_row)

    results_df = pd.DataFrame(results_list)
    classifier_dfs = {clf: pd.DataFrame(rows) for clf, rows in classifier_data.items()}

    # Save
    results_file = os.path.join(save_dir, "layer_evaluation_results.csv")
    results_df.to_csv(results_file, index=False)
    for clf_name, df in classifier_dfs.items():
        clf_file = os.path.join(save_dir, f"results_{clf_name.lower().replace(' ', '_')}.csv")
        df.to_csv(clf_file, index=False)

    print(f"Combined results saved to {results_file}")
    print(f"Best model: {best_model_info['classifier_name']} ({best_model_info['accuracy']:.4f})")

    return results_df, classifier_dfs, best_model_info


##################################
#
##################################
def plot_layer_performance(results_df: pd.DataFrame, save_dir: str = "./steering_vectors_dotProduct"):
    """
    Визуализация performance по слоям и классификаторам
    """
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Extract layer numbers for correct sorting
    def extract_layer_number(layer_name):
        try:
            return int(layer_name.split('_')[1])
        except:
            return 0

    # Sortiere DataFrame nach Layer-Nummern
    results_df_sorted = results_df.copy()
    results_df_sorted['layer_num'] = results_df_sorted['layer'].apply(extract_layer_number)
    results_df_sorted = results_df_sorted.sort_values('layer_num')

    # Hole alle Klassifikatoren
    classifiers = results_df['classifier'].unique()

    #metrics = ['accuracy', 'precision', 'recall', 'f1']
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_types = ['train', 'dev']
    #colors = sns.color_palette("bright", n_colors=len(classifiers)) #{'train': 'tab:blue', 'dev': 'tab:orange'}
    #classificator_colors = dict(zip(classifiers, colors))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        for classifier in classifiers:
            clf_data = results_df_sorted[results_df_sorted['classifier'] == classifier]
            for mtype in metric_types:
                col = f"{mtype}_{metric}"
                label = f"{classifier} ({mtype})"
                style = '-' if mtype == 'dev' else '--'
                ax.plot(clf_data['layer_num'], clf_data[col],
                        marker='o', linewidth=2, markersize=4,
                        label=label, linestyle=style) ##<-- color=colors[mtype]  color=colors
        ax.set_title(f'{metric.capitalize()} by Layer (Train & Dev)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Layer Number', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 33, 2))  # Layer 0 bis 32, alle 2

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'classifier_performance_by_layer_train_dev.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot 2: Separate Plots pro Classifier (Train & Dev)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for classifier in classifiers:
        clf_data = results_df_sorted[results_df_sorted['classifier'] == classifier]
        #colors = classificator_colors[classifier]
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            for mtype in metric_types:
                col = f"{mtype}_{metric}"
                style = '-' if mtype == 'dev' else '--'
                ax.plot(clf_data['layer_num'], clf_data[col],
                        marker='o', linewidth=2, markersize=4,
                        label=f"{mtype.capitalize()}", linestyle=style)
                ax.fill_between(clf_data['layer_num'], clf_data[col], alpha=0.15)
            ax.set_title(f'{classifier} - {metric.capitalize()} (Train & Dev)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Layer Number', fontsize=12)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(0, 33, 2))
            # Highlight best performing layer (Dev)
            best_idx = clf_data[f"dev_{metric}"].idxmax()
            best_layer = clf_data.loc[best_idx, 'layer_num']
            best_value = clf_data.loc[best_idx, f"dev_{metric}"]
            ax.scatter(best_layer, best_value, color='red', s=100, zorder=5)
            ax.annotate(f'Best Dev: L{best_layer}\n{best_value:.3f}',
                        xy=(best_layer, best_value),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        plt.suptitle(f'{classifier} Performance Across Layers (Train & Dev)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{classifier.lower().replace(" ", "_")}_performance_train_dev.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    # Plot 3: Heatmap für Dev-Accuracy (Train-Heatmap optional)
    plt.figure(figsize=(12, 8))
    pivot_dev_accuracy = results_df_sorted.pivot(index='layer', columns='classifier', values='dev_accuracy')
    layer_order = sorted(pivot_dev_accuracy.index.tolist(), key=extract_layer_number)
    pivot_dev_accuracy = pivot_dev_accuracy.reindex(layer_order)
    sns.heatmap(pivot_dev_accuracy, annot=True, cmap='viridis', fmt='.3f',
                cbar_kws={'label': 'Dev Accuracy'})
    plt.title('Dev Accuracy Heatmap by Layer and Classifier', fontsize=16, fontweight='bold')
    plt.xlabel('Classifier', fontsize=12)
    plt.ylabel('Layer', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'dev_accuracy_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot 4: Correlation by Layer (Train & Dev)
    plt.figure(figsize=(12, 6))
    for mtype  in metric_types:
        col = f"{mtype}_abs_correlation"
        correlation_data = results_df_sorted.groupby(['layer_num', 'layer'])[col].first().reset_index()
        plt.plot(correlation_data['layer_num'], correlation_data[col],
                 marker='o', linewidth=3, markersize=8, alpha=0.7, label=f"{mtype.capitalize()}") #color=color
        plt.fill_between(correlation_data['layer_num'], correlation_data[col], alpha=0.15) #color=color
        # Highlight best correlation (Dev)
        if mtype == 'dev':
            best_corr_idx = correlation_data[col].idxmax()
            best_layer = correlation_data.loc[best_corr_idx, 'layer_num']
            best_corr = correlation_data.loc[best_corr_idx, col]
            plt.scatter(best_layer, best_corr, color='red', s=150, zorder=5)
            plt.annotate(f'Best Dev: Layer {best_layer}\nCorr: {best_corr:.3f}',
                         xy=(best_layer, best_corr),
                         xytext=(10, 10), textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.title('Absolute Correlation by Layer (Train & Dev)', fontsize=16, fontweight='bold')
    plt.xlabel('Layer Number', fontsize=12)
    plt.ylabel('Absolute Correlation', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 33, 2))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_by_layer_train_dev.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot 5: Best Dev-Accuracy Pro Layer (Classifier)
    plt.figure(figsize=(12, 6))
    layer_best = []
    for layer_num in range(33):  # Layer 0-32
        layer_name = f'layer_{layer_num}'
        layer_data = results_df[results_df['layer'] == layer_name]
        if not layer_data.empty:
            best_row = layer_data.loc[layer_data['dev_accuracy'].idxmax()]
            layer_best.append({
                'layer_num': layer_num,
                'layer': layer_name,
                'best_classifier': best_row['classifier'],
                'best_dev_accuracy': best_row['dev_accuracy'],
                'best_dev_f1': best_row['dev_f1']
            })
    layer_best_df = pd.DataFrame(layer_best)
    classifier_colors = dict(zip(classifiers, plt.cm.Set3(np.linspace(0, 1, len(classifiers)))))
    for classifier in classifiers:
        clf_data = layer_best_df[layer_best_df['best_classifier'] == classifier]
        plt.scatter(clf_data['layer_num'], clf_data['best_dev_accuracy'],
                    label=f'{classifier} (best)', s=100, alpha=0.8,
                    color=classifier_colors[classifier])
    plt.plot(layer_best_df['layer_num'], layer_best_df['best_dev_accuracy'],
             'k--', alpha=0.5, linewidth=1, label='Best overall')
    plt.title('Best Classifier Dev Accuracy by Layer', fontsize=16, fontweight='bold')
    plt.xlabel('Layer Number', fontsize=12)
    plt.ylabel('Best Dev Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 33, 2))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'best_classifier_by_layer_dev.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"All plots saved to {plots_dir}")

    # Summary of the best Layer pro Classifier (Dev)
    print("\n=== BEST LAYERS PER CLASSIFIER (Dev) ===")
    for classifier in classifiers:
        clf_data = results_df[results_df['classifier'] == classifier]
        best_layer = clf_data.loc[clf_data['dev_accuracy'].idxmax()]
        print(f"{classifier}: {best_layer['layer']} (Dev Accuracy: {best_layer['dev_accuracy']:.4f})")

