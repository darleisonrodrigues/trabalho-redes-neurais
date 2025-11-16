"""
Funções Auxiliares para o Projeto
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import warnings
warnings.filterwarnings('ignore')

# Configurações de plot
plt.style.use('default')
sns.set_palette("husl")

def load_spiral_data(filepath='spiral_d.csv'):
    """
    Carrega e prepara os dados do spiral dataset.
    
    Parameters:
    -----------
    filepath : str
        Caminho para o arquivo CSV
        
    Returns:
    --------
    tuple
        (X_original, X_scaled, y, scaler) - dados originais, normalizados, labels e scaler
    """
    try:
        df = pd.read_csv(filepath, header=None)
        df.columns = ['X1', 'X2', 'Class']
        
        print(f" Dados carregados com sucesso!")
        print(f"   Shape: {df.shape}")
        print(f"   Classes únicas: {df['Class'].unique()}")
        print(f"   Distribuição das classes:")
        print(f"   {df['Class'].value_counts()}")
        
        # Separar features e target
        X = df[['X1', 'X2']].values
        y = df['Class'].values
        
        # Normalizar features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X, X_scaled, y, scaler
        
    except FileNotFoundError:
        print(f" Erro: Arquivo '{filepath}' não encontrado!")
        return None, None, None, None
    except Exception as e:
        print(f" Erro ao carregar dados: {e}")
        return None, None, None, None

def calculate_specificity(y_true, y_pred):
    """
    Calcula a especificidade (taxa de verdadeiros negativos).
    
    Parameters:
    -----------
    y_true : array-like
        Labels verdadeiras
    y_pred : array-like
        Labels preditas
        
    Returns:
    --------
    float
        Valor da especificidade
    """
    try:
        cm = custom_confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            # Para casos multiclasse, calcular especificidade média
            specificity_scores = []
            for i in range(cm.shape[0]):
                tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
                fp = np.sum(cm[:, i]) - cm[i, i]
                specificity_scores.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            return np.mean(specificity_scores)
    except:
        return 0.0

def custom_confusion_matrix(y_true, y_pred):
    """
    Implementa matriz de confusão própria (sem usar sklearn).
    
    Parameters:
    -----------
    y_true : array-like
        Labels verdadeiras
    y_pred : array-like
        Labels preditas
        
    Returns:
    --------
    numpy.ndarray
        Matriz de confusão calculada manualmente
    """
    # Obter classes únicas
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    # Criar mapeamento de classes para índices
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Inicializar matriz de confusão
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # Preencher matriz de confusão
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = class_to_idx[true_label]
        pred_idx = class_to_idx[pred_label]
        cm[true_idx, pred_idx] += 1
    
    return cm

def calculate_all_metrics(y_true, y_pred):
    """
    Calcula todas as métricas de avaliação.
    
    Parameters:
    -----------
    y_true : array-like
        Labels verdadeiras
    y_pred : array-like
        Labels preditas
        
    Returns:
    --------
    dict
        Dicionário com todas as métricas
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'specificity': calculate_specificity(y_true, y_pred),
        'confusion_matrix': custom_confusion_matrix(y_true, y_pred)  # Usar implementação própria
    }

def plot_data_visualization(X_original, X_scaled, y, save_path='results/plots/data_visualization.png'):
    """
    Cria visualização dos dados originais e normalizados.
    
    Parameters:
    -----------
    X_original : array-like
        Dados originais
    X_scaled : array-like
        Dados normalizados
    y : array-like
        Labels
    save_path : str
        Caminho para salvar o plot
    """
    plt.figure(figsize=(15, 6))
    
    # Plot 1: Dados originais
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_original[:, 0], X_original[:, 1], c=y, cmap='RdYlBu', alpha=0.7, s=20)
    plt.colorbar(scatter)
    plt.title('Dados Originais - Spiral Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('X1', fontsize=12)
    plt.ylabel('X2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Dados normalizados
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='RdYlBu', alpha=0.7, s=20)
    plt.colorbar(scatter)
    plt.title('Dados Normalizados - Spiral Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('X1 (normalizado)', fontsize=12)
    plt.ylabel('X2 (normalizado)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Visualização salva em: {save_path}")
    plt.close()  # Fechar figura para liberar memória

def plot_confusion_matrix(cm, title, save_path=None):
    """
    Plota uma matriz de confusão.
    
    Parameters:
    -----------
    cm : array-like
        Matriz de confusão
    title : str
        Título do plot
    save_path : str, optional
        Caminho para salvar o plot
    """
    plt.figure(figsize=(8, 6))
    cm_int = cm.astype(int) if cm.dtype != int else cm
    sns.heatmap(cm_int, annot=True, fmt='d', cmap='Blues', cbar=True,
                square=True, linewidths=0.5)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predito', fontsize=12)
    plt.ylabel('Real', fontsize=12)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Matriz de confusão salva em: {save_path}")
    
    plt.tight_layout()
    plt.close()  # Fechar figura para liberar memória

def plot_multiple_confusion_matrices(results_dict, save_path='results/confusion_matrices/comparison.png'):
    """
    Plota múltiplas matrizes de confusão para comparação.
    
    Parameters:
    -----------
    results_dict : dict
        Dicionário com resultados dos modelos
    save_path : str
        Caminho para salvar o plot
    """
    n_models = len(results_dict)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if hasattr(axes, '__len__') else [axes]
    else:
        axes = axes.flatten()
    
    for i, (model_name, results) in enumerate(results_dict.items()):
        if i < len(axes):
            cm = results.get('confusion_matrix', np.zeros((2, 2)))
            accuracy = results.get('accuracy', 0.0)
            
            # Garantir que a matriz de confusão seja de inteiros
            cm_int = cm.astype(int) if cm.dtype != int else cm
            sns.heatmap(cm_int, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[i], cbar=True, square=True)
            axes[i].set_title(f'{model_name}\nAccuracy: {accuracy:.3f}', 
                            fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Predito')
            axes[i].set_ylabel('Real')
    
    # Remover subplots extras
    for i in range(len(results_dict), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Comparação de matrizes salva em: {save_path}")
    plt.close()  # Fechar figura para liberar memória

def create_statistics_table(simulation_results, save_path='results/stats/statistics_table.csv'):
    """
    Cria tabela com estatísticas das simulações.
    
    Parameters:
    -----------
    simulation_results : dict
        Resultados das simulações
    save_path : str
        Caminho para salvar a tabela
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame com as estatísticas
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
    
    # Criar DataFrame para cada métrica
    all_stats = {}
    
    for metric in metrics:
        stats_data = []
        for model_name in simulation_results.keys():
            values = np.array(simulation_results[model_name][metric])
            stats_data.append({
                'Modelo': model_name,
                'Média': np.mean(values),
                'Desvio_Padrão': np.std(values),
                'Maior_Valor': np.max(values),
                'Menor_Valor': np.min(values),
                'Mediana': np.median(values)
            })
        
        all_stats[metric] = pd.DataFrame(stats_data)
    
    # Salvar tabelas
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with pd.ExcelWriter(save_path.replace('.csv', '.xlsx')) as writer:
        for metric, df in all_stats.items():
            df.to_excel(writer, sheet_name=metric.title(), index=False)
    
    print(f" Estatísticas salvas em: {save_path.replace('.csv', '.xlsx')}")
    
    return all_stats

def print_metrics_summary(metrics_dict, model_name="Modelo"):
    """
    Imprime resumo das métricas de um modelo.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dicionário com as métricas
    model_name : str
        Nome do modelo
    """
    print(f"\n Resumo - {model_name}")
    print("-" * 40)
    print(f"Acurácia:      {metrics_dict['accuracy']:.4f}")
    print(f"Precisão:      {metrics_dict['precision']:.4f}")
    print(f"Recall:        {metrics_dict['recall']:.4f}")
    print(f"F1-Score:      {metrics_dict['f1']:.4f}")
    print(f"Especificidade: {metrics_dict['specificity']:.4f}")

def create_boxplot_comparison(simulation_results, save_path='results/plots/boxplot_comparison.png'):
    """
    Cria boxplots para comparação das métricas entre modelos.
    
    Parameters:
    -----------
    simulation_results : dict
        Resultados das simulações
    save_path : str
        Caminho para salvar o plot
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        data = []
        labels = []
        
        for model_name in simulation_results.keys():
            data.append(simulation_results[model_name][metric])
            labels.append(model_name)
        
        box_plot = axes[i].boxplot(data, labels=labels, patch_artist=True)
        
        # Colorir boxplots
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        for patch, color in zip(box_plot['boxes'], colors[:len(labels)]):
            patch.set_facecolor(color)
        
        axes[i].set_title(f'Distribuição - {metric.title()}', fontsize=14, fontweight='bold')
        axes[i].set_ylabel(metric.title(), fontsize=12)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='x', rotation=45)
    
    # Remover subplot extra
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Boxplots salvos em: {save_path}")
    plt.close()  # Fechar figura para liberar memória

def split_data_with_validation(X, y, test_size=0.2, random_state=42):
    """
    Divide os dados em treino e teste com validação estratificada.
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Labels
    test_size : float
        Proporção dos dados de teste
    random_state : int
        Seed para reprodutibilidade
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, 
                           random_state=random_state, stratify=y)

def save_model_results(model_name, results, filepath_prefix='results/stats/'):
    """
    Salva os resultados de um modelo em arquivo JSON.
    
    Parameters:
    -----------
    model_name : str
        Nome do modelo
    results : dict
        Resultados do modelo
    filepath_prefix : str
        Prefixo do caminho do arquivo
    """
    import json
    
    # Converter numpy arrays para listas para serialização JSON
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value
    
    filepath = f"{filepath_prefix}{model_name.lower()}_results.json"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f" Resultados do {model_name} salvos em: {filepath}")



def print_section_header(section_name, section_number):
    """
    Imprime cabeçalho de seção.
    
    Parameters:
    -----------
    section_name : str
        Nome da seção
    section_number : int
        Número da seção
    """
    print(f"\n{'='*60}")
    print(f"  {section_number}. {section_name.upper()}")
    print(f"{'='*60}")