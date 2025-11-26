# Análise de Casos Extremos - análise detalhada dos casos extremos das simulações

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import learning_curve
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

from utils import custom_confusion_matrix, plot_confusion_matrix, split_data_with_validation


def analyze_extreme_cases_detailed(simulation_results, X_scaled, y, save_path_prefix='results/'):
    print("\n ANÁLISE DETALHADA DOS CASOS EXTREMOS")
    print("=" * 70)
    
    # Import local para evitar dependências circulares
    from perceptron_simple import PerceptronSimple
    from adaline import ADALINE
    from mlp import MLP
    from rbf import RBF
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
    extreme_cases = {}
    
    # Configuração dos modelos
    models_config = {
        'Perceptron': lambda rs: PerceptronSimple(max_iter=1000, random_state=rs),
        'ADALINE': lambda rs: ADALINE(learning_rate=0.01, max_iter=1000, random_state=rs),
        'MLP': lambda rs: MLP(hidden_layer_sizes=(10,), max_iter=1000, random_state=rs),
        'RBF': lambda rs: RBF(n_centers=10, sigma=None, random_state=rs)
    }
    
    for metric in metrics:
        print(f"\n MÉTRICA: {metric.upper()}")
        print("-" * 50)
        
        extreme_cases[metric] = {}
        
        for model_name in simulation_results.keys():
            values = np.array(simulation_results[model_name][metric])
            max_idx = np.argmax(values)
            min_idx = np.argmin(values)
            
            print(f"{model_name}:")
            print(f"   Melhor caso: {values[max_idx]:.4f} (Simulação {max_idx + 1})")
            print(f"   Pior caso: {values[min_idx]:.4f} (Simulação {min_idx + 1})")
            
            # Armazenar informações dos casos extremos
            extreme_cases[metric][model_name] = {
                'best': {'value': values[max_idx], 'simulation': max_idx + 1},
                'worst': {'value': values[min_idx], 'simulation': min_idx + 1}
            }
            
            # Treinar modelos para os casos extremos
            for case_type, case_info in [('best', extreme_cases[metric][model_name]['best']), 
                                        ('worst', extreme_cases[metric][model_name]['worst'])]:
                
                # Reproduzir a mesma divisão dos dados
                sim_idx = case_info['simulation'] - 1
                X_train, X_test, y_train, y_test = split_data_with_validation(
                    X_scaled, y, test_size=0.2, random_state=sim_idx
                )
                
                # Treinar modelo
                try:
                    if model_name in models_config:
                        model = models_config[model_name](sim_idx)
                        
                        if hasattr(model, 'fit'):
                            if model_name == 'ADALINE':
                                model.fit(X_train, y_train, verbose=False)
                            else:
                                model.fit(X_train, y_train)
                        
                        y_pred = model.predict(X_test)
                        
                        # Calcular matriz de confusão personalizada
                        cm = custom_confusion_matrix(y_test, y_pred)
                        case_info['confusion_matrix'] = cm
                        case_info['y_test'] = y_test
                        case_info['y_pred'] = y_pred
                        
                        # Plotar matriz de confusão
                        title = f'{model_name} - {case_type.title()} {metric.title()}\nSim {case_info["simulation"]}: {case_info["value"]:.4f}'
                        save_path = f'{save_path_prefix}confusion_matrices/{model_name}_{metric}_{case_type}_case.png'
                        plot_confusion_matrix(cm, title, save_path)
                        
                except Exception as e:
                    print(f"   Erro ao treinar {model_name} para caso {case_type}: {e}")
    
    return extreme_cases


def plot_learning_curves_extreme_cases(extreme_cases, X_scaled, y, save_path='results/plots/learning_curves_extreme_cases.png'):
    # Plota curvas de aprendizado para os casos extremos selecionados
    print(f"\n GERANDO CURVAS DE APRENDIZADO PARA CASOS EXTREMOS")
    
    # Selecionar alguns casos representativos (accuracy)
    if 'accuracy' in extreme_cases:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        model_configs = {
            'Perceptron': Perceptron(max_iter=1000, random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
        }
        
        plot_idx = 0
        for model_name, model in model_configs.items():
            if model_name in extreme_cases['accuracy'] and plot_idx < 4:
                try:
                    # Plotar curva de aprendizado
                    train_sizes, train_scores, val_scores = learning_curve(
                        model, X_scaled, y, cv=5, n_jobs=-1,
                        train_sizes=np.linspace(0.1, 1.0, 10)
                    )
                    
                    train_mean = np.mean(train_scores, axis=1)
                    train_std = np.std(train_scores, axis=1)
                    val_mean = np.mean(val_scores, axis=1)
                    val_std = np.std(val_scores, axis=1)
                    
                    axes[plot_idx].plot(train_sizes, train_mean, 'o-', color='blue', 
                                       label=f'{model_name} - Treino')
                    axes[plot_idx].fill_between(train_sizes, train_mean - train_std, 
                                               train_mean + train_std, alpha=0.2, color='blue')
                    
                    axes[plot_idx].plot(train_sizes, val_mean, 'o-', color='red', 
                                       label=f'{model_name} - Validação')
                    axes[plot_idx].fill_between(train_sizes, val_mean - val_std, 
                                               val_mean + val_std, alpha=0.2, color='red')
                    
                    best_acc = extreme_cases['accuracy'][model_name]['best']['value']
                    worst_acc = extreme_cases['accuracy'][model_name]['worst']['value']
                    
                    axes[plot_idx].set_title(f'{model_name} - Curva de Aprendizado\n'
                                            f'Melhor: {best_acc:.3f} | Pior: {worst_acc:.3f}')
                    axes[plot_idx].set_xlabel('Tamanho do Conjunto de Treinamento')
                    axes[plot_idx].set_ylabel('Score de Acurácia')
                    axes[plot_idx].legend()
                    axes[plot_idx].grid(True, alpha=0.3)
                    
                    plot_idx += 1
                    
                except Exception as e:
                    print(f"   Erro ao gerar curva para {model_name}: {e}")
        
        # Remover subplots não utilizados
        for i in range(plot_idx, 4):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Curvas de aprendizado salvas em: {save_path}")
        plt.close()  # Fechar figura para liberar memória