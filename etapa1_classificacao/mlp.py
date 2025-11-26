# Implementação do MLP (Multi-Layer Perceptron) para classificação

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from utils import calculate_all_metrics, print_metrics_summary

class MLP:
    
    def __init__(self, hidden_layer_sizes=(10,), activation='relu', solver='adam',
                 learning_rate_init=0.01, max_iter=1000, random_state=42, **kwargs):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.model = None
        self.is_fitted = False
        self.training_history = {'loss': []}
        
    def fit(self, X_train, y_train, verbose=False):
        # Treina o modelo MLP
        if verbose:
            print(" Treinando MLP...")
            print(f"   Arquitetura: {self.hidden_layer_sizes}")
            print(f"   Ativação: {self.activation}")
            print(f"   Solver: {self.solver}")
        
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            **self.kwargs
        )
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Armazenar histórico de loss se disponível
        if hasattr(self.model, 'loss_curve_'):
            self.training_history['loss'] = self.model.loss_curve_
        
        if verbose:
            print(" MLP treinado com sucesso!")
            if hasattr(self.model, 'n_iter_'):
                print(f"   Iterações realizadas: {self.model.n_iter_}")
            if hasattr(self.model, 'loss_'):
                print(f"   Loss final: {self.model.loss_:.6f}")
        
        return self
    
    def predict(self, X_test):
        # Faz predições usando o modelo treinado
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        # Retorna probabilidades de predição
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        return self.model.predict_proba(X_test)
    
    def evaluate(self, X_test, y_test, verbose=True):
        # Avalia o desempenho do modelo
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        y_pred = self.predict(X_test)
        metrics = calculate_all_metrics(y_test, y_pred)
        
        if verbose:
            architecture_str = f"MLP{self.hidden_layer_sizes}"
            print_metrics_summary(metrics, architecture_str)
        
        return metrics
    
    def plot_loss_curve(self, save_path=None):
        # Plota a curva de loss durante o treinamento
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        if not self.training_history['loss']:
            print("  Histórico de loss não disponível.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history['loss'], 'b-', linewidth=2)
        plt.title(f'MLP{self.hidden_layer_sizes} - Curva de Loss', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Iterações', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Curva de loss salva em: {save_path}")
        
        plt.tight_layout()
        plt.close()  # Fechar figura para liberar memória
    
    def plot_learning_curve(self, X, y, cv=5, save_path=None):
        # Plota a curva de aprendizado (training vs validation score)
        # Criar um novo modelo para a curva de aprendizado
        model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            **self.kwargs
        )
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Score de Treinamento')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.2, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Score de Validação')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.2, color='red')
        
        plt.title(f'MLP{self.hidden_layer_sizes} - Curva de Aprendizado', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Tamanho do Conjunto de Treinamento', fontsize=12)
        plt.ylabel('Accuracy Score', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Curva de aprendizado salva em: {save_path}")
        
        plt.tight_layout()
        plt.close()  # Fechar figura para liberar memória
        
        return train_sizes, train_scores, val_scores
    
    def get_hyperparameters(self):
        # Retorna os hiperparâmetros do modelo
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'solver': self.solver,
            'learning_rate_init': self.learning_rate_init,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'algorithm': 'MLP',
            'type': 'Non-linear Multi-layer Classifier'
        }

def create_mlp_configurations():
    # Cria diferentes configurações de MLP para análise de under/overfitting
    configurations = {
        'Underfitted_MLP': {
            'hidden_layer_sizes': (2,),
            'max_iter': 50,
            'description': 'Modelo subdimensionado - propenso a underfitting'
        },
        'Balanced_MLP': {
            'hidden_layer_sizes': (10,),
            'max_iter': 1000,
            'description': 'Modelo balanceado - boa generalização'
        },
        'Overfitted_MLP': {
            'hidden_layer_sizes': (100, 50, 25),
            'max_iter': 2000,
            'description': 'Modelo superdimensionado - propenso a overfitting'
        }
    }
    
    print(" Configurações MLP criadas:")
    for name, config in configurations.items():
        print(f"   • {name}: {config['hidden_layer_sizes']} - {config['description']}")
    
    return configurations

def create_standard_mlp():
    # Cria um MLP com configuração padrão otimizada
    print(" Criando MLP padrão com hiperparâmetros otimizados...")
    print(" Hiperparâmetros escolhidos:")
    print("   • hidden_layer_sizes=(10,): Uma camada oculta com 10 neurônios")
    print("   • activation='relu': Função de ativação ReLU")
    print("   • solver='adam': Otimizador Adam")
    print("   • learning_rate_init=0.01: Taxa de aprendizado inicial")
    print("   • max_iter=1000: Número máximo de iterações")
    
    return MLP(
        hidden_layer_sizes=(10,),
        activation='relu',
        solver='adam',
        learning_rate_init=0.01,
        max_iter=1000,
        random_state=42
    )

def analyze_overfitting_underfitting(X_train, X_test, y_train, y_test):
    # Analisa underfitting e overfitting com diferentes configurações MLP
    print("\n ANÁLISE DE UNDERFITTING/OVERFITTING - MLP")
    print("=" * 60)
    
    configurations = create_mlp_configurations()
    results = {}
    
    for name, config in configurations.items():
        print(f"\n Treinando {name}...")
        print(f"   Arquitetura: {config['hidden_layer_sizes']}")
        
        # Criar e treinar modelo
        mlp = MLP(
            hidden_layer_sizes=config['hidden_layer_sizes'],
            max_iter=config['max_iter'],
            activation='relu',
            solver='adam',
            random_state=42
        )
        
        mlp.fit(X_train, y_train, verbose=True)
        
        # Avaliar no conjunto de treino e teste
        y_pred_train = mlp.predict(X_train)
        y_pred_test = mlp.predict(X_test)
        
        train_metrics = calculate_all_metrics(y_train, y_pred_train)
        test_metrics = calculate_all_metrics(y_test, y_pred_test)
        
        # Armazenar resultados
        results[name] = {
            'model': mlp,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_accuracy': train_metrics['accuracy'],
            'test_accuracy': test_metrics['accuracy'],
            'accuracy_diff': abs(train_metrics['accuracy'] - test_metrics['accuracy']),
            'configuration': config
        }
        
        print(f"    Acurácia Treino: {train_metrics['accuracy']:.4f}")
        print(f"    Acurácia Teste:  {test_metrics['accuracy']:.4f}")
        print(f"    Diferença:       {results[name]['accuracy_diff']:.4f}")
        
        # Interpretação
        if results[name]['accuracy_diff'] > 0.1:
            if train_metrics['accuracy'] > test_metrics['accuracy']:
                print("   🔴 Indicativo de OVERFITTING")
            else:
                print("   🔵 Baixo desempenho geral")
        elif test_metrics['accuracy'] < 0.6:
            print("   🟡 Indicativo de UNDERFITTING")
        else:
            print("   🟢 Modelo BALANCEADO")
    
    return results

def test_mlp():
    # Função de teste para o MLP
    print(" Testando MLP...")
    
    # Dados sintéticos para teste
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, 
                              n_informative=2, random_state=42, n_clusters_per_class=1)
    y = np.where(y == 0, -1, 1)  # Converter para {-1, 1}
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Testar modelo padrão
    mlp = create_standard_mlp()
    mlp.fit(X_train, y_train, verbose=True)
    metrics = mlp.evaluate(X_test, y_test)
    
    # Plotar curvas
    mlp.plot_loss_curve()
    mlp.plot_learning_curve(X, y)
    
    print(" Teste concluído com sucesso!")
    return metrics

if __name__ == "__main__":
    # Executar teste se o arquivo for executado diretamente
    test_mlp()