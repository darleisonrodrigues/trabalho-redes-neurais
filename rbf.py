"""
Implementação de RBF (Radial Basis Function Network)
===================================================

Este módulo implementa a Rede de Base Radial para classificação.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from utils import calculate_all_metrics, print_metrics_summary

class RBF:
    
    def __init__(self, n_centers=10, sigma=None, random_state=42):
        self.n_centers = n_centers
        self.sigma = sigma
        self.random_state = random_state
        
        # Atributos definidos durante o treinamento
        self.centers = None
        self.weights = None
        self.sigma_computed = None
        self.is_fitted = False
        
    def _gaussian_rbf(self, X, center, sigma):
        # Calcula a função de base radial Gaussiana
        distances = euclidean_distances(X, center.reshape(1, -1)).flatten()
        return np.exp(-(distances**2) / (2 * sigma**2))
    
    def _calculate_design_matrix(self, X):
        # Calcula a matriz de design (matriz G)
        n_samples = X.shape[0]
        G = np.zeros((n_samples, self.n_centers))
        
        for i in range(n_samples):
            for j in range(self.n_centers):
                G[i, j] = self._gaussian_rbf(X[i:i+1], self.centers[j], self.sigma_computed)
        
        return G
    
    def _determine_centers(self, X):
        # Determina os centros usando K-means
        # Aplicar K-means para encontrar centros
        kmeans = KMeans(n_clusters=self.n_centers, random_state=self.random_state, n_init=10)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_
        
    def _determine_sigma(self, X):
        # Determina o parâmetro sigma se não foi especificado
        if self.sigma is None:
            # Calcular sigma como a distância média entre centros dividida pelo número de centros
            distances = euclidean_distances(self.centers)
            # Remover diagonal (distância zero de um centro para ele mesmo)
            distances = distances[distances > 0]
            self.sigma_computed = np.mean(distances) / np.sqrt(2 * self.n_centers)
        else:
            self.sigma_computed = self.sigma
    
    def fit(self, X_train, y_train, verbose=False):
        # Treina a rede RBF
        if verbose:
            print(" Treinando RBF...")
            print(f"   Número de centros: {self.n_centers}")
        
        # Etapa 1: Determinar centros usando K-means (aprendizado não supervisionado)
        if verbose:
            print("    Determinando centros com K-means...")
        self._determine_centers(X_train)
        
        # Etapa 2: Determinar sigma
        self._determine_sigma(X_train)
        if verbose:
            print(f"    Sigma calculado: {self.sigma_computed:.4f}")
        
        # Etapa 3: Calcular matriz de design
        if verbose:
            print("    Calculando matriz de design...")
        G = self._calculate_design_matrix(X_train)
        
        # Etapa 4: Resolver para os pesos usando pseudo-inversa (aprendizado supervisionado)
        if verbose:
            print("     Calculando pesos com pseudo-inversa...")
        try:
            # Usar pseudo-inversa de Moore-Penrose
            self.weights = np.linalg.pinv(G).dot(y_train)
        except np.linalg.LinAlgError:
            # Em caso de erro, usar solução por mínimos quadrados com regularização
            if verbose:
                print("     Usando regularização Ridge...")
            lambda_reg = 1e-6
            self.weights = np.linalg.solve(G.T.dot(G) + lambda_reg * np.eye(G.shape[1]), G.T.dot(y_train))
        
        self.is_fitted = True
        
        if verbose:
            print(" RBF treinada com sucesso!")
            print(f"    Pesos calculados: shape {self.weights.shape}")
        
        return self
    
    def predict(self, X_test):
        # Faz predições usando o modelo treinado
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        # Calcular matriz de design para dados de teste
        G_test = self._calculate_design_matrix(X_test)
        
        # Calcular saída da rede
        output = G_test.dot(self.weights)
        
        # Aplicar função degrau para classificação
        return np.where(output >= 0, 1, -1)
    
    def predict_continuous(self, X_test):
        # Retorna saídas contínuas (antes da função degrau)
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        G_test = self._calculate_design_matrix(X_test)
        return G_test.dot(self.weights)
    
    def evaluate(self, X_test, y_test, verbose=True):
        # Avalia o desempenho do modelo
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        y_pred = self.predict(X_test)
        metrics = calculate_all_metrics(y_test, y_pred)
        
        if verbose:
            print_metrics_summary(metrics, f"RBF (n_centers={self.n_centers})")
        
        return metrics
    
    def plot_rbf_functions(self, X_range=None, save_path=None):
        # Plota as funções RBF no espaço 2D
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        if self.centers.shape[1] != 2:
            print("  Visualização disponível apenas para dados 2D.")
            return
        
        # Definir range se não fornecido
        if X_range is None:
            margin = 1.0
            x_min, x_max = self.centers[:, 0].min() - margin, self.centers[:, 0].max() + margin
            y_min, y_max = self.centers[:, 1].min() - margin, self.centers[:, 1].max() + margin
        else:
            x_min, x_max, y_min, y_max = X_range
        
        # Criar grid
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Plotar centros e círculos representando sigma
        plt.figure(figsize=(12, 8))
        
        # Plot centros
        plt.scatter(self.centers[:, 0], self.centers[:, 1], c='red', s=100, 
                   marker='x', linewidth=3, label=f'Centros RBF (n={self.n_centers})')
        
        # Plot círculos representando a largura sigma
        circle_points = np.linspace(0, 2*np.pi, 100)
        for i, center in enumerate(self.centers):
            circle_x = center[0] + self.sigma_computed * np.cos(circle_points)
            circle_y = center[1] + self.sigma_computed * np.sin(circle_points)
            plt.plot(circle_x, circle_y, 'r--', alpha=0.5, linewidth=1)
        
        plt.title(f'RBF Network - Centros e Funções de Base\nσ = {self.sigma_computed:.4f}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('X1', fontsize=12)
        plt.ylabel('X2', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Gráfico RBF salvo em: {save_path}")
        
        plt.tight_layout()
        plt.close()  # Fechar figura para liberar memória
    
    def get_hyperparameters(self):
        # Retorna os hiperparâmetros do modelo
        return {
            'n_centers': self.n_centers,
            'sigma': self.sigma,
            'sigma_computed': self.sigma_computed,
            'random_state': self.random_state,
            'algorithm': 'RBF Network',
            'type': 'Radial Basis Function Network'
        }
    
    def get_centers_and_weights(self):
        # Retorna os centros e pesos aprendidos
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        return self.centers.copy(), self.weights.copy(), self.sigma_computed

def create_rbf_configurations():
    # Cria diferentes configurações de RBF para análise de under/overfitting
    configurations = {
        'Underfitted_RBF': {
            'n_centers': 2,
            'sigma': 2.0,
            'description': 'Poucos centros - propenso a underfitting'
        },
        'Balanced_RBF': {
            'n_centers': 10,
            'sigma': None,  # Será calculado automaticamente
            'description': 'Número moderado de centros - boa generalização'
        },
        'Overfitted_RBF': {
            'n_centers': 50,
            'sigma': 0.1,
            'description': 'Muitos centros com sigma pequeno - propenso a overfitting'
        }
    }
    
    print(" Configurações RBF criadas:")
    for name, config in configurations.items():
        sigma_str = f"σ={config['sigma']}" if config['sigma'] else "σ=auto"
        print(f"   • {name}: {config['n_centers']} centros, {sigma_str} - {config['description']}")
    
    return configurations

def create_standard_rbf():
    # Cria uma RBF com configuração padrão otimizada
    print(" Criando RBF padrão com hiperparâmetros otimizados...")
    print(" Hiperparâmetros escolhidos:")
    print("   • n_centers=10: Número moderado de centros")
    print("   • sigma=auto: Largura calculada automaticamente")
    print("   • Algoritmo: K-means + Pseudo-inversa")
    
    return RBF(n_centers=10, sigma=None, random_state=42)

def analyze_rbf_overfitting_underfitting(X_train, X_test, y_train, y_test):
    # Analisa underfitting e overfitting com diferentes configurações RBF
    print("\n ANÁLISE DE UNDERFITTING/OVERFITTING - RBF")
    print("=" * 60)
    
    configurations = create_rbf_configurations()
    results = {}
    
    for name, config in configurations.items():
        print(f"\n Treinando {name}...")
        print(f"   Centros: {config['n_centers']}")
        
        # Criar e treinar modelo
        rbf = RBF(
            n_centers=config['n_centers'],
            sigma=config['sigma'],
            random_state=42
        )
        
        rbf.fit(X_train, y_train, verbose=True)
        
        # Avaliar no conjunto de treino e teste
        y_pred_train = rbf.predict(X_train)
        y_pred_test = rbf.predict(X_test)
        
        train_metrics = calculate_all_metrics(y_train, y_pred_train)
        test_metrics = calculate_all_metrics(y_test, y_pred_test)
        
        # Armazenar resultados
        results[name] = {
            'model': rbf,
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

def test_rbf():
    # Função de teste para a RBF
    print(" Testando RBF...")
    
    # Dados sintéticos para teste
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, 
                              n_informative=2, random_state=42, n_clusters_per_class=2)
    y = np.where(y == 0, -1, 1)  # Converter para {-1, 1}
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Testar modelo padrão
    rbf = create_standard_rbf()
    rbf.fit(X_train, y_train, verbose=True)
    metrics = rbf.evaluate(X_test, y_test)
    
    # Plotar funções RBF
    rbf.plot_rbf_functions()
    
    print(" Teste concluído com sucesso!")
    return metrics

if __name__ == "__main__":
    # Executar teste se o arquivo for executado diretamente
    test_rbf()