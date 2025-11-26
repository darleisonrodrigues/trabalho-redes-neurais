# Implementação do ADALINE (Adaptive Linear Neuron) para classificação binária

import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_all_metrics, print_metrics_summary

class ADALINE:
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tolerance=1e-6, random_state=42):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state
        
        # Atributos a serem definidos durante o treinamento
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.is_fitted = False
        self.converged = False
        self.n_iter_actual = 0
        
    def _net_input(self, X):
        # Calcula a entrada líquida (net input)
        return np.dot(X, self.weights) + self.bias
    
    def _activation(self, X):
        # Função de ativação linear (identidade)
        return self._net_input(X)
    
    def fit(self, X_train, y_train, verbose=False):
        # Treina o modelo ADALINE
        if verbose:
            print(" Treinando ADALINE...")
        
        # Configurar seed
        np.random.seed(self.random_state)
        
        # Inicializar pesos e bias
        n_features = X_train.shape[1]
        self.weights = np.random.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias = 0.0
        
        self.cost_history = []
        prev_cost = float('inf')
        
        # Algoritmo de treinamento
        for i in range(self.max_iter):
            # Forward pass
            net_input = self._net_input(X_train)
            output = self._activation(X_train)
            
            # Calcular erro e custo
            errors = y_train - output
            cost = (errors**2).sum() / (2.0 * len(y_train))
            self.cost_history.append(cost)
            
            # Verificar convergência
            if abs(prev_cost - cost) < self.tolerance:
                self.converged = True
                if verbose:
                    print(f" Convergência atingida na iteração {i+1}")
                break
                
            # Backward pass (atualização dos pesos)
            self.weights += self.learning_rate * X_train.T.dot(errors) / len(y_train)
            self.bias += self.learning_rate * errors.sum() / len(y_train)
            
            prev_cost = cost
            self.n_iter_actual = i + 1
            
            # Mostrar progresso
            if verbose and (i + 1) % 100 == 0:
                print(f"Iteração {i+1}: Custo = {cost:.6f}")
        
        self.is_fitted = True
        
        if verbose:
            if not self.converged:
                print(f"  Máximo de iterações ({self.max_iter}) atingido")
            print(f" ADALINE treinado! Custo final: {self.cost_history[-1]:.6f}")
        
        return self
    
    def predict(self, X_test):
        # Faz predições usando o modelo treinado
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        return np.where(self._net_input(X_test) >= 0.0, 1, -1)
    
    def predict_proba(self, X_test):
        # Retorna as saídas lineares (antes da função degrau)
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        return self._activation(X_test)
    
    def evaluate(self, X_test, y_test, verbose=True):
        # Avalia o desempenho do modelo
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        y_pred = self.predict(X_test)
        metrics = calculate_all_metrics(y_test, y_pred)
        
        if verbose:
            print_metrics_summary(metrics, "ADALINE")
        
        return metrics
    
    def plot_cost_history(self, save_path=None):
        # Plota a evolução do custo durante o treinamento
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.cost_history) + 1), self.cost_history, 
                'b-', linewidth=2, markersize=6)
        plt.title('ADALINE - Evolução do Custo Durante o Treinamento', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Iterações', fontsize=12)
        plt.ylabel('Custo (MSE)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Escala logarítmica para melhor visualização
        
        if self.converged:
            plt.axvline(x=self.n_iter_actual, color='red', linestyle='--', 
                       label=f'Convergência (iter {self.n_iter_actual})')
            plt.legend()
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Gráfico de custo salvo em: {save_path}")
        
        plt.tight_layout()
        plt.close()  # Fechar figura para liberar memória
    
    def get_hyperparameters(self):
        # Retorna os hiperparâmetros do modelo
        return {
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'tolerance': self.tolerance,
            'random_state': self.random_state,
            'algorithm': 'ADALINE',
            'type': 'Linear Adaptive Classifier'
        }
    
    def get_weights(self):
        # Retorna os pesos aprendidos pelo modelo
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        return self.weights.copy(), self.bias

def create_adaline_with_hyperparameters():
    # Cria um ADALINE com hiperparâmetros otimizados
    print(" Criando ADALINE com hiperparâmetros otimizados...")
    print(" Hiperparâmetros escolhidos:")
    print("   • learning_rate=0.01: Taxa de aprendizado moderada")
    print("   • max_iter=1000: Número máximo de iterações")
    print("   • tolerance=1e-6: Critério de convergência do custo")
    print("   • Algoritmo: Gradiente descendente com MSE")
    
    return ADALINE(learning_rate=0.01, max_iter=1000, tolerance=1e-6, random_state=42)

def analyze_adaline_performance(X_train, X_test, y_train, y_test, verbose=True):
    # Análise completa do desempenho do ADALINE
    # Criar e treinar modelo
    adaline = create_adaline_with_hyperparameters()
    adaline.fit(X_train, y_train, verbose=verbose)
    
    # Avaliar modelo
    metrics = adaline.evaluate(X_test, y_test, verbose=verbose)
    
    # Informações adicionais sobre treinamento
    if verbose:
        print(f" Número de iterações realizadas: {adaline.n_iter_actual}")
        print(f" Convergência: {'Sim' if adaline.converged else 'Não'}")
        print(f" Custo final: {adaline.cost_history[-1]:.6f}")
    
    # Mostrar pesos aprendidos
    weights, bias = adaline.get_weights()
    if verbose:
        print(f"  Pesos aprendidos: {weights}")
        print(f" Bias: {bias:.6f}")
    
    return adaline, metrics

def test_adaline():
    # Função de teste para o ADALINE
    print(" Testando ADALINE...")
    
    # Dados sintéticos para teste
    from sklearn.model_selection import train_test_split
    
    np.random.seed(42)
    X_test = np.random.randn(200, 2)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int) * 2 - 1  # Converter para {-1, 1}
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_test, y_test, test_size=0.3, random_state=42
    )
    
    # Testar modelo
    adaline = ADALINE(learning_rate=0.01, max_iter=500)
    adaline.fit(X_train, y_train, verbose=True)
    metrics = adaline.evaluate(X_val, y_val)
    
    # Plotar evolução do custo
    adaline.plot_cost_history()
    
    print(" Teste concluído com sucesso!")
    return metrics

if __name__ == "__main__":
    # Executar teste se o arquivo for executado diretamente
    test_adaline()