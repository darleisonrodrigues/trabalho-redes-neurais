# Implementação do Perceptron Simples para classificação binária

import numpy as np
from sklearn.linear_model import Perceptron
from utils import calculate_all_metrics, print_metrics_summary

class PerceptronSimple:
    
    def __init__(self, max_iter=1000, random_state=42):
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        
    def fit(self, X_train, y_train):
        # Treina o modelo Perceptron
        print(" Treinando Perceptron Simples...")
        
        self.model = Perceptron(
            max_iter=self.max_iter,
            random_state=self.random_state,
            fit_intercept=True,
            shuffle=True
        )
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        print(" Perceptron Simples treinado com sucesso!")
        return self
    
    def predict(self, X_test):
        # Faz predições usando o modelo treinado
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test, verbose=True):
        # Avalia o desempenho do modelo
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        y_pred = self.predict(X_test)
        metrics = calculate_all_metrics(y_test, y_pred)
        
        if verbose:
            print_metrics_summary(metrics, "Perceptron Simples")
        
        return metrics
    
    def get_hyperparameters(self):
        # Retorna os hiperparâmetros do modelo
        return {
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'algorithm': 'Perceptron Simples',
            'type': 'Linear Classifier'
        }
    
    def get_weights(self):
        # Retorna os pesos aprendidos pelo modelo
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        return self.model.coef_[0], self.model.intercept_[0]
    
    def decision_function(self, X):
        # Calcula a função de decisão para os dados de entrada
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        return self.model.decision_function(X)

def create_perceptron_with_hyperparameters():
    # Cria um Perceptron com hiperparâmetros otimizados
    print(" Criando Perceptron Simples com hiperparâmetros otimizados...")
    print(" Hiperparâmetros escolhidos:")
    print("   • max_iter=1000: Número suficiente para convergência")
    print("   • random_state=42: Garantir reprodutibilidade")
    print("   • Algoritmo: Perceptron clássico de Rosenblatt")
    
    return PerceptronSimple(max_iter=1000, random_state=42)

def analyze_perceptron_performance(X_train, X_test, y_train, y_test):
    # Análise completa do desempenho do Perceptron
    # Criar e treinar modelo
    perceptron = create_perceptron_with_hyperparameters()
    perceptron.fit(X_train, y_train)
    
    # Avaliar modelo
    metrics = perceptron.evaluate(X_test, y_test, verbose=True)
    
    # Informações adicionais sobre convergência
    if hasattr(perceptron.model, 'n_iter_'):
        print(f" Número de iterações para convergência: {perceptron.model.n_iter_}")
    
    # Mostrar pesos aprendidos
    weights, bias = perceptron.get_weights()
    print(f"  Pesos aprendidos: {weights}")
    print(f" Bias: {bias:.4f}")
    
    return perceptron, metrics

def test_perceptron():
    # Função de teste para o Perceptron Simples
    print(" Testando Perceptron Simples...")
    
    # Dados sintéticos para teste
    np.random.seed(42)
    X_test = np.random.randn(100, 2)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int) * 2 - 1  # Converter para {-1, 1}
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_test, y_test, test_size=0.3, random_state=42
    )
    
    # Testar modelo
    perceptron = PerceptronSimple()
    perceptron.fit(X_train, y_train)
    metrics = perceptron.evaluate(X_val, y_val)
    
    print(" Teste concluído com sucesso!")
    return metrics

if __name__ == "__main__":
    # Executar teste se o arquivo for executado diretamente
    from sklearn.model_selection import train_test_split
    test_perceptron()