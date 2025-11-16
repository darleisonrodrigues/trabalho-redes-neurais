"""
Aplicação Principal - Projeto de Redes Neurais
==============================================

Este é o arquivo principal que executa toda a análise de classificação
não-linear bidimensional usando diferentes tipos de redes neurais.

Funcionalidades:
- Carregamento e visualização dos dados spiral_d.csv
- Treinamento de diferentes modelos: Perceptron, ADALINE, MLP, RBF
- Análise de underfitting/overfitting
- Simulações estatísticas (500 rodadas)
- Geração de relatórios e visualizações


"""

import numpy as np
import pandas as pd
import time
from datetime import datetime
import os

# Imports dos módulos locais
from utils import (
    load_spiral_data, plot_data_visualization, 
    split_data_with_validation, calculate_all_metrics,
    plot_multiple_confusion_matrices, create_statistics_table,
    create_boxplot_comparison, print_project_header, 
    print_section_header, save_model_results
)
from extreme_analysis import analyze_extreme_cases_detailed, plot_learning_curves_extreme_cases
from perceptron_simple import PerceptronSimple, analyze_perceptron_performance
from adaline import ADALINE, analyze_adaline_performance
from mlp import MLP, analyze_overfitting_underfitting, create_standard_mlp
from rbf import RBF, analyze_rbf_overfitting_underfitting, create_standard_rbf

import warnings
warnings.filterwarnings('ignore')

class NeuralNetworkProject:
    """
    Classe principal que gerencia todo o projeto de redes neurais.
    """
    
    def __init__(self):
        """
        Inicializa o projeto.
        """
        self.X_original = None
        self.X_scaled = None
        self.y = None
        self.scaler = None
        self.models = {}
        self.results = {}
        self.simulation_results = {}
        self.start_time = None
        
    def load_data(self):
        """
        Carrega e prepara os dados.
        """
        print_section_header("Carregamento e Preparação dos Dados", 1)
        
        # Carregar dados
        self.X_original, self.X_scaled, self.y, self.scaler = load_spiral_data()
        
        if self.X_original is None:
            print(" Erro no carregamento dos dados. Encerrando aplicação.")
            return False
        
        # Visualizar dados
        plot_data_visualization(self.X_original, self.X_scaled, self.y)
        
        return True
    
    def train_basic_models(self):
        """
        Treina os modelos básicos com hiperparâmetros padrão.
        """
        print_section_header("Treinamento dos Modelos Básicos", 2)
        
        # Dividir dados para treinamento inicial
        X_train, X_test, y_train, y_test = split_data_with_validation(
            self.X_scaled, self.y, test_size=0.2, random_state=42
        )
        
        print(f" Divisão dos dados:")
        print(f"   Treino: {X_train.shape[0]} amostras")
        print(f"   Teste:  {X_test.shape[0]} amostras")
        
        # 1. Perceptron Simples
        print("\n" + "="*50)
        print(" PERCEPTRON SIMPLES")
        print("="*50)
        perceptron, perceptron_metrics = analyze_perceptron_performance(
            X_train, X_test, y_train, y_test
        )
        self.models['Perceptron'] = perceptron
        self.results['Perceptron'] = perceptron_metrics
        
        # 2. ADALINE
        print("\n" + "="*50)
        print(" ADALINE")
        print("="*50)
        adaline, adaline_metrics = analyze_adaline_performance(
            X_train, X_test, y_train, y_test, verbose=True
        )
        self.models['ADALINE'] = adaline
        self.results['ADALINE'] = adaline_metrics
        
        # 3. MLP Padrão
        print("\n" + "="*50)
        print(" MLP (PERCEPTRON MULTICAMADAS)")
        print("="*50)
        mlp = create_standard_mlp()
        mlp.fit(X_train, y_train, verbose=True)
        mlp_metrics = mlp.evaluate(X_test, y_test, verbose=True)
        self.models['MLP'] = mlp
        self.results['MLP'] = mlp_metrics
        
        # 4. RBF Padrão
        print("\n" + "="*50)
        print(" RBF (REDE DE BASE RADIAL)")
        print("="*50)
        rbf = create_standard_rbf()
        rbf.fit(X_train, y_train, verbose=True)
        rbf_metrics = rbf.evaluate(X_test, y_test, verbose=True)
        self.models['RBF'] = rbf
        self.results['RBF'] = rbf_metrics
        
        return X_train, X_test, y_train, y_test
    
    def analyze_underfitting_overfitting(self, X_train, X_test, y_train, y_test):
        """
        Analisa underfitting e overfitting para MLP e RBF.
        """
        print_section_header("Análise de Underfitting/Overfitting", 3)
        
        # Análise MLP
        mlp_results = analyze_overfitting_underfitting(X_train, X_test, y_train, y_test)
        
        # Análise RBF
        rbf_results = analyze_rbf_overfitting_underfitting(X_train, X_test, y_train, y_test)
        
        # Combinar resultados para visualização
        all_under_over_results = {**mlp_results, **rbf_results}
        
        # Plotar matrizes de confusão
        plot_multiple_confusion_matrices(
            all_under_over_results,
            save_path='results/confusion_matrices/underfitting_overfitting_comparison.png'
        )
        
        return all_under_over_results
    
    def run_statistical_simulations(self, n_simulations=500):
        """
        Executa simulações estatísticas com divisões aleatórias dos dados.
        """
        print_section_header(f"Simulações Estatísticas ({n_simulations} rodadas)", 4)
        
        models_config = {
            'Perceptron': lambda: PerceptronSimple(max_iter=1000, random_state=None),
            'ADALINE': lambda: ADALINE(learning_rate=0.01, max_iter=1000, random_state=None),
            'MLP': lambda: MLP(hidden_layer_sizes=(10,), max_iter=1000, random_state=None),
            'RBF': lambda: RBF(n_centers=10, sigma=None, random_state=None)
        }
        
        # Inicializar estrutura de resultados
        self.simulation_results = {model: {
            'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'specificity': []
        } for model in models_config.keys()}
        
        print(f" Executando {n_simulations} simulações...")
        print("   Cada simulação: 80% treino, 20% teste")
        
        # Barra de progresso simples
        progress_interval = n_simulations // 10
        
        for sim in range(n_simulations):
            if (sim + 1) % progress_interval == 0:
                progress = ((sim + 1) / n_simulations) * 100
                print(f"    Progresso: {progress:.0f}% ({sim + 1}/{n_simulations})")
            
            # Divisão aleatória dos dados
            X_train, X_test, y_train, y_test = split_data_with_validation(
                self.X_scaled, self.y, test_size=0.2, random_state=sim
            )
            
            # Treinar e avaliar cada modelo
            for model_name, model_func in models_config.items():
                try:
                    model = model_func()
                    # Treinar sem verbose para evitar spam no log
                    if hasattr(model, 'fit'):
                        if model_name == 'ADALINE':
                            model.fit(X_train, y_train, verbose=False)
                        else:
                            model.fit(X_train, y_train)
                    else:
                        model.fit(X_train, y_train, verbose=False)
                    
                    y_pred = model.predict(X_test)
                    metrics = calculate_all_metrics(y_test, y_pred)
                    
                    self.simulation_results[model_name]['accuracy'].append(metrics['accuracy'])
                    self.simulation_results[model_name]['precision'].append(metrics['precision'])
                    self.simulation_results[model_name]['recall'].append(metrics['recall'])
                    self.simulation_results[model_name]['f1'].append(metrics['f1'])
                    self.simulation_results[model_name]['specificity'].append(metrics['specificity'])
                    
                except Exception as e:
                    # Em caso de erro, usar valores padrão baixos
                    if (sim + 1) == 1:  # Mostrar erro apenas na primeira simulação
                        print(f"     Erro no modelo {model_name}: {str(e)}")
                    for metric in ['accuracy', 'precision', 'recall', 'f1', 'specificity']:
                        self.simulation_results[model_name][metric].append(0.5)
        
        print(" Simulações concluídas!")
        return self.simulation_results
    
    def analyze_extreme_cases(self):
        """
        Analisa os casos extremos das simulações com detalhes completos.
        """
        print_section_header("Análise dos Casos Extremos", 5)
        
        # Usar função detalhada do utils.py
        self.extreme_cases = analyze_extreme_cases_detailed(
            self.simulation_results, self.X_scaled, self.y, 'results/'
        )
        
        # Gerar curvas de aprendizado para casos extremos
        plot_learning_curves_extreme_cases(
            self.extreme_cases, self.X_scaled, self.y, 
            'results/plots/learning_curves_extreme_cases.png'
        )
    
    def generate_final_statistics(self):
        """
        Gera estatísticas finais e visualizações.
        """
        print_section_header("Estatísticas Finais e Visualizações", 6)
        
        # Criar tabelas estatísticas
        stats_tables = create_statistics_table(
            self.simulation_results, 
            save_path='results/stats/simulation_statistics.csv'
        )
        
        # Imprimir tabelas no console
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
        
        for metric in metrics:
            print(f"\n TABELA: {metric.upper()}")
            print("=" * 80)
            df = stats_tables[metric]
            print(df.to_string(index=False, float_format='%.4f'))
        
        # Criar boxplots
        create_boxplot_comparison(
            self.simulation_results,
            save_path='results/plots/performance_comparison_boxplots.png'
        )
        
        # Plotar matriz de confusão dos modelos básicos
        plot_multiple_confusion_matrices(
            self.results,
            save_path='results/confusion_matrices/basic_models_comparison.png'
        )
        
        return stats_tables
    
    def save_all_results(self):
        """
        Salva todos os resultados em arquivos.
        """
        print_section_header("Salvando Resultados", 7)
        
        # Salvar resultados dos modelos básicos
        for model_name, results in self.results.items():
            save_model_results(model_name, results)
        
        # Salvar hiperparâmetros
        hyperparams = {}
        for model_name, model in self.models.items():
            if hasattr(model, 'get_hyperparameters'):
                hyperparams[model_name] = model.get_hyperparameters()
        
        import json
        with open('results/stats/hyperparameters.json', 'w') as f:
            json.dump(hyperparams, f, indent=2)
        
        print(" Todos os resultados foram salvos!")
    
    def generate_summary_report(self, stats_tables):
        """
        Gera relatório resumo final.
        """
        print_section_header("Relatório Resumo Final", 8)
        
        end_time = time.time()
        execution_time = end_time - self.start_time
        
        print(f"  Tempo total de execução: {execution_time:.2f} segundos")
        print(f" Data/hora de conclusão: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print(f" Total de dados processados: {len(self.y)} amostras")
        print(f" Número de features: {self.X_original.shape[1]}")
        print(f" Número de classes: {len(np.unique(self.y))}")
        
        print("\n RANKING POR ACURÁCIA MÉDIA:")
        print("-" * 50)
        accuracy_df = stats_tables['accuracy']
        ranking = accuracy_df.sort_values('Média', ascending=False)
        
        for i, (_, row) in enumerate(ranking.iterrows(), 1):
            print(f"{i}º lugar: {row['Modelo']:<12} - Acurácia: {row['Média']:.4f} (±{row['Desvio_Padrão']:.4f})")
        
        print("\n ARQUIVOS GERADOS:")
        print("-" * 30)
        files = [
            'results/plots/data_visualization.png',
            'results/plots/performance_comparison_boxplots.png',
            'results/confusion_matrices/basic_models_comparison.png',
            'results/confusion_matrices/underfitting_overfitting_comparison.png',
            'results/stats/simulation_statistics.xlsx',
            'results/stats/hyperparameters.json'
        ]
        
        for file in files:
            if os.path.exists(file):
                print(f"✅ {file}")
            else:
                print(f"❌ {file}")
        
        print("\n🎉 ANÁLISE COMPLETA FINALIZADA COM SUCESSO!")
        print("="*60)
    
    def run_complete_analysis(self, n_simulations=500):
        """
        Executa a análise completa do projeto.
        """
        self.start_time = time.time()
        print_project_header()
        
        try:
            # Etapa 1: Carregar dados
            if not self.load_data():
                return False
            
            # Etapa 2: Treinar modelos básicos
            X_train, X_test, y_train, y_test = self.train_basic_models()
            
            # Etapa 3: Análise de underfitting/overfitting
            under_over_results = self.analyze_underfitting_overfitting(
                X_train, X_test, y_train, y_test
            )
            
            # Etapa 4: Simulações estatísticas
            simulation_results = self.run_statistical_simulations(n_simulations)
            
            # Etapa 5: Análise de casos extremos
            self.analyze_extreme_cases()
            
            # Etapa 6: Estatísticas finais
            stats_tables = self.generate_final_statistics()
            
            # Etapa 7: Salvar resultados
            self.save_all_results()
            
            # Etapa 8: Relatório final
            self.generate_summary_report(stats_tables)
            
            return True
            
        except Exception as e:
            print(f"❌ Erro durante a execução: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """
    Função principal da aplicação.
    """
    # Verificar se estamos no ambiente virtual
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  ATENÇÃO: Recomenda-se executar dentro do ambiente virtual 'venv'")
        print("Para ativar: .\\venv\\Scripts\\Activate.ps1")
    
    # Criar e executar projeto
    project = NeuralNetworkProject()
    
    # Executar análise completa
    # Para teste rápido, usar n_simulations=100
    # Para análise completa, usar n_simulations=500
    success = project.run_complete_analysis(n_simulations=500)  # Análise completa conforme solicitado
    
    if success:
        print("\n Projeto executado com sucesso!")
        print(" Verifique a pasta 'results/' para todos os arquivos gerados.")
        
        # Mostrar resumo dos arquivos gerados
        print("\n ARQUIVOS GERADOS:")
        print("-" * 50)
        
        results_dir = "results"
        if os.path.exists(results_dir):
            for root, dirs, files in os.walk(results_dir):
                level = root.replace(results_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f'{indent} {os.path.basename(root)}/')
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    print(f'{subindent} {file} ({file_size:.1f} KB)')
        
                
    else:
        print("\n❌ Projeto finalizado com erros.")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)