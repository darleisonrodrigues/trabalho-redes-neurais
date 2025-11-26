# TRABALHO VIVENCIAL - REDES NEURAIS
## Classificação Não Linear e Previsão de Séries Temporais

### 📁 **ESTRUTURA DO PROJETO**

```
trabalho-redes-neurais/
├── etapa1_classificacao/           # PRIMEIRA ETAPA - Classificação
│   ├── app.py                     # Arquivo principal da etapa 1
│   ├── spiral_d.csv              # Dataset espiral
│   ├── perceptron_simple.py      # Perceptron Simples
│   ├── adaline.py                # ADALINE
│   ├── mlp.py                    # MLP (Multi-Layer Perceptron)
│   ├── rbf.py                    # RBF (Radial Basis Function)
│   ├── utils.py                  # Funções auxiliares
│   ├── extreme_analysis.py       # Análise de casos extremos
│   └── results/                  # Resultados da etapa 1
│       ├── confusion_matrices/   # Matrizes de confusão (42 arquivos)
│       ├── plots/               # Gráficos e visualizações (3 arquivos)
│       └── stats/               # Estatísticas em Excel/JSON (6 arquivos)
│
├── etapa2_series_temporais/       # SEGUNDA ETAPA - Séries Temporais
│   ├── lstm_furnas.py            # Arquivo principal da etapa 2
│   ├── furnas.csv                # Dataset vazão Furnas (60 anos)
│   ├── plots/                    # Visualizações da etapa 2
│   ├── models/                   # Modelos LSTM salvos
│   └── results/                  # Resultados da etapa 2
│
├── venv/                         # Ambiente virtual Python 3.11
├── requirements.txt              # Dependências do projeto
├── .gitignore                    # Arquivos ignorados pelo git
└── README.md                     # Este arquivo
```

## 🎯 **ETAPA 1: CLASSIFICAÇÃO NÃO LINEAR**

### **Descrição do Projeto**

Este projeto implementa e analisa diferentes tipos de redes neurais para classificação não linear bidimensional usando o dataset `spiral_d.csv`. O projeto é parte da primeira etapa de um trabalho acadêmico sobre Machine Learning e Redes Neurais.

##  Objetivos

1. **Classificação Não Linear Bidimensional** usando diferentes arquiteturas de redes neurais
2. **Análise de Underfitting/Overfitting** com diferentes topologias
3. **Validação Estatística** através de 500 simulações com divisão aleatória dos dados
4. **Comparação de Desempenho** entre diferentes algoritmos

## Estrutura do Projeto

```
RedesNeurais/
├── app.py                    # Arquivo principal para executar o projeto
├── spiral_d.csv             # Dataset com dados em espiral (2 features, 2 classes)
├── furnas.csv               # Dataset para segunda etapa (séries temporais)
├── utils.py                 # Funções auxiliares (métricas, visualizações, etc.)
├── perceptron_simple.py     # Implementação do Perceptron Simples
├── adaline.py               # Implementação do ADALINE
├── mlp.py                   # Implementação do MLP (Multi-Layer Perceptron)
├── rbf.py                   # Implementação da RBF (Radial Basis Function)
├── venv/                    # Ambiente virtual Python
└── results/                 # Pasta com todos os resultados gerados
    ├── confusion_matrices/  # Matrizes de confusão
    ├── plots/              # Gráficos e visualizações
    └── stats/              # Tabelas estatísticas e métricas
```

## Como Executar

### 1. Ativar o ambiente virtual
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Executar o projeto completo
```powershell
python app.py
```

### 3. Executar módulos individuais (para testes)
```powershell
# Testar Perceptron Simples
python perceptron_simple.py

# Testar ADALINE
python adaline.py

# Testar MLP
python mlp.py

# Testar RBF
python rbf.py
```

##  Modelos Implementados

### 1. **Perceptron Simples** (`perceptron_simple.py`)
- **Algoritmo**: Perceptron clássico de Rosenblatt
- **Características**: Classificador linear, adequado para dados linearmente separáveis
- **Hiperparâmetros**: `max_iter=1000`, `random_state=42`

### 2. **ADALINE** (`adaline.py`)
- **Algoritmo**: Adaptive Linear Neuron
- **Características**: Utiliza gradiente descendente e função de custo MSE
- **Hiperparâmetros**: `learning_rate=0.01`, `max_iter=1000`, `tolerance=1e-6`

### 3. **MLP** (`mlp.py`)
- **Algoritmo**: Multi-Layer Perceptron com backpropagation
- **Características**: Múltiplas camadas, funções de ativação não-lineares
- **Configurações**:
  - **Underfitted**: `(2,)` neurônios, 50 iterações
  - **Balanced**: `(10,)` neurônios, 1000 iterações  
  - **Overfitted**: `(100,50,25)` neurônios, 2000 iterações

### 4. **RBF** (`rbf.py`)
- **Algoritmo**: Radial Basis Function Network
- **Características**: Funções gaussianas, centros determinados por K-means
- **Configurações**:
  - **Underfitted**: 2 centros, σ=2.0
  - **Balanced**: 10 centros, σ=auto
  - **Overfitted**: 50 centros, σ=0.1

## Dataset

### `spiral_d.csv`
- **Formato**: CSV sem cabeçalho
- **Estrutura**: 3 colunas (X1, X2, Class)
- **Amostras**: 1.400 registros
- **Classes**: {1.0, -1.0}
- **Distribuição**: 1000 amostras da classe 1.0, 400 da classe -1.0
- **Características**: Dados em formato espiral (não linearmente separável)

##  Análises Realizadas

### 1. **Visualização dos Dados**
- Gráfico de espalhamento dos dados originais e normalizados
- Coloração por classe para visualizar padrão não-linear

### 2. **Treinamento dos Modelos**
- Justificativa dos hiperparâmetros escolhidos
- Métricas de desempenho: Acurácia, Precisão, Recall, F1-Score, Especificidade
- Matrizes de confusão para cada modelo

### 3. **Análise de Underfitting/Overfitting**
- Diferentes topologias para MLP e RBF
- Comparação entre acurácia de treino vs teste
- Identificação de casos de sub/superdimensionamento

### 4. **Validação Estatística (500 Simulações)**
- Divisão aleatória: 80% treino, 20% teste
- Cálculo de estatísticas: média, desvio-padrão, maior/menor valor
- Análise de casos extremos (melhor e pior performance)

### 5. **Visualizações Geradas**
- Boxplots comparativos entre modelos
- Matrizes de confusão múltiplas
- Curvas de aprendizado (para MLP)
- Visualização das funções RBF

##  Métricas de Avaliação

Todas as métricas são calculadas para cada modelo:

- **Acurácia**: (TP + TN) / (TP + TN + FP + FN)
- **Precisão**: TP / (TP + FP)  
- **Recall (Sensibilidade)**: TP / (TP + FN)
- **Especificidade**: TN / (TN + FP)
- **F1-Score**: 2 × (Precisão × Recall) / (Precisão + Recall)

## Arquivos Gerados

Após a execução, os seguintes arquivos são gerados em `results/`:

### Visualizações (`plots/`)
- `data_visualization.png` - Visualização inicial dos dados
- `performance_comparison_boxplots.png` - Comparação de desempenho
- `adaline_cost_curve.png` - Curva de custo do ADALINE (se executado individualmente)
- `mlp_learning_curve.png` - Curva de aprendizado do MLP

### Matrizes de Confusão (`confusion_matrices/`)
- `basic_models_comparison.png` - Comparação dos modelos básicos
- `underfitting_overfitting_comparison.png` - Análise de under/overfitting

### Estatísticas (`stats/`)
- `simulation_statistics.xlsx` - Planilha com estatísticas das 500 simulações
- `hyperparameters.json` - Hiperparâmetros utilizados
- `*_results.json` - Resultados individuais de cada modelo

## 🔧 Dependências

O projeto utiliza as seguintes bibliotecas Python:

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

Todas as dependências são instaladas automaticamente no ambiente virtual.

## 📝 Observações Importantes

1. **Ambiente Virtual**: Sempre execute dentro do ambiente virtual `venv`
2. **Tempo de Execução**: A análise completa (500 simulações) pode levar alguns minutos
3. **Teste Rápido**: Para teste, altere `n_simulations=100` no `app.py`
4. **Reprodutibilidade**: Seeds aleatórias garantem resultados reproduzíveis
5. **Logs**: O programa fornece logs detalhados durante a execução

##  Próximos Passos

Esta é a **Primeira Etapa** do projeto. A segunda etapa incluirá:
- **Previsão de Séries Temporais** usando `furnas.csv`
- Implementação de redes neurais para dados temporais
- Análise de tendências e sazonalidade


**Para executar:** `python app.py` (dentro do ambiente virtual)