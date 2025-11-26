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
│       ├── plots/               # Gráficos e visualizações (4 arquivos)
│       └── stats/               # Estatísticas em Excel/JSON (6 arquivos)
│
├── etapa2_series_temporais/       # SEGUNDA ETAPA - Séries Temporais
│   ├── lstm_furnas.py            # Arquivo principal da etapa 2
│   ├── furnas.csv                # Dataset vazão Furnas (60 anos)
│   ├── plots/                    # Visualizações da etapa 2 (3 arquivos)
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

### **Objetivos**

1. **Classificação Não Linear Bidimensional** usando diferentes arquiteturas de redes neurais
2. **Análise de Underfitting/Overfitting** com diferentes topologias
3. **Validação Estatística** através de 500 simulações com divisão aleatória dos dados
4. **Comparação de Desempenho** entre diferentes algoritmos

### **Como Executar a Etapa 1**

#### 1. Ativar o ambiente virtual
```powershell
.\venv\Scripts\Activate.ps1
```

#### 2. Navegar para a pasta da etapa 1
```powershell
cd etapa1_classificacao
```

#### 3. Executar o projeto completo
```powershell
python app.py
```

#### 4. Executar módulos individuais (para testes)
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

### **Modelos Implementados**

#### 1. **Perceptron Simples** (`perceptron_simple.py`)
- **Algoritmo**: Perceptron clássico de Rosenblatt
- **Características**: Classificador linear, adequado para dados linearmente separáveis
- **Hiperparâmetros**: `max_iter=1000`, `random_state=42`

#### 2. **ADALINE** (`adaline.py`)
- **Algoritmo**: Adaptive Linear Neuron
- **Características**: Utiliza gradiente descendente e função de custo MSE
- **Hiperparâmetros**: `learning_rate=0.01`, `max_iter=1000`, `tolerance=1e-6`

#### 3. **MLP** (`mlp.py`)
- **Algoritmo**: Multi-Layer Perceptron com backpropagation
- **Características**: Múltiplas camadas, funções de ativação não-lineares
- **Configurações**:
  - **Underfitted**: `(2,)` neurônios, 50 iterações
  - **Balanced**: `(10,)` neurônios, 1000 iterações  
  - **Overfitted**: `(100,50,25)` neurônios, 2000 iterações

#### 4. **RBF** (`rbf.py`)
- **Algoritmo**: Radial Basis Function Network
- **Características**: Funções gaussianas, centros determinados por K-means
- **Configurações**:
  - **Underfitted**: 2 centros, σ=2.0
  - **Balanced**: 10 centros, σ=auto
  - **Overfitted**: 50 centros, σ=0.1

### **Dataset - `spiral_d.csv`**
- **Formato**: CSV sem cabeçalho
- **Estrutura**: 3 colunas (X1, X2, Class)
- **Amostras**: 1.400 registros
- **Classes**: {1.0, -1.0}
- **Distribuição**: 1000 amostras da classe 1.0, 400 da classe -1.0
- **Características**: Dados em formato espiral (não linearmente separável)

---

## 📈 **ETAPA 2: PREVISÃO DE SÉRIES TEMPORAIS**

### **Descrição do Projeto**

Esta etapa implementa redes LSTM (Long Short-Term Memory) para previsão de séries temporais usando dados históricos de vazão da Usina Hidrelétrica de Furnas. O objetivo é prever valores futuros baseado em padrões temporais dos últimos 60 anos.

### **Objetivos**

1. **Análise Exploratória** de dados de séries temporais hidrológicas
2. **Implementação de LSTM** para previsão de vazão hidrelétrica
3. **Avaliação de Performance** usando métricas específicas para regressão
4. **Visualização de Resultados** comparando predições vs valores reais

### **Como Executar a Etapa 2**

#### 1. Ativar o ambiente virtual (se não estiver ativo)
```powershell
.\venv\Scripts\Activate.ps1
```

#### 2. Navegar para a pasta da etapa 2
```powershell
cd etapa2_series_temporais
```

#### 3. Executar a análise LSTM
```powershell
python lstm_furnas.py
```

### **Modelo Implementado**

#### **LSTM** (`lstm_furnas.py`)
- **Arquitetura**: 50 neurônios LSTM → Dropout(0.2) → Dense(12)
- **Configuração**: Janela de 12 meses para predizer próximos 12 meses
- **Divisão**: 48 anos para treino + 12 anos para teste
- **Otimizador**: Adam com early stopping

### **Dataset - `furnas.csv`**
- **Formato**: CSV com dados mensais
- **Período**: 60 anos (708 observações mensais)
- **Variável**: Vazão hidrelétrica (m³/s)
- **Características**: Sazonalidade clara com variabilidade inter-anual

---

## 📊 **ANÁLISES REALIZADAS**

### **Etapa 1:**

#### 1. **Visualização dos Dados**
- Gráfico de espalhamento dos dados originais e normalizados
- Coloração por classe para visualizar padrão não-linear

#### 2. **Treinamento dos Modelos**
- Justificativa dos hiperparâmetros escolhidos
- Métricas de desempenho: Acurácia, Precisão, Recall, F1-Score, Especificidade
- Matrizes de confusão para cada modelo

#### 3. **Análise de Underfitting/Overfitting**
- Diferentes topologias para MLP e RBF
- Comparação entre acurácia de treino vs teste
- Identificação de casos de sub/superdimensionamento

#### 4. **Validação Estatística (500 Simulações)**
- Divisão aleatória: 70% treino, 30% teste
- Cálculo de estatísticas: média, desvio-padrão, maior/menor valor
- Análise de casos extremos (melhor e pior performance)

#### 5. **Visualizações Geradas**
- Boxplots comparativos entre modelos
- Matrizes de confusão múltiplas
- Curvas de aprendizado para casos extremos

### **Etapa 2:**

#### 1. **Análise Exploratória Temporal**
- Série temporal completa (60 anos de dados)
- Identificação de padrões sazonais
- Distribuições mensais via boxplots
- Análise de tendências de longo prazo

#### 2. **Modelagem LSTM**
- Preparação de dados temporais com janela deslizante
- Normalização dos dados para o treinamento
- Implementação de arquitetura LSTM com dropout
- Early stopping para evitar overfitting

#### 3. **Avaliação e Visualização**
- Métricas de regressão: MAE, RMSE, R²
- Comparação visual entre predições e valores reais
- Análise de correlação e dispersão dos resultados

---


## 📏 **MÉTRICAS DE AVALIAÇÃO**

### **Etapa 1 - Classificação:**
- **Acurácia**: (TP + TN) / (TP + TN + FP + FN)
- **Precisão**: TP / (TP + FP)  
- **Recall (Sensibilidade)**: TP / (TP + FN)
- **Especificidade**: TN / (TN + FP)
- **F1-Score**: 2 × (Precisão × Recall) / (Precisão + Recall)

### **Etapa 2 - Regressão:**
- **MAE**: Mean Absolute Error (Erro Absoluto Médio)
- **RMSE**: Root Mean Square Error (Raiz do Erro Quadrático Médio)
- **R²**: Coeficiente de Determinação

---

## 🔧 **DEPENDÊNCIAS**

O projeto utiliza as seguintes bibliotecas Python:

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tensorflow>=2.20.0     # Para LSTM (Etapa 2)
openpyxl>=3.0.0        # Para arquivos Excel
```

Todas as dependências são instaladas automaticamente no ambiente virtual.

---

## 📝 **OBSERVAÇÕES IMPORTANTES**

1. **Ambiente Virtual**: Sempre execute dentro do ambiente virtual `venv`
2. **Tempo de Execução**: A análise completa da Etapa 1 (500 simulações) pode levar alguns minutos
3. **Teste Rápido**: Para teste da Etapa 1, altere `n_simulations=100` no `app.py`
4. **Reprodutibilidade**: Seeds aleatórias garantem resultados reproduzíveis
5. **Logs**: Ambos os programas fornecem logs detalhados durante a execução

---

## ✅ **STATUS DO PROJETO**

- [x] **Etapa 1** - Classificação não linear ✅ **CONCLUÍDA**
- [x] **Etapa 2** - Séries temporais LSTM ✅ **CONCLUÍDA**  
- [x] **Validação estatística** ✅ **500 simulações (Etapa 1)**
- [x] **Organização do código** ✅ **Estruturada em pastas**
- [x] **Documentação completa** ✅ **README atualizado**

---

## 🎓 **EXECUÇÃO RÁPIDA**

**Para executar todo o projeto:**

```powershell
# Ativar ambiente virtual
.\venv\Scripts\Activate.ps1

# Executar Etapa 1 (Classificação)
cd etapa1_classificacao
python app.py
cd ..

# Executar Etapa 2 (Séries Temporais)  
cd etapa2_series_temporais
python lstm_furnas.py
cd ..
```

**Tempo estimado:** ~10-15 minutos para execução completa de ambas as etapas
```

