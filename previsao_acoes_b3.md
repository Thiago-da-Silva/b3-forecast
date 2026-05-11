# Previsão de Ações da B3: LSTM vs. SARIMAX

Estudo comparativo entre redes neurais LSTM e modelos SARIMAX para previsão de preços de fechamento de ativos da Bolsa de Valores do Brasil (B3), com dashboard interativo em Python.

---

## Sumário

- [Visão Geral](#visão-geral)
- [Ativos Analisados](#ativos-analisados)
- [Stack Tecnológica](#stack-tecnológica)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Roadmap de Desenvolvimento](#roadmap-de-desenvolvimento)
- [Modelos Implementados](#modelos-implementados)
- [Métricas de Avaliação](#métricas-de-avaliação)
- [Dashboard](#dashboard)
- [Referências](#referências)

---

## Visão Geral

O mercado de ações é caracterizado por alta volatilidade, ruído e não-linearidade, tornando a previsão de preços um dos problemas mais desafiadores em ciência de dados. Este projeto propõe a implementação e comparação estruturada de dois paradigmas preditivos:

- **SARIMAX** — modelo estatístico clássico, transparente e eficaz para componentes lineares e sazonais, utilizado como *baseline*.
- **LSTM** *(Long Short-Term Memory)* — arquitetura de aprendizado profundo capaz de capturar dependências temporais de longo prazo e padrões não-lineares.
- **Modelo Híbrido** *(sugestão)* — SARIMAX para a componente linear + LSTM para modelar o resíduo, combinando os pontos fortes de ambas as abordagens.

O objetivo central é determinar qual abordagem apresenta menor taxa de erro preditivo em diferentes dinâmicas de mercado, avaliadas sobre três ativos de setores distintos.

---

## Ativos Analisados

| Ticker | Empresa | Setor | Característica |
|--------|---------|-------|----------------|
| `PETR4` | Petrobras | Energia / Commodities | Alta sensibilidade a variáveis externas (câmbio, petróleo) |
| `VALE3` | Vale | Mineração | Correlacionado a preços de minério e mercado chinês |
| `ITUB4` | Itaú Unibanco | Financeiro | Ciclos econômicos domésticos, maior estabilidade relativa |

Período de coleta: **mínimo 5 anos** (recomendado 2019–2024), cobrindo ciclos de mercado variados, incluindo a pandemia de COVID-19 e o ciclo de alta da taxa Selic.

---

## Stack Tecnológica

```
Coleta de dados   →  yfinance
Manipulação       →  pandas, numpy
Modelo estatístico →  pmdarima (auto_arima)
Modelo deep learning → TensorFlow / Keras
Tuning de hiperparâmetros → Optuna
Pré-processamento / métricas → scikit-learn
Visualização      →  Plotly Express
Dashboard         →  Streamlit
Avaliação estatística → arch (teste Diebold-Mariano)
```

---

## Estrutura do Projeto

```
b3forecast/
│
├── src/                               # Todo o código executável do projeto
│   ├── pipeline.py                    # Orquestrador — ponto de entrada único
│   ├── collect.py                     # Etapa 1: coleta de dados via yfinance
│   ├── preprocess.py                  # Etapa 2: limpeza, features, normalização
│   ├── train_sarimax.py               # Etapa 3: treinamento e serialização do SARIMAX
│   ├── train_lstm.py                  # Etapa 4: arquitetura, tuning e treinamento do LSTM
│   ├── train_hybrid.py                # Etapa 5: modelo híbrido (resíduo SARIMAX → LSTM)
│   └── evaluate.py                    # Etapa 6: métricas, Diebold-Mariano, exportação
│
├── data/
│   ├── raw/                           # Gerado pela Etapa 1
│   │   ├── PETR4_raw.csv             # OHLCV bruto — Petrobras
│   │   ├── VALE3_raw.csv             # OHLCV bruto — Vale
│   │   ├── ITUB4_raw.csv             # OHLCV bruto — Itaú
│   │   ├── IBOV_raw.csv              # Índice Bovespa (variável exógena)
│   │   ├── USDBRL_raw.csv            # Câmbio USD/BRL (variável exógena)
│   │   └── SELIC_raw.csv             # Taxa Selic diária (variável exógena)
│   │
│   └── processed/                     # Gerado pela Etapa 2
│       ├── PETR4_processed.csv        # Retornos log + features técnicas + normalização
│       ├── VALE3_processed.csv
│       └── ITUB4_processed.csv
│
├── artifacts/                         # Gerado pelas Etapas 3, 4 e 5
│   ├── PETR4_sarimax.pkl             # Modelo SARIMAX serializado — Petrobras
│   ├── VALE3_sarimax.pkl
│   ├── ITUB4_sarimax.pkl
│   ├── PETR4_lstm.keras              # Pesos da rede LSTM treinada — Petrobras
│   ├── VALE3_lstm.keras
│   ├── ITUB4_lstm.keras
│   └── lstm_best_params.json         # Melhores hiperparâmetros encontrados pelo Optuna
│
├── evaluation/                        # Gerado pela Etapa 6
│   ├── metrics.csv                    # Tabela consolidada: modelo × ativo × métrica
│   ├── predictions/
│   │   ├── PETR4_predictions.csv     # Colunas: data, real, sarimax, lstm, hybrid
│   │   ├── VALE3_predictions.csv
│   │   └── ITUB4_predictions.csv
│   └── residuals/
│       ├── PETR4_residuals.csv        # Resíduos por modelo — input do dashboard
│       ├── VALE3_residuals.csv
│       └── ITUB4_residuals.csv
│
├── dashboard/
│   ├── app.py                         # Entrada principal do Streamlit
│   └── pages/
│       ├── 01_serie_temporal.py       # Gráfico real vs. previsto por ativo e modelo
│       ├── 02_metricas.py             # Tabela comparativa de métricas
│       ├── 03_erro_temporal.py        # Evolução do RMSE ao longo do período de teste
│       └── 04_residuos.py             # Histograma e Q-Q plot dos resíduos
│
├── requirements.txt                   # Dependências do projeto
└── README.md                          # Este arquivo
```

> **Como rodar:** `python src/pipeline.py`
> O pipeline verifica automaticamente quais etapas já foram concluídas e retoma de onde parou.

---

## Roadmap de Desenvolvimento

### Fase 1 — Coleta de Dados

- Extração automatizada via `yfinance` para os três ativos
- Dados OHLCV (*Open, High, Low, Close, Volume*) + dados diários
- Coleta de variáveis exógenas: índice IBOV, taxa de câmbio USD/BRL, taxa Selic
- Persistência dos dados brutos em CSV para evitar re-downloads

### Fase 2 — Pré-processamento e Feature Engineering

- Tratamento de dias sem pregão e splits de ações
- Cálculo de **retornos logarítmicos** como variável-alvo (preços brutos são não-estacionários)
- Engenharia de features técnicas:
  - Médias móveis simples: SMA 20, SMA 50
  - RSI (*Relative Strength Index*)
  - ATR (*Average True Range*)
  - Lags da série temporal (1, 5, 21 dias)
- Normalização separada por ativo e por variável
- Divisão temporal: **70% treino / 15% validação / 15% teste**

> ⚠️ **Atenção:** nunca usar divisão aleatória — dados de séries temporais devem ser divididos por ordem cronológica.

### Fase 3 — Implementação dos Modelos

#### SARIMAX
- Uso de `pmdarima.auto_arima` com `seasonal=True` e `m=5` (semana de trading)
- Inclusão de variáveis exógenas: IBOV, USD/BRL, Selic
- Diagnóstico de resíduos (Ljung-Box, normalidade)

#### LSTM
- Arquitetura: camadas LSTM empilhadas com `Dropout` entre elas
- Hiperparâmetros principais: tamanho da janela de lookback (30/60/90 dias), unidades por camada, taxa de dropout
- Callbacks: `EarlyStopping` e `ReduceLROnPlateau`
- Normalização com `MinMaxScaler` do scikit-learn

#### Modelo Híbrido *(sugestão)*
- Etapa 1: SARIMAX captura a componente linear/tendência
- Etapa 2: LSTM é treinado sobre os resíduos do SARIMAX
- Previsão final: soma das duas predições

### Fase 4 — Treinamento e Validação

- Estratégia: **Walk-forward validation** (janela expansível)
  - Treinamento até o mês T → avaliação no mês T+1 → desliza e repete
  - Simula o uso real do modelo em produção e evita *data leakage*
- Tuning de hiperparâmetros do LSTM com **Optuna** (otimização bayesiana)
  - Mais eficiente que GridSearch manual para espaços de busca contínuos

### Fase 5 — Avaliação Comparativa

Métricas computadas por modelo e por ativo:

| Métrica | Descrição |
|---------|-----------|
| MAE | Erro absoluto médio |
| RMSE | Raiz do erro quadrático médio |
| MAPE | Erro percentual absoluto médio |
| R² | Coeficiente de determinação |
| Diebold-Mariano | Teste estatístico de significância entre modelos |

> O **teste Diebold-Mariano** verifica formalmente se a diferença de desempenho entre dois modelos é estatisticamente significativa — evitando conclusões baseadas em variância amostral.

### Fase 6 — Exportação de Resultados

- `evaluation/predictions/<ativo>_predictions.csv` — previsões + valores reais por data
- `evaluation/metrics.csv` — tabela consolidada de métricas por modelo e ativo
- `evaluation/residuals/` — resíduos por modelo para análise no dashboard

### Como o pipeline orquestra tudo

Cada etapa implementa uma função `run(ticker)` e o orquestrador `pipeline.py` as chama em sequência, verificando se o arquivo de saída esperado já existe antes de executar:

```python
# src/pipeline.py
from pathlib import Path
import collect, preprocess, train_sarimax, train_lstm, train_hybrid, evaluate

TICKERS = ["PETR4", "VALE3", "ITUB4"]

def run():
    for ticker in TICKERS:
        if not Path(f"data/raw/{ticker}_raw.csv").exists():
            print(f"[1/6] Coletando {ticker}...")
            collect.run(ticker)

        if not Path(f"data/processed/{ticker}_processed.csv").exists():
            print(f"[2/6] Pré-processando {ticker}...")
            preprocess.run(ticker)

        if not Path(f"artifacts/{ticker}_sarimax.pkl").exists():
            print(f"[3/6] Treinando SARIMAX — {ticker}...")
            train_sarimax.run(ticker)

        if not Path(f"artifacts/{ticker}_lstm.keras").exists():
            print(f"[4/6] Treinando LSTM — {ticker}...")
            train_lstm.run(ticker)

        if not Path(f"evaluation/predictions/{ticker}_predictions.csv").exists():
            print(f"[5/6] Rodando modelo híbrido — {ticker}...")
            train_hybrid.run(ticker)

    if not Path("evaluation/metrics.csv").exists():
        print("[6/6] Avaliando modelos...")
        evaluate.run()

    print("Pipeline concluído. Rode: streamlit run dashboard/app.py")

if __name__ == "__main__":
    run()
```

> **Vantagem:** se o processo travar (erro de rede, treino interrompido), basta rodar `python src/pipeline.py` novamente — as etapas já concluídas são puladas automaticamente.

### Fase 7 — Dashboard Python (Streamlit)

Painéis do dashboard:

1. **Série temporal real vs. prevista** — seletor de ativo e modelo, gráfico interativo Plotly
2. **Tabela comparativa de métricas** — SARIMAX vs. LSTM vs. Híbrido por ativo
3. **Erro por janela temporal** — evolução do RMSE ao longo do período de teste
4. **Distribuição dos resíduos** — histograma e Q-Q plot por modelo

Deploy gratuito disponível via **Streamlit Cloud**.

---

## Modelos Implementados

### SARIMAX

O modelo SARIMA(*p,d,q*)(*P,D,Q*)[m] com variáveis exógenas (X) representa o *baseline* do projeto. Sua vantagem está na interpretabilidade e na capacidade explícita de modelar sazonalidade e tendência.

**Parâmetros selecionados via:** `pmdarima.auto_arima`

### LSTM

Redes *Long Short-Term Memory* são variantes de RNNs que solucionam o problema do gradiente desvanecente por meio de portas lógicas (*forget gate*, *input gate*, *output gate*) e células de memória. Isso permite aprender dependências temporais de longo prazo, essenciais para padrões complexos em séries financeiras.

### Modelo Híbrido *(sugerido)*

Combina a capacidade do SARIMAX de capturar estrutura linear com a capacidade do LSTM de modelar não-linearidades nos resíduos. Abordagem recorrente na literatura recente de forecasting financeiro.

---

## Métricas de Avaliação

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2   = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}
```

Teste Diebold-Mariano via pacote `arch`:

```python
from arch.unitroot.cointegration import engle_granger
# ou via statsmodels / implementação manual do DM test
```

---

## Dashboard

Para executar o dashboard localmente:

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

---

## Referências

1. Box, G. E. P.; Jenkins, G. M.; Reinsel, G. C.; Ljung, G. M. *Time Series Analysis: Forecasting and Control*. 5ª ed. John Wiley & Sons, 2015.
2. Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. 3ª ed. O'Reilly Media, 2022.
3. Goodfellow, I.; Bengio, Y.; Courville, A. *Deep Learning*. MIT Press, 2016.
4. Diebold, F. X.; Mariano, R. S. Comparing Predictive Accuracy. *Journal of Business & Economic Statistics*, v. 13, n. 3, p. 253–263, 1995.

---

*Henrique Kuroda, Thiago Silva — Engenharia da Computação, Fundação Hermínio Ometto, Araras, Brasil.*
