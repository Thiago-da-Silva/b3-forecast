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
Tuning de hiperparâmetros → Optuna (planejado — ainda não implementado)
Pré-processamento / métricas → scikit-learn
Visualização      →  Plotly Express
Dashboard         →  Streamlit
Avaliação estatística → statsmodels (teste Diebold-Mariano com correção HAC)
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
│   ├── train_lstm.py                  # Etapa 4: arquitetura e treinamento do LSTM
│   ├── config.py                      # Constantes compartilhadas (split, lookback, etc.)
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
├── artifacts/                         # Gerado pelas Etapas 2, 3, 4 e 5
│   ├── <ativo>_sarimax.pkl          # Modelo SARIMAX serializado
│   ├── <ativo>_lstm.keras           # Rede LSTM treinada
│   ├── <ativo>_hybrid_lstm.keras    # LSTM treinado sobre os resíduos do SARIMAX
│   ├── <ativo>_scaler.pkl           # StandardScaler das features (Etapa 2)
│   ├── <ativo>_ret_scaler.pkl       # StandardScaler do alvo log_return (Etapa 2)
│   └── lstm_params.json             # Hiperparâmetros base do LSTM
│
├── evaluation/                        # Gerado pela Etapa 6
│   ├── metrics.csv                    # Tabela consolidada: modelo × ativo × métrica
│   ├── predictions/                  # um CSV por modelo e ativo
│   │   ├── <ativo>_sarimax_predictions.csv  # data, real, sarimax, split
│   │   ├── <ativo>_lstm_predictions.csv     # data, real, lstm, split
│   │   └── <ativo>_hybrid_predictions.csv   # data, real, sarimax, hybrid, split
│   └── residuals/
│       ├── PETR4_residuals.csv        # Resíduos por modelo — input do dashboard
│       ├── VALE3_residuals.csv
│       └── ITUB4_residuals.csv
│
├── dashboard/
│   └── app.py                         # Dashboard Streamlit (página única):
│                                      # série temporal + tabela de métricas
│
├── requirements.txt                   # Dependências do projeto
├── .python-version                    # Versão do Python (3.13)
└── README.md                          # Este arquivo
```

> **Como rodar:** `python src/pipeline.py`
> O pipeline executa as 6 etapas em sequência. Reexecutar refaz todas as etapas
> (tornar o pipeline idempotente é uma melhoria planejada).

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
- Normalização com `StandardScaler` (features e alvo), ajustado apenas no treino

#### Modelo Híbrido *(sugestão)*
- Etapa 1: SARIMAX captura a componente linear/tendência
- Etapa 2: LSTM é treinado sobre os resíduos do SARIMAX
- Previsão final: soma das duas predições

### Fase 4 — Treinamento e Validação

- **SARIMAX:** validação *walk-forward* one-step-ahead — prevê 1 dia, observa o
  valor real, atualiza o modelo e avança, em validação e teste.
- **LSTM / Híbrido:** split temporal fixo 70/15/15 com `EarlyStopping` e
  `ReduceLROnPlateau`. (*Walk-forward* para o LSTM é uma melhoria planejada.)
- Tuning de hiperparâmetros com **Optuna**: **planejado** — atualmente os
  hiperparâmetros do LSTM são fixos (definidos em `src/config.py`).

### Fase 5 — Avaliação Comparativa

Métricas computadas por modelo e por ativo:

| Métrica | Descrição |
|---------|-----------|
| MAE | Erro absoluto médio |
| RMSE | Raiz do erro quadrático médio |
| sMAPE | Erro percentual absoluto simétrico (bem-definido perto de zero) |
| Theil's U | Desempenho relativo ao modelo ingênuo (U < 1 supera o naive) |
| Hit Rate | Percentual de acertos na direção do movimento |
| R² | Coeficiente de determinação |
| Diebold-Mariano | Teste de significância entre modelos (variância HAC) |

> O **teste Diebold-Mariano** verifica formalmente se a diferença de desempenho entre dois modelos é estatisticamente significativa — evitando conclusões baseadas em variância amostral.

### Fase 6 — Exportação de Resultados

- `evaluation/predictions/<ativo>_<modelo>_predictions.csv` — previsões + valores reais por data
- `evaluation/metrics.csv` — tabela consolidada de métricas por modelo e ativo
- `evaluation/residuals/` — resíduos por modelo para análise no dashboard

### Como o pipeline orquestra tudo

Cada etapa expõe uma função `run()` (sem argumentos) que processa os três ativos
internamente. O orquestrador `src/pipeline.py` as chama em sequência:

```python
# src/pipeline.py (resumo)
import src.collect as collect
import src.preprocess as preprocess
import src.train_sarimax as train_sarimax
import src.train_lstm as train_lstm
import src.train_hybrid as train_hybrid
import src.evaluate as evaluate

def run():
    collect.run()        # [1/6] coleta (yfinance + BCB)
    preprocess.run()     # [2/6] features + normalização
    train_sarimax.run()  # [3/6] SARIMAX (baseline, walk-forward)
    train_lstm.run()     # [4/6] LSTM
    train_hybrid.run()   # [5/6] híbrido (LSTM sobre resíduos do SARIMAX)
    evaluate.run()       # [6/6] métricas + Diebold-Mariano
```

> **Atenção:** atualmente o pipeline reexecuta **todas** as etapas a cada chamada.
> Pular as já concluídas (idempotência) é uma melhoria planejada.

### Fase 7 — Dashboard Python (Streamlit)

Painéis do dashboard:

1. **Série temporal real vs. prevista** — seletor de ativo e modelo, gráfico interativo Plotly ✅
2. **Tabela comparativa de métricas** — SARIMAX vs. LSTM vs. Híbrido por ativo ✅
3. **Erro por janela temporal** — evolução do RMSE ao longo do período de teste 🔜 (planejado)
4. **Distribuição dos resíduos** — histograma e Q-Q plot por modelo 🔜 (planejado)

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
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    # sMAPE: simétrico e bem-definido para valores próximos de zero
    smape = np.mean(2 * np.abs(y_true - y_pred) /
                    (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    r2    = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape, "R2": r2}
```

Teste Diebold-Mariano com variância robusta (HAC / Newey-West) via `statsmodels`:

```python
from statsmodels.regression.linear_model import OLS
# regride o diferencial de perda (d) sobre uma constante com cov_type="HAC";
# a estatística-t do intercepto é a estatística DM corrigida
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
