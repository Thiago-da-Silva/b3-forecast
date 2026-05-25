# Previsão de Ações da B3: LSTM vs. SARIMAX vs. Híbrido

Estudo comparativo entre redes neurais LSTM, modelos SARIMAX e abordagens Híbridas para previsão de retornos de ativos da Bolsa de Valores do Brasil (B3). O projeto conta com um pipeline de Machine Learning de ponta a ponta e um dashboard interativo em Python (Streamlit).

---

## Sumário

- [Visão Geral](#visão-geral)
- [Ativos Analisados](#ativos-analisados)
- [Avanços Arquiteturais e Correções (Refatoração Crítica)](#avanços-arquiteturais-e-correções-refatoração-crítica)
- [Stack Tecnológica](#stack-tecnológica)
- [Estrutura do Projeto e Pipeline](#estrutura-do-projeto-e-pipeline)
- [Modelos Implementados](#modelos-implementados)
- [Métricas de Avaliação](#métricas-de-avaliação)
- [Dashboard](#dashboard)

---

## Visão Geral

O mercado de ações é caracterizado por alta volatilidade, ruído e não-linearidade, tornando a previsão de preços um dos problemas mais desafiadores em ciência de dados. Este projeto propõe a implementação e comparação estruturada de três paradigmas preditivos:

- **SARIMAX** — Modelo estatístico clássico, transparente e eficaz para componentes lineares e sazonais, utilizado como *baseline*.
- **LSTM** *(Long Short-Term Memory)* — Arquitetura de aprendizado profundo capaz de capturar dependências temporais de longo prazo.
- **Modelo Híbrido** — Utiliza o SARIMAX para capturar a tendência/componente linear e o LSTM para modelar os resíduos não-lineares, combinando os pontos fortes de ambas as abordagens.

O objetivo central é determinar qual abordagem apresenta menor taxa de erro preditivo e maior taxa de acerto direcional (*Hit Rate*) avaliadas sobre três ativos de setores distintos.

---

## Ativos Analisados

| Ticker | Empresa | Setor | Característica |
|--------|---------|-------|----------------|
| `PETR4` | Petrobras | Energia / Commodities | Alta sensibilidade a variáveis externas (câmbio, petróleo) |
| `VALE3` | Vale | Mineração | Correlacionado a preços de minério e mercado chinês |
| `ITUB4` | Itaú Unibanco | Financeiro | Ciclos econômicos domésticos, maior estabilidade relativa |

Período de coleta atual: **2019 a Maio de 2026** (Cobrindo ciclos completos: Bull Markets, pandemia de COVID-19 e alterações bruscas da Selic).

---

## Avanços Arquiteturais e Correções (Refatoração Crítica)

O projeto passou por um rigoroso processo de auditoria (QA) focado em resolver problemas clássicos na aplicação de Deep Learning ao mercado financeiro:

1. **Features 100% Estacionárias:** Preços brutos (OHLCV) causam colapso de média no LSTM (distribuição shift). Foram convertidos para osciladores e razões estacionárias (ex: distância percentual da Média Móvel, amplitude da vela, ATR percentual). A taxa Selic também foi preservada em nível suavizado (MA21) em vez de diff diário esparso.
2. **Fim do Viés Otimista (Bull Market Bias):** Modelos de IA na bolsa tendem a viciar em chutar apenas "Alta", já que o mercado sobe na maior parte do tempo. O LSTM puro foi consertado usando **balanceamento de classes (sample weighting)** e uma **Função de Perda Direcional Customizada (Directional Loss)** que aplica uma forte multa matemática caso o modelo erre a direção (sinal) da ação. O viés foi completamente zerado.
3. **Fim do Data Leakage:** A otimização de hiperparâmetros (Tuning) agora seleciona os melhores modelos rigorosamente com base no erro do conjunto de Validação (`val_mae`), mantendo o conjunto de Teste intacto e à prova de vazamento de dados.
4. **Tuning Dinâmico por Ativo:** O pipeline agora busca a melhor janela temporal (lookback), quantidade de neurônios, dropout e learning rate individualmente para cada ativo (ITUB4, PETR4, VALE3) antes de treinar.

**Resultado:** O modelo Híbrido, livre de vieses e treinado em dados estacionários, atingiu uma taxa de acerto de direção (**Hit Rate**) impressionante, na casa dos **81% para o ativo ITUB4** no período de teste.

---

## Stack Tecnológica

```
Coleta de dados   →  yfinance, python-bcb
Manipulação       →  pandas, numpy
Modelo estatístico →  pmdarima (auto_arima)
Modelo deep learning → TensorFlow / Keras
Tuning de hiperparâmetros → Grid/Random Search integrado
Pré-processamento / métricas → scikit-learn
Visualização      →  Plotly Express, Plotly Graph Objects
Dashboard         →  Streamlit
Avaliação estatística → statsmodels (teste Diebold-Mariano com correção HAC)
```

---

## Estrutura do Projeto e Pipeline

O orquestrador `src/pipeline.py` executa o fluxo completo (7 etapas) com um único comando:

```python
# python src/pipeline.py

[1/7] Coletando dados (yfinance + BCB)...
[2/7] Pré-processando dados e calculando features técnicas...
[3/7] Treinando modelo SARIMAX (Baseline)...
[4/7] Tunando hiperparâmetros LSTM por ativo...
[5/7] Treinando rede neural LSTM...
[6/7] Treinando modelo Híbrido (SARIMAX + LSTM Residual)...
[7/7] Avaliando modelos e aplicando Teste Diebold-Mariano...
```

### Divisão Temporal
Os dados não sofrem cross-validation aleatório (inválido para séries temporais), mas sim um *split* cronológico rigoroso:
* **70%** Treinamento
* **15%** Validação (Utilizado pelo EarlyStopping e no Tuning)
* **15%** Teste (Intocável; usado apenas na tabela final de métricas)

---

## Modelos Implementados

### SARIMAX
O modelo SARIMA(*p,d,q*)(*P,D,Q*)[m] com variáveis exógenas (IBOV, Dólar, Selic MA21) representa o *baseline* do projeto. Avaliado em regime *walk-forward* one-step-ahead no período de teste.

### LSTM Customizado
Redes *Long Short-Term Memory* tunadas por ativo. Possuem arquitetura flexível baseada nos hiperparâmetros salvos em `artifacts/<ativo>_best_params.json` e otimizadas com taxa de aprendizado decrescente (`ReduceLROnPlateau`).

### Modelo Híbrido (O Vencedor)
O SARIMAX projeta a base diária e os seus resíduos (erros) são passados como "alvo" para o LSTM aprender as falhas não-lineares do modelo estático. A previsão final é a soma linear das duas saídas.

---

## Métricas de Avaliação

A tabela de resultados consolida múltiplas visões (disponível no Dashboard e em `evaluation/metrics.csv`):

| Métrica | Descrição |
|---------|-----------|
| **MAE** | Erro absoluto médio na escala padronizada original. |
| **RMSE** | Raiz do erro quadrático médio (pune erros grotescos de previsão). |
| **sMAPE** | Erro percentual absoluto simétrico. |
| **Theil's U** | Desempenho relativo ao modelo ingênuo (Prever 0). Valores < 1.0 atestam inteligência artificial real. |
| **Hit Rate** | Taxa percentual de acerto direcional (se a bolsa fechou verde ou vermelha). A métrica mais importante para o negócio. |
| **R²** | Coeficiente de determinação (normalmente negativo para ações estocásticas diárias). |
| **Métricas em BRL** | Conversão matemática da previsão para o valor original do preço da ação em Reais (R$). |
| **Diebold-Mariano**| P-Value do teste estatístico. P < 0.05 prova que o LSTM/Híbrido venceu o SARIMAX por mérito, e não por sorte estatística. |

---

## Dashboard

Para visualizar interativamente os resultados, execute o dashboard localmente:

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

### Funcionalidades do Dashboard:
1. **Três formatos de Gráfico Temporal Interativo:**
   - **Retorno Acumulado (%):** Curva de patrimônio. Mostra a evolução real vs. as tendências capturadas pelo robô em um gráfico de linhas.
   - **Variação Percentual Diária (%):** Gráfico de **barras agrupadas** revelando de forma cristalina os dias em que o modelo acertou o "Sobe ou Desce" (Hit).
   - **Retorno Log Padronizado:** A visão original dos z-scores matemáticos.
2. **Tabela de Métricas Completa:** Visão granular das métricas de todos os ativos avaliados no período estrito de Teste.
3. **Glossário Integrado:** Explicação em texto de todas as métricas estatísticas em linguagem simples.

---

*Henrique Kuroda, Thiago Silva — Engenharia da Computação, Fundação Hermínio Ometto, Araras, Brasil.*