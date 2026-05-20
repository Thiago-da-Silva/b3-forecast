# Revisão de Código — b3-forecast
> Análise focada em erros que afetam diretamente os resultados: métricas, cálculos de indicadores, treinamento e avaliação estatística.

---

## Sumário de Severidade

| Severidade | Quantidade | Arquivos afetados |
|---|---|---|
| 🔴 Fatal (invalida resultados) | 6 | `preprocess.py`, `evaluate.py` |
| 🟠 Alto (degrada precisão) | 6 | `preprocess.py`, `train_lstm.py`, `train_sarimax.py` |
| 🟡 Médio (perda de dados de avaliação) | 1 | `train_lstm.py` |

**Ordem de prioridade recomendada:**
1. Alinhamento por data no `evaluate.py`
2. MAPE sobre log_return
3. Teste Diebold-Mariano sem HAC
4. RSI com SMA em vez de EWM
5. ATR sem True Range
6. Lags não-estacionários
7. `inf` no RSI não tratado
8. Taxa de aprendizado inicial
9. Limite de épocas
10. Seeds determinísticos
11. Lookback cruzando fronteira de split
12. SARIMAX walk-forward

---

## `preprocess.py`

### 🔴 [FATAL] RSI calculado com SMA — fórmula errada

O RSI de Wilder usa suavização exponencial (`ewm`), não média simples (`rolling.mean`). A diferença é substancial: o RSI com SMA reage mais lentamente, produz valores diferentes em mercados de tendência e não respeita a escala 0–100 da forma esperada pelo mercado. Este indicador é uma feature direta do LSTM — entrar com RSI errado é treinar o modelo em ruído estruturado.

**Impacto:** todas as features RSI passadas ao LSTM e ao modelo híbrido estão numericamente erradas. O modelo aprende correlações falsas entre RSI incorreto e log_return.

```python
# ❌ ERRADO — SMA (não é o RSI de Wilder)
delta = df["Close"].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
df["rsi"] = 100 - (100 / (1 + gain / loss))

# ✅ CORRETO — EWM com com=13 (equivale ao período 14 de Wilder)
delta = df["Close"].diff()
gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
df["rsi"] = 100 - (100 / (1 + gain / loss))
```

---

### 🔴 [FATAL] ATR ignora o gap entre pregões — não é o True Range real

A fórmula atual calcula apenas `High - Low` (amplitude intraday). O ATR correto usa o *True Range*: `max(High−Low, |High−Close_prev|, |Low−Close_prev|)`. A diferença é crítica em dias de gap (abertura muito acima/abaixo do fechamento anterior — comum após earnings e eventos macro). Nesses dias, o ATR atual subestima severamente a volatilidade real.

**Impacto:** volatilidade subestimada sistematicamente nos dias de maior movimentação — exatamente os dias mais informativos para o modelo.

```python
# ❌ ERRADO — ignora gap entre fechamentos
df["atr"] = (df["High"] - df["Low"]).rolling(14).mean()

# ✅ CORRETO — True Range com correção de gap
prev_close = df["Close"].shift(1)
tr = pd.concat([
    df["High"] - df["Low"],
    (df["High"] - prev_close).abs(),
    (df["Low"]  - prev_close).abs()
], axis=1).max(axis=1)
df["atr"] = tr.ewm(com=13, adjust=False).mean()  # Wilder's EWM
```

---

### 🔴 [FATAL] Lags usam o preço de fechamento (não-estacionário) como feature

Os lags `lag_1`, `lag_5` e `lag_21` estão sobre o `Close` bruto — que tem tendência de longo prazo e não é estacionário. Isso cria features correlacionadas com o nível de preço absoluto, não com a dinâmica de retorno. Após normalizar com MinMaxScaler, essa informação se distorce ainda mais. Os lags deveriam ser sobre `log_return`, que é a série que o modelo tenta prever.

**Impacto:** o LSTM aprende a relação entre preço absoluto passado e retorno futuro — uma correlação espúria que não generaliza para fora da janela de treino.

```python
# ❌ ERRADO — lags do preço absoluto (não-estacionário)
df["lag_1"]  = df["Close"].shift(1)
df["lag_5"]  = df["Close"].shift(5)
df["lag_21"] = df["Close"].shift(21)

# ✅ CORRETO — lags do retorno logarítmico (estacionário, alinhado com o target)
df["lag_1"]  = df["log_return"].shift(1)
df["lag_5"]  = df["log_return"].shift(5)
df["lag_21"] = df["log_return"].shift(21)
```

---

### 🔴 [FATAL] RSI produz `inf` quando todos os movimentos são de alta — não capturado por `dropna()`

Quando `loss == 0` (período de apenas altas), a divisão `gain / loss` gera `inf`. O `df.dropna()` **não remove `inf`** por padrão no pandas — esses valores atravessam silenciosamente para o CSV e para o LSTM. Valores `inf` distorcem a normalização do MinMaxScaler (`scaler.fit_transform` com `inf` gera saída inválida para todas as amostras daquele split).

**Impacto:** silencioso e intermitente — só ocorre em períodos de rally forte. Pode corromper a normalização inteira de um ativo como PETR4 em alta.

```python
# ❌ ATUAL — dropna() não remove inf
df = df.dropna()

# ✅ CORRETO — tratar inf antes de dropna
df = df.replace([np.inf, -np.inf], np.nan).dropna()
```

---

### 🟠 [ALTO] Escala do `log_return` incompatível com as demais features normalizadas

Todas as features estão em `[0, 1]` após o MinMaxScaler. O `log_return` é readicionado sem escala, com valores típicos na faixa `[-0.1, +0.1]`. Na janela de sequência do LSTM, os neurônios recebem uma mistura de `[0, 1]` com `[-0.1, 0.1]`. As gates LSTM aplicam sigmoid/tanh uniformemente — as features `[0, 1]` dominarão, e os lags de `log_return` (escala 10× menor) serão sub-representados pelo gradiente.

**Impacto:** o LSTM ignora parcialmente a informação mais importante (retornos passados) por desequilíbrio de escala entre features.

```python
# ❌ ATUAL — log_return fora de escala (~[-0.1, 0.1]) com features em [0, 1]
train_df["log_return"] = train["log_return"].values

# ✅ CORRETO — normalizar log_return separadamente (StandardScaler preserva sinal)
from sklearn.preprocessing import StandardScaler

ret_scaler = StandardScaler()
train_df["log_return"] = ret_scaler.fit_transform(train[["log_return"]])
val_df["log_return"]   = ret_scaler.transform(val[["log_return"]])
test_df["log_return"]  = ret_scaler.transform(test[["log_return"]])

# Salvar ret_scaler em artifacts para inverse_transform depois
import pickle
with open(PROCESSED_DIR / f"{name}_ret_scaler.pkl", "wb") as f:
    pickle.dump(ret_scaler, f)
```

---

### 🟠 [ALTO] Scaler não é salvo em disco — resultados irreproduzíveis

O `MinMaxScaler` é treinado em `preprocess.py` e descartado. Para inverter as previsões de volta à escala original ou para aplicar a mesma normalização em dados novos, o scaler precisa estar disponível. Sem ele, qualquer re-execução parcial do pipeline pode gerar scalers diferentes se os dados mudarem, quebrando todos os modelos já treinados.

```python
# ✅ ADICIONAR ao final de process():
import pickle

scaler_path = PROCESSED_DIR / f"{name}_scaler.pkl"
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"    Scaler salvo: {scaler_path}")
```

---

## `evaluate.py`

### 🔴 [FATAL] MAPE sobre `log_return` produz valores sem sentido (pode chegar a 10.000%+)

O `log_return` de um dia típico da B3 vale algo como `0.008` (0,8%). Um erro de previsão de `0.005` sobre esse valor resulta em `MAPE = 62%`. Em dias de retorno próximo de zero (`0.0001`), o MAPE explode para `5000%` mesmo com erro absoluto mínimo. O `epsilon=1e-8` não resolve — é 100.000× menor que um retorno típico e não muda o cálculo. A tabela de métricas mostrará MAPE na casa das centenas, tornando a coluna inutilizável para comparar modelos.

**Impacto:** a métrica mais legível para comparação entre modelos estará completamente corrompida. Qualquer leitor do CSV vai concluir que todos os modelos são terríveis.

```python
# ❌ ERRADO — MAPE não é definido para retornos próximos de zero
epsilon = 1e-8
mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100

# ✅ OPÇÃO 1 — sMAPE (symmetric MAPE, bem-definido para valores próximos de zero)
smape = np.mean(
    2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
) * 100

# ✅ OPÇÃO 2 — Theil's U (mais informativo: U < 1 supera naive, U = 1 é igual ao naive)
naive   = np.roll(y_true, 1); naive[0] = y_true[0]
theil_u = (
    np.sqrt(np.mean((y_true - y_pred) ** 2)) /
    np.sqrt(np.mean((y_true - naive)  ** 2))
)
```

> **Recomendação:** substituir MAPE por sMAPE e adicionar Theil's U como métrica complementar. Manter MAE e RMSE como métricas primárias — são as mais confiáveis para retornos.

---

### 🔴 [FATAL] Teste Diebold-Mariano sem correção HAC — p-valores sistematicamente baixos demais

A variância do diferencial de perda `d` é estimada com `np.var(d, ddof=1)` — um estimador que assume **ausência de autocorrelação**. Séries de erros quadráticos de modelos financeiros são quase sempre autocorrelacionadas (volatility clustering). Sem correção Newey-West (HAC), o denominador da estatística DM é subestimado, tornando o `dm_stat` artificialmente grande e o `p_value` artificialmente pequeno. O resultado prático: o teste vai rejeitar H0 com frequência mesmo quando os modelos são equivalentes.

**Impacto:** a conclusão principal do projeto — qual modelo é estatisticamente superior — pode ser falsa.

```python
# ❌ ERRADO — variância sem correção de autocorrelação
var_d   = np.var(d, ddof=1)
dm_stat = mean_d / np.sqrt(var_d / n)

# ✅ CORRETO — variância HAC (Newey-West) via statsmodels
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

def diebold_mariano_test(y_true, y_pred1, y_pred2):
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2
    d  = e1**2 - e2**2
    n  = len(d)

    # Regressão de d em constante com erros robustos HAC (Newey-West)
    res     = OLS(d, np.ones(n)).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": int(n ** 0.25)}  # regra empírica para lags ótimos
    )
    dm_stat = res.tvalues[0]
    p_value = res.pvalues[0]
    return dm_stat, p_value
```

---

### 🔴 [FATAL] Alinhamento do LSTM por `[-len(y_true):]` pode comparar dias errados

O conjunto de teste do modelo híbrido e o do LSTM standalone têm comprimentos distintos. A linha `pred_lstm = test_lstm["lstm"].values[-len(y_true):]` alinha cortando os *últimos* N registros do LSTM. Se os tamanhos diferirem por 1 dia, cada linha da comparação estará off-by-one: o LSTM estará sendo comparado com o real de um dia diferente, e o MAE/RMSE final será calculado incorretamente.

**Impacto:** MAE/RMSE do LSTM podem estar calculados comparando a previsão do dia T com o real do dia T+1.

```python
# ❌ PERIGOSO — alinha por posição, não por data
pred_lstm = test_lstm["lstm"].values[-len(y_true):]

# ✅ CORRETO — carregar com index de data e alinhar por interseção
df_hybrid = pd.read_csv(
    RESULTS_DIR / f"{name}_hybrid_predictions.csv",
    index_col="Date", parse_dates=True
)
df_lstm = pd.read_csv(
    RESULTS_DIR / f"{name}_lstm_predictions.csv",
    index_col="Date", parse_dates=True
)

test_hybrid = df_hybrid[df_hybrid["split"] == "test"]
test_lstm   = df_lstm[df_lstm["split"] == "test"]

# Interseção garante que só datas presentes em ambos são comparadas
datas_comuns = test_hybrid.index.intersection(test_lstm.index)
y_true       = test_hybrid.loc[datas_comuns, "real"].values
pred_sarimax = test_hybrid.loc[datas_comuns, "sarimax"].values
pred_hybrid  = test_hybrid.loc[datas_comuns, "hybrid"].values
pred_lstm    = test_lstm.loc[datas_comuns, "lstm"].values
```

---

### 🟠 [ALTO] R² não é uma métrica adequada para retornos financeiros

O R² mede quanto da variância de `y_true` o modelo explica. Para log_returns de ações (relação sinal-ruído < 5%), um modelo bem calibrado pode ter R² < 0. Um modelo ingênuo que prevê 0 para todos os dias tem R² = 0 — e vai aparecer "melhor" que o LSTM em vários ativos. A coluna R² da tabela vai induzir interpretações erradas.

**Recomendação:** substituir ou complementar R² com **Theil's U** (U < 1 = supera o modelo ingênuo) ou **Hit Rate direcional** (percentual de acertos na direção do movimento).

```python
# ✅ Hit Rate direcional — % de dias em que o modelo acertou a direção
hit_rate = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100

# ✅ Theil's U — desempenho relativo ao naive (prever retorno zero)
naive   = np.zeros_like(y_true)  # baseline: prever retorno zero
theil_u = (
    np.sqrt(np.mean((y_true - y_pred) ** 2)) /
    np.sqrt(np.mean((y_true - naive)  ** 2))
)
```

---

## `train_lstm.py`

### 🟠 [ALTO] Taxa de aprendizado inicial `0.001` é alta demais para séries financeiras de baixo SNR

O Adam padrão usa `lr=0.001`. Log-returns de ações têm relação sinal-ruído extremamente baixa (< 5%). Com LR alto, o gradiente da loss MSE oscila sem convergir, e o modelo aprende o ruído antes do sinal. O `ReduceLROnPlateau(factor=0.5, patience=5)` vai reduzindo gradativamente, mas as primeiras 10–15 épocas já danificaram os pesos. Com `EPOCHS=50`, esses épocas ruins representam 20–30% do treino total.

```python
# ❌ ATUAL — LR padrão do Adam (0.001), alto para dados financeiros
modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ✅ CORRETO — LR inicial menor, mais adequado para séries de baixo SNR
from tensorflow.keras.optimizers import Adam

modelo.compile(
    optimizer=Adam(learning_rate=1e-4),  # 0.0001 em vez de 0.001
    loss='mse',
    metrics=['mae']
)
```

---

### 🟠 [ALTO] `EPOCHS=50` é insuficiente — EarlyStopping não consegue encontrar o ótimo real

O EarlyStopping com `patience=15` precisa de espaço para agir. Com apenas 50 épocas, se o melhor peso for encontrado na época 5 (comum com LR que oscila no início), o modelo para na época 20 — mas o ótimo global com LR já reduzido pode estar na época 80 ou além. O limite de 50 épocas é o gargalo, não o EarlyStopping. Aumentar para 300–500 e deixar o callback trabalhar livremente.

```python
# ❌ ATUAL — limita artificialmente o treinamento
EPOCHS = 50

# ✅ CORRETO — EarlyStopping com patience maior define o limite natural
EPOCHS = 300

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,               # aumentado de 15 para 20
        restore_best_weights=True,
        min_delta=1e-6
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,                # aumentado de 5 para 7
        min_lr=1e-6,
        min_delta=1e-6
    )
]
```

---

### 🟠 [ALTO] Nenhum seed fixo — resultados diferentes a cada execução

Sem seed determinístico, duas execuções do mesmo código produzem modelos com pesos diferentes e métricas distintas. Para um projeto comparativo (SARIMAX vs LSTM vs Híbrido), se o LSTM precisar ser re-treinado por qualquer motivo, os números da tabela de métricas mudam — invalidando a comparação com SARIMAX, que é determinístico.

```python
# ✅ ADICIONAR no topo de train_lstm.py e train_hybrid.py, antes de qualquer import do TF
import random
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
```

---

### 🟡 [MÉDIO] Sequências não usam o final do split anterior como lookback inicial

Em `criar_sequencias(val_df, ...)`, os primeiros 30 dias de validação são usados apenas como features (janela de lookback) — a primeira previsão de val ocorre no dia 31. Na realidade, os 30 dias anteriores ao início de val são os últimos 30 dias de train, que já existem. Essa perda representa ~16% a menos de dados de avaliação em val e test, e a transição train→val fica com uma lacuna temporal que não existe nos dados reais.

**Impacto:** menos dias avaliados no conjunto de teste, e a primeira previsão de cada split usa contexto exclusivamente interno ao split, sem continuidade com o período anterior.

```python
# ❌ ATUAL — primeiro lookback de val constrói janela dentro de val (desperdiça 30 dias)
X_val, y_val   = criar_sequencias(val_df,  target_idx, LOOKBACK)
X_test, y_test = criar_sequencias(test_df, target_idx, LOOKBACK)

# ✅ CORRETO — usar os últimos LOOKBACK dias do split anterior como prefixo
val_with_prefix  = pd.concat([train_df.iloc[-LOOKBACK:], val_df])
test_with_prefix = pd.concat([val_df.iloc[-LOOKBACK:],  test_df])

X_val,  y_val  = criar_sequencias(val_with_prefix,  target_idx, LOOKBACK)
X_test, y_test = criar_sequencias(test_with_prefix, target_idx, LOOKBACK)

# Os y gerados terão exatamente len(val_df) e len(test_df) amostras — sem perda
```

---

## `train_sarimax.py`

### 🟠 [ALTO] Previsão de val e test é multi-step em bloco — não walk-forward

`modelo.predict(n_periods=len(val), X=exog_val)` gera ~190 previsões de uma vez, onde cada passo usa a previsão anterior (não o valor real). Isso não simula o uso real, onde se prevê 1 dia, observa o resultado, atualiza e prevê o próximo. O erro se acumula ao longo dos 190 dias e o desempenho reportado é pior do que o one-step-ahead real. A comparação com o LSTM (que faz one-step-ahead) fica desequilibrada: o SARIMAX aparece mais fraco do que realmente é.

**Impacto:** o SARIMAX é penalizado artificialmente na comparação. O Diebold-Mariano pode concluir que LSTM é superior quando o SARIMAX one-step seria competitivo.

```python
# ❌ ATUAL — previsão multi-step em bloco (erro acumulativo de 190 dias)
pred_val = modelo.predict(n_periods=len(val), X=exog_val)

# ✅ CORRETO — walk-forward one-step-ahead
pred_val = []
for i in range(len(val)):
    yhat = modelo.predict(n_periods=1, X=exog_val.iloc[[i]])
    pred_val.append(float(yhat[0]))
    modelo.update([y_val.iloc[i]], X=exog_val.iloc[[i]])  # atualiza com valor real
pred_val = np.array(pred_val)

# Repetir a mesma lógica para o conjunto de teste
pred_test = []
for i in range(len(test)):
    yhat = modelo.predict(n_periods=1, X=exog_test.iloc[[i]])
    pred_test.append(float(yhat[0]))
    modelo.update([y_test.iloc[i]], X=exog_test.iloc[[i]])
pred_test = np.array(pred_test)
```

> **Atenção:** o walk-forward é consideravelmente mais lento (uma chamada de `predict` por dia de dados). Para ~190 dias de val + ~190 de test, esperar 5–15 minutos por ativo dependendo da ordem ARIMA selecionada.

---

## Resumo das Correções por Arquivo

### `preprocess.py`
- [ ] RSI: `rolling(14).mean()` → `ewm(com=13, adjust=False).mean()`
- [ ] ATR: `High - Low` → True Range com `prev_close`
- [ ] Lags: `df["Close"].shift(n)` → `df["log_return"].shift(n)`
- [ ] Tratar `inf` antes do `dropna`: `df.replace([np.inf, -np.inf], np.nan).dropna()`
- [ ] Normalizar `log_return` com `StandardScaler` separado
- [ ] Serializar o `MinMaxScaler` e o `StandardScaler` em `artifacts/`

### `evaluate.py`
- [ ] Substituir MAPE por sMAPE
- [ ] Adicionar Theil's U e Hit Rate direcional
- [ ] Teste DM: `np.var(d)` → estimador HAC via `OLS(...).fit(cov_type="HAC")`
- [ ] Alinhamento LSTM: `[-len(y_true):]` → interseção de datas

### `train_lstm.py`
- [ ] `optimizer='adam'` → `Adam(learning_rate=1e-4)`
- [ ] `EPOCHS = 50` → `EPOCHS = 300`
- [ ] Adicionar seeds no topo do arquivo
- [ ] Lookback cruzado: prefixar val e test com os últimos `LOOKBACK` dias do split anterior

### `train_hybrid.py`
- [ ] `Adam(learning_rate=1e-4)` (mesmo ajuste do LSTM)
- [ ] `EPOCHS = 300` (mesmo ajuste do LSTM)
- [ ] Adicionar seeds no topo do arquivo

### `train_sarimax.py`
- [ ] Substituir previsão em bloco por walk-forward one-step-ahead em val e test

---

*Revisão gerada com base nos arquivos: `preprocess.py`, `train_lstm.py`, `train_sarimax.py`, `train_hybrid.py`, `evaluate.py`*
