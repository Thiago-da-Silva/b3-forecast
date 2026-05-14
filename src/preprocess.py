import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
EXOGENOUS = ["IBOV", "USDBRL", "SELIC"]

def find_stocks():
    stocks = []
    for path in RAW_DIR.glob("*_raw.csv"):
        name = path.stem.replace("_raw", "")
        if name not in EXOGENOUS:
            stocks.append(name)
    return stocks

def run():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stocks = find_stocks()
    print(f"Ações encontradas: {stocks}")
    for name in stocks:
        print(f"Processando: {name}")
        process(name)
    print("Processamento concluído.")

def process(name):
    df = pd.read_csv(RAW_DIR / f"{name}_raw.csv", index_col="Date", parse_dates=True)

    # --- Retorno logarítmico ---
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # --- Variáveis exógenas ---
    for exog in EXOGENOUS:
        path = RAW_DIR / f"{exog}_raw.csv"
        if path.exists():
            col = pd.read_csv(path, index_col="Date", parse_dates=True)["Close"]
            df = df.join(col.rename(exog.lower()), how="left")
    df = df.ffill()

    # --- Média Móvel Simples / Simple Moving Average ---
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()

    # --- Índice de Força Relativa / Relative Strength Index ---
    delta        = df["Close"].diff()
    gain         = delta.clip(lower=0).rolling(14).mean()
    loss         = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"]    = 100 - (100 / (1 + gain / loss))

    # --- Average True Range ---
    df["atr"]    = (df["High"] - df["Low"]).rolling(14).mean()
    df["lag_1"]  = df["Close"].shift(1)
    df["lag_5"]  = df["Close"].shift(5)
    df["lag_21"] = df["Close"].shift(21)

    # --- Remove linhas com NaN (devido a indicadores) ---
    df = df.dropna()

    # --- Split cronológico 70/15/15 ---
    n     = len(df)
    train = df.iloc[:int(n * 0.70)]
    val   = df.iloc[int(n * 0.70):int(n * 0.85)]
    test  = df.iloc[int(n * 0.85):]

    # --- Normalização ---
    scaler       = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled   = scaler.transform(val)
    test_scaled  = scaler.transform(test)

    # --- Reconstrói DataFrames com os nomes de coluna ---
    train_df = pd.DataFrame(train_scaled, index=train.index, columns=df.columns)
    val_df   = pd.DataFrame(val_scaled,   index=val.index,   columns=df.columns)
    test_df  = pd.DataFrame(test_scaled,  index=test.index,  columns=df.columns)

    processed = pd.concat([train_df, val_df, test_df])
    processed.to_csv(PROCESSED_DIR / f"{name}_processed.csv")
    print(f"    Salvo: {name}_processed.csv ({len(processed)} linhas)")


