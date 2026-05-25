import sys
import pandas as pd
from pathlib import Path
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

def find_stocks():
    stocks = []
    for path in RAW_DIR.glob("*_raw.csv"):
        name = path.stem.replace("_raw", "")
        if name not in config.EXOGENOUS:
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
    for exog in config.EXOGENOUS:
        path = RAW_DIR / f"{exog}_raw.csv"
        if path.exists():
            col = pd.read_csv(path, index_col="Date", parse_dates=True)["Close"]
            df = df.join(col.rename(exog.lower()), how="left")
    df = df.ffill()

    # --- Exógenas estacionárias ---
    for exog in ("ibov", "usdbrl"):
        if exog in df.columns:
            df[exog] = np.log(df[exog] / df[exog].shift(1))
    # SELIC: manter o nível (não aplicar diff — é estacionário no período 2019-2026)

    # --- Indicadores técnicos ---
    # SMAs calculadas antes do drop do Close
    sma_20 = df["Close"].rolling(20).mean()
    sma_50 = df["Close"].rolling(50).mean()

    # SMA como razão ao preço (estacionária — mede afastamento percentual da média)
    df["sma_20_ratio"] = (df["Close"] / sma_20) - 1
    df["sma_50_ratio"] = (df["Close"] / sma_50) - 1

    # RSI (já estacionário — oscilador 0 a 100)
    delta     = df["Close"].diff()
    gain      = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss      = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))

    # ATR relativo ao preço (estacionário — mede volatilidade percentual)
    prev_close = df["Close"].shift(1)
    true_range = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_raw        = true_range.ewm(com=13, adjust=False).mean()
    df["atr_ratio"] = atr_raw / df["Close"]

    # Amplitude e direção relativa da vela (estacionárias)
    df["high_low_ratio"]    = (df["High"]  - df["Low"])   / df["Close"]
    df["open_close_ratio"]  = (df["Open"]  - df["Close"]) / df["Close"]

    # Volume como variação percentual (estacionária)
    df["volume_change"] = df["Volume"].pct_change()

    # Lags do log_return (estacionários)
    df["lag_1"]  = df["log_return"].shift(1)
    df["lag_5"]  = df["log_return"].shift(5)
    df["lag_21"] = df["log_return"].shift(21)

    # Remover colunas de preço absoluto (não-estacionárias)
    df = df.drop(columns=["Open", "High", "Low", "Close", "Volume"])

    # --- Remove linhas com NaN/inf ---
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # --- Split cronológico 70/15/15 ---
    n     = len(df)
    train = df.iloc[:int(n * config.TRAIN_END)]
    val   = df.iloc[int(n * config.TRAIN_END):int(n * config.VAL_END)]
    test  = df.iloc[int(n * config.VAL_END):]

    features = [col for col in df.columns if col != "log_return"]

    # --- Normalização das features (StandardScaler ajustado APENAS no treino) ---
    scaler       = StandardScaler()
    train_scaled = scaler.fit_transform(train[features])
    val_scaled   = scaler.transform(val[features])
    test_scaled  = scaler.transform(test[features])

    train_df = pd.DataFrame(train_scaled, index=train.index, columns=features)
    val_df   = pd.DataFrame(val_scaled,   index=val.index,   columns=features)
    test_df  = pd.DataFrame(test_scaled,  index=test.index,  columns=features)

    # --- Normalização do alvo ---
    ret_scaler = StandardScaler()
    train_df["log_return"] = ret_scaler.fit_transform(train[["log_return"]]).ravel()
    val_df["log_return"]   = ret_scaler.transform(val[["log_return"]]).ravel()
    test_df["log_return"]  = ret_scaler.transform(test[["log_return"]]).ravel()

    # Garante a ordem original das colunas
    train_df = train_df[df.columns]
    val_df   = val_df[df.columns]
    test_df  = test_df[df.columns]

    processed = pd.concat([train_df, val_df, test_df])
    processed.to_csv(PROCESSED_DIR / f"{name}_processed.csv")
    print(f"    Salvo: {name}_processed.csv ({len(processed)} linhas)")

    # --- Serializa scalers ---
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / f"{name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(ARTIFACTS_DIR / f"{name}_ret_scaler.pkl", "wb") as f:
        pickle.dump(ret_scaler, f)
    print(f"    Scalers salvos: {name}_scaler.pkl, {name}_ret_scaler.pkl")

if __name__ == "__main__":
    run()