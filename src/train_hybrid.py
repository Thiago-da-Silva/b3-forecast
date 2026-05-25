import os
import sys
import random
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Seeds determinísticos — devem ser fixados antes de importar/usar o TensorFlow
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR   = PROJECT_ROOT / "evaluation" / "predictions"
RESIDUALS_DIR = PROJECT_ROOT / "evaluation" / "residuals"

EPOCHS = config.EPOCHS
BATCH_SIZE = config.BATCH_SIZE
LOOKBACK = config.LOOKBACK

def descobrir_ativos():
    ativos = []
    for arquivo in PROCESSED_DIR.glob("*_processed.csv"):
        name = arquivo.stem.replace("_processed", "")
        ativos.append(name)
    return sorted(ativos)

def criar_sequencias(X_data, y_data, lookback):
    X, y = [], []
    for i in range(lookback, len(X_data)):
        X.append(X_data[i-lookback:i, :])
        y.append(y_data[i])
    return np.array(X), np.array(y)

def carregar_e_formatar(name):
    df = pd.read_csv(PROCESSED_DIR / f"{name}_processed.csv", index_col="Date", parse_dates=True)
    sarimax_df = pd.read_csv(RESULTS_DIR / f"{name}_sarimax_predictions.csv", index_col="Date", parse_dates=True)

    datas_comuns = df.index.intersection(sarimax_df.index)
    df         = df.loc[datas_comuns]
    sarimax_df = sarimax_df.loc[datas_comuns]

    residuos = sarimax_df["real"] - sarimax_df["sarimax"]

    train_idx = sarimax_df["split"] == "train"
    val_idx   = sarimax_df["split"] == "val"
    test_idx  = sarimax_df["split"] == "test"

    df_train  = df[train_idx];  res_train = residuos[train_idx]
    df_val    = df[val_idx];    res_val   = residuos[val_idx]
    df_test   = df[test_idx];   res_test  = residuos[test_idx]

    # Prefixo para val e test (evita perder LOOKBACK amostras no início de cada split)
    df_val_pref   = pd.concat([df_train.iloc[-LOOKBACK:],  df_val])
    res_val_pref  = pd.concat([res_train.iloc[-LOOKBACK:], res_val])
    df_test_pref  = pd.concat([df_val.iloc[-LOOKBACK:],    df_test])
    res_test_pref = pd.concat([res_val.iloc[-LOOKBACK:],   res_test])

    X_train, y_train = criar_sequencias(df_train.values,       res_train.values,       LOOKBACK)
    X_val, y_val     = criar_sequencias(df_val_pref.values,    res_val_pref.values,    LOOKBACK)
    X_test, y_test   = criar_sequencias(df_test_pref.values,   res_test_pref.values,   LOOKBACK)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), sarimax_df

def construir_modelo(input_shape, units=64, dropout=0.2):
    modelo = Sequential([
        Input(shape=input_shape),
        LSTM(units, return_sequences=True),
        Dropout(dropout),
        LSTM(units // 2, return_sequences=False),
        Dropout(dropout),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    modelo.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
    return modelo

def treinar(name, X_train, y_train, X_val, y_val):
    print(f"    Treinando LSTM sobre resíduos para {name}...")

    params_path = ARTIFACTS_DIR / f"{name}_best_params.json"
    if params_path.exists():
        with open(params_path) as f:
            p = json.load(f)
        units   = p.get("units",   64)
        dropout = p.get("dropout", 0.2)
        lr      = p.get("lr",      1e-3)
        print(f"    Usando hiperparâmetros tunados: units={units}, dropout={dropout}, lr={lr}")
    else:
        units   = 64
        dropout = 0.2
        lr      = 1e-3

    modelo = construir_modelo((X_train.shape[1], X_train.shape[2]), units, dropout)
    modelo.optimizer.learning_rate.assign(lr)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, min_delta=1e-6),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, min_delta=1e-6)
    ]

    modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0
    )

    val_loss, val_mae = modelo.evaluate(X_val, y_val, verbose=0)
    print(f"    Validação — MAE Resíduos: {val_mae:.5f}  MSE: {val_loss:.5f}")

    return modelo

def salvar_modelo(modelo, name):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{name}_hybrid_lstm.keras"
    modelo.save(path)
    print(f"    Modelo residual salvo: {path}")

def salvar_previsoes(name, modelo, X_train, X_val, X_test, sarimax_df):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESIDUALS_DIR.mkdir(parents=True, exist_ok=True)

    pred_res_train = modelo.predict(X_train, verbose=0).flatten()
    pred_res_val   = modelo.predict(X_val,   verbose=0).flatten()
    pred_res_test  = modelo.predict(X_test,  verbose=0).flatten()

    # Treino perde os primeiros LOOKBACK dias (sem split anterior para prefixar)
    # Val e test cobrem todas as datas graças ao prefixo aplicado em carregar_e_formatar
    train_dates = sarimax_df[sarimax_df["split"] == "train"].index[LOOKBACK:]
    val_dates   = sarimax_df[sarimax_df["split"] == "val"].index
    test_dates  = sarimax_df[sarimax_df["split"] == "test"].index

    base_train = sarimax_df.loc[train_dates]
    base_val   = sarimax_df.loc[val_dates]
    base_test  = sarimax_df.loc[test_dates]

    hybrid_train = base_train["sarimax"].values + pred_res_train
    hybrid_val   = base_val["sarimax"].values   + pred_res_val
    hybrid_test  = base_test["sarimax"].values  + pred_res_test

    df_result = pd.DataFrame({
        "real":    np.concatenate([base_train["real"].values, base_val["real"].values, base_test["real"].values]),
        "sarimax": np.concatenate([base_train["sarimax"].values, base_val["sarimax"].values, base_test["sarimax"].values]),
        "hybrid":  np.concatenate([hybrid_train, hybrid_val, hybrid_test]),
        "split":   (["train"] * len(train_dates) + ["val"] * len(val_dates) + ["test"] * len(test_dates))
    }, index=train_dates.append(val_dates).append(test_dates))

    path = RESULTS_DIR / f"{name}_hybrid_predictions.csv"
    df_result.to_csv(path, index_label="Date")
    print(f"    Previsões híbridas salvas: {path}")

    df_residuos = pd.DataFrame({
        "res_real":            df_result["real"] - df_result["sarimax"],
        "res_previsto_lstm":   np.concatenate([pred_res_train, pred_res_val, pred_res_test]),
        "split":               df_result["split"]
    }, index=df_result.index)
    df_residuos.to_csv(RESIDUALS_DIR / f"{name}_residuals.csv", index_label="Date")

def run():
    ativos = descobrir_ativos()
    print(f"Ativos encontrados: {ativos}")

    for name in ativos:
        print(f"\n  Rodando Modelo Híbrido — {name}...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test), sarimax_df = carregar_e_formatar(name)
        modelo = treinar(name, X_train, y_train, X_val, y_val)
        salvar_modelo(modelo, name)
        salvar_previsoes(name, modelo, X_train, X_val, X_test, sarimax_df)
        print(f"  Concluído: {name}")

if __name__ == "__main__":
    run()