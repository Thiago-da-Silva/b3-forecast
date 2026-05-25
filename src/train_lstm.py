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

EPOCHS     = config.EPOCHS
BATCH_SIZE = config.BATCH_SIZE
TARGET_COL = config.TARGET_COL

def _carregar_params(name):
    params_path = ARTIFACTS_DIR / f"{name}_best_params.json"
    if params_path.exists():
        with open(params_path, "r") as f:
            p = json.load(f)
            return p.get("lookback", config.LOOKBACK), p.get("units", 64), p.get("dropout", 0.2), p.get("lr", 1e-3)
    return config.LOOKBACK, 64, 0.2, 1e-3

def descobrir_ativos():
    ativos = []
    for arquivo in PROCESSED_DIR.glob("*_processed.csv"):
        name = arquivo.stem.replace("_processed", "")
        ativos.append(name)
    return sorted(ativos)

def criar_sequencias(df, target_col_idx, lookback):
    data = df.values
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, :])
        y.append(data[i, target_col_idx])
    return np.array(X), np.array(y)

def carregar_e_formatar(name, lookback):
    df = pd.read_csv(PROCESSED_DIR / f"{name}_processed.csv", index_col="Date", parse_dates=True)
    target_idx = df.columns.get_loc(TARGET_COL)

    n = len(df)
    train_df = df.iloc[:int(n * config.TRAIN_END)]
    val_df   = df.iloc[int(n * config.TRAIN_END):int(n * config.VAL_END)]
    test_df  = df.iloc[int(n * config.VAL_END):]

    val_com_prefixo  = pd.concat([train_df.iloc[-lookback:], val_df])
    test_com_prefixo = pd.concat([val_df.iloc[-lookback:],  test_df])

    X_train, y_train = criar_sequencias(train_df, target_idx, lookback)
    X_val, y_val     = criar_sequencias(val_com_prefixo, target_idx, lookback)
    X_test, y_test   = criar_sequencias(test_com_prefixo, target_idx, lookback)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (train_df, val_df, test_df)

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
    print(f"    Treinando rede neural para {name}...")

    # Carregar hiperparâmetros tunados se existirem
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
        print(f"    Sem params tunados — usando defaults")

    modelo = construir_modelo((X_train.shape[1], X_train.shape[2]), units, dropout)
    modelo.optimizer.learning_rate.assign(lr)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, min_delta=1e-6),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, min_delta=1e-6)
    ]

    # Balanceamento de classes por direção (Alta vs Baixa)
    pos_mask = y_train > 0
    neg_mask = y_train <= 0
    pos_count = np.sum(pos_mask)
    neg_count = np.sum(neg_mask)
    total = len(y_train)

    weight_pos = total / (2.0 * pos_count) if pos_count > 0 else 1.0
    weight_neg = total / (2.0 * neg_count) if neg_count > 0 else 1.0

    sample_weights = np.ones_like(y_train)
    sample_weights[pos_mask] = weight_pos
    sample_weights[neg_mask] = weight_neg

    modelo.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0
    )

    val_loss, val_mae = modelo.evaluate(X_val, y_val, verbose=0)
    print(f"    Validação — MAE: {val_mae:.5f}  MSE: {val_loss:.5f}")

    return modelo

def salvar_modelo(modelo, name):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{name}_lstm.keras"
    modelo.save(path)
    print(f"    Modelo salvo: {path}")

def salvar_previsoes(name, modelo, X_train, X_val, X_test, dfs, lookback):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = dfs
    
    pred_train = modelo.predict(X_train, verbose=0).flatten()
    pred_val   = modelo.predict(X_val, verbose=0).flatten()
    pred_test  = modelo.predict(X_test, verbose=0).flatten()
    
    real_train = train_df.iloc[lookback:][TARGET_COL].values
    real_val   = val_df[TARGET_COL].values
    real_test  = test_df[TARGET_COL].values

    df_result = pd.DataFrame({
        "real":    np.concatenate([real_train, real_val, real_test]),
        "lstm":    np.concatenate([pred_train, pred_val, pred_test]),
        "split":   (["train"] * len(real_train) +
                    ["val"]   * len(real_val)   +
                    ["test"]  * len(real_test))
    })

    datas_train = train_df.index[lookback:]
    datas_val   = val_df.index
    datas_test  = test_df.index
    df_result.index = datas_train.append(datas_val).append(datas_test)

    path = RESULTS_DIR / f"{name}_lstm_predictions.csv"
    df_result.to_csv(path, index_label="Date")
    print(f"    Previsões salvas: {path}")

def run():
    ativos = descobrir_ativos()
    print(f"Ativos encontrados: {ativos}")

    for name in ativos:
        print(f"\n  Treinando LSTM — {name}...")
        lookback, _, _, _ = _carregar_params(name)
        (X_train, y_train), (X_val, y_val), (X_test, y_test), dfs = carregar_e_formatar(name, lookback)
        modelo = treinar(name, X_train, y_train, X_val, y_val)
        salvar_modelo(modelo, name)
        salvar_previsoes(name, modelo, X_train, X_val, X_test, dfs, lookback)
        print(f"  Concluído: {name}")

if __name__ == "__main__":
    run()