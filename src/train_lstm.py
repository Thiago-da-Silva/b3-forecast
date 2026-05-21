import os
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

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR   = PROJECT_ROOT / "evaluation" / "predictions"

# Hiperparâmetros base
LOOKBACK = 30 # Tamanho da janela (dias)
EPOCHS = 300
BATCH_SIZE = 32

TARGET_COL = "log_return"

def descobrir_ativos():
    ativos = []
    for arquivo in PROCESSED_DIR.glob("*_processed.csv"):
        name = arquivo.stem.replace("_processed", "")
        ativos.append(name)
    return sorted(ativos)

def criar_sequencias(df, target_col_idx, lookback):
    """
    Transforma dados 2D em 3D para o LSTM.
    X formato: (amostras, lookback, features)
    y formato: (amostras,)
    """
    data = df.values
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, :])
        y.append(data[i, target_col_idx])
    return np.array(X), np.array(y)

def carregar_e_formatar(name):
    df = pd.read_csv(
        PROCESSED_DIR / f"{name}_processed.csv",
        index_col="Date",
        parse_dates=True
    )
    
    # Precisamos do índice da coluna target
    target_idx = df.columns.get_loc(TARGET_COL)

    # Divisão 70/15/15 mantendo a consistência com as outras etapas
    n = len(df)
    train_df = df.iloc[:int(n * 0.70)]
    val_df   = df.iloc[int(n * 0.70):int(n * 0.85)]
    test_df  = df.iloc[int(n * 0.85):]

    # Criação de sequências.
    # Val e test são prefixados com os últimos LOOKBACK dias do split anterior:
    # evita descartar os primeiros LOOKBACK dias de avaliação e preserva a
    # continuidade temporal real entre os splits.
    val_com_prefixo  = pd.concat([train_df.iloc[-LOOKBACK:], val_df])
    test_com_prefixo = pd.concat([val_df.iloc[-LOOKBACK:],  test_df])

    X_train, y_train = criar_sequencias(train_df, target_idx, LOOKBACK)
    X_val, y_val     = criar_sequencias(val_com_prefixo, target_idx, LOOKBACK)
    X_test, y_test   = criar_sequencias(test_com_prefixo, target_idx, LOOKBACK)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (train_df, val_df, test_df)

def construir_modelo(input_shape):
    modelo = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1) # Saída é o log_return
    ])
    
    modelo.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])
    return modelo

def treinar(name, X_train, y_train, X_val, y_val):
    print(f"    Treinando rede neural para {name}...")
    
    modelo = construir_modelo((X_train.shape[1], X_train.shape[2]))
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, min_delta=1e-6),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, min_delta=1e-6)
    ]

    history = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0 # Deixar no 0 para não sujar o terminal; mostrará apenas o resumo abaixo
    )
    
    # Avaliação no conjunto de validação
    val_loss, val_mae = modelo.evaluate(X_val, y_val, verbose=0)
    print(f"    Validação — MAE: {val_mae:.5f}  MSE: {val_loss:.5f}")
    
    return modelo

def salvar_modelo(modelo, name):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{name}_lstm.keras"
    modelo.save(path)
    print(f"    Modelo salvo: {path}")

def salvar_previsoes(name, modelo, X_train, X_val, X_test, dfs):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = dfs
    
    pred_train = modelo.predict(X_train, verbose=0).flatten()
    pred_val   = modelo.predict(X_val, verbose=0).flatten()
    pred_test  = modelo.predict(X_test, verbose=0).flatten()
    
    # Treino perde os primeiros LOOKBACK dias (não há split anterior para prefixar).
    # Val e test foram prefixados, então cobrem todas as suas datas.
    real_train = train_df.iloc[LOOKBACK:][TARGET_COL].values
    real_val   = val_df[TARGET_COL].values
    real_test  = test_df[TARGET_COL].values

    df_result = pd.DataFrame({
        "real":    np.concatenate([real_train, real_val, real_test]),
        "lstm":    np.concatenate([pred_train, pred_val, pred_test]),
        "split":   (["train"] * len(real_train) +
                    ["val"]   * len(real_val)   +
                    ["test"]  * len(real_test))
    })

    datas_train = train_df.index[LOOKBACK:]
    datas_val   = val_df.index
    datas_test  = test_df.index
    df_result.index = datas_train.append(datas_val).append(datas_test)

    path = RESULTS_DIR / f"{name}_lstm_predictions.csv"
    df_result.to_csv(path, index_label="Date")
    print(f"    Previsões salvas: {path}")

def run():
    ativos = descobrir_ativos()
    print(f"Ativos encontrados: {ativos}")

    # Salva os parâmetros base para referência futura no dashboard ou métricas
    params_path = ARTIFACTS_DIR / "lstm_params.json"
    with open(params_path, "w") as f:
        json.dump({"lookback": LOOKBACK, "epochs": EPOCHS, "batch_size": BATCH_SIZE}, f)

    for name in ativos:
        print(f"\n  Treinando LSTM — {name}...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test), dfs = carregar_e_formatar(name)
        modelo = treinar(name, X_train, y_train, X_val, y_val)
        salvar_modelo(modelo, name)
        salvar_previsoes(name, modelo, X_train, X_val, X_test, dfs)
        print(f"  Concluído: {name}")

if __name__ == "__main__":
    run()