import os
import sys
import random
import itertools
import json
import pandas as pd
import numpy as np
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from train_lstm import criar_sequencias

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TARGET_COL    = config.TARGET_COL
EPOCHS        = 150
BATCH_SIZE    = config.BATCH_SIZE


def carregar_e_formatar_tune(name, lookback):
    df = pd.read_csv(
        PROCESSED_DIR / f"{name}_processed.csv",
        index_col="Date", parse_dates=True
    )
    target_idx = df.columns.get_loc(TARGET_COL)
    n          = len(df)
    train_df   = df.iloc[:int(n * config.TRAIN_END)]
    val_df     = df.iloc[int(n * config.TRAIN_END):int(n * config.VAL_END)]
    test_df    = df.iloc[int(n * config.VAL_END):]

    val_com_prefixo  = pd.concat([train_df.iloc[-lookback:], val_df])
    test_com_prefixo = pd.concat([val_df.iloc[-lookback:],   test_df])

    X_train, y_train = criar_sequencias(train_df,        target_idx, lookback)
    X_val,   y_val   = criar_sequencias(val_com_prefixo, target_idx, lookback)
    X_test,  y_test  = criar_sequencias(test_com_prefixo, target_idx, lookback)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def _construir_modelo(input_shape, units, dropout, lr):
    modelo = Sequential([
        Input(shape=input_shape),
        LSTM(units, return_sequences=True),
        Dropout(dropout),
        LSTM(units // 2, return_sequences=False),
        Dropout(dropout),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    modelo.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return modelo


def _tunar_ativo(asset):
    lookbacks      = [15, 30, 45]
    units_list     = [32, 64, 128]
    dropouts       = [0.1, 0.2, 0.3]
    learning_rates = [1e-3, 5e-4]

    all_combinations = list(itertools.product(lookbacks, units_list, dropouts, learning_rates))
    random.shuffle(all_combinations)
    trials = all_combinations[:12]

    best_val_mae = float('inf')
    best_params  = None
    results      = []

    for lookback, units, dropout, lr in trials:
        print(f"  Trial — lookback={lookback}, units={units}, dropout={dropout}, lr={lr}")

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            carregar_e_formatar_tune(asset, lookback)

        modelo = _construir_modelo(X_train.shape[1:], units, dropout, lr)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20,
                          restore_best_weights=True, min_delta=1e-6)
        ]
        modelo.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=0
        )

        # Seleção SOMENTE por val_mae — test é apenas informativo
        pred_val  = modelo.predict(X_val,  verbose=0).flatten()
        pred_test = modelo.predict(X_test, verbose=0).flatten()

        val_mae  = mean_absolute_error(y_val,  pred_val)
        test_mae = mean_absolute_error(y_test, pred_test)
        hit_rate = np.mean(np.sign(y_test) == np.sign(pred_test)) * 100

        print(f"    val_mae={val_mae:.5f}  test_mae={test_mae:.5f}  hit_rate={hit_rate:.1f}%")

        results.append({
            'asset': asset, 'lookback': lookback, 'units': units,
            'dropout': dropout, 'lr': lr,
            'val_mae': val_mae, 'test_mae': test_mae, 'hit_rate': hit_rate
        })

        # Critério de seleção: val_mae
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_params  = {
                'lookback': lookback,
                'units':    units,
                'dropout':  dropout,
                'lr':       lr
            }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Salvar best_params por ativo
    params_path = ARTIFACTS_DIR / f"{asset}_best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    # Salvar tabela completa de resultados
    pd.DataFrame(results).to_csv(
        ARTIFACTS_DIR / f"{asset}_tuning_results.csv", index=False
    )

    print(f"\n  ✅ {asset} — Best val MAE: {best_val_mae:.5f} | Params: {best_params}")


def run_tuning():
    ativos = sorted([
        f.stem.replace("_processed", "")
        for f in PROCESSED_DIR.glob("*_processed.csv")
    ])
    print(f"Iniciando tuning para: {ativos}")
    for asset in ativos:
        print(f"\n{'='*45}")
        print(f" Tuning — {asset}")
        print(f"{'='*45}")
        _tunar_ativo(asset)
    print("\n✅ Tuning concluído para todos os ativos.")


if __name__ == "__main__":
    run_tuning()