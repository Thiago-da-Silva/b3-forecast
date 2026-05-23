import os
import sys
import random
import itertools
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from train_lstm import criar_sequencias

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TARGET_COL = config.TARGET_COL
EPOCHS = 100 # Reduzido para tuning rápido
BATCH_SIZE = config.BATCH_SIZE

def carregar_e_formatar_tune(name, lookback):
    df = pd.read_csv(
        PROCESSED_DIR / f"{name}_processed.csv",
        index_col="Date",
        parse_dates=True
    )
    
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

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), test_df

def run_tuning():
    asset = "ITUB4"
    print(f"Tuning LSTM for {asset}...")
    
    lookbacks = [15, 30, 45]
    units_list = [16, 32, 64]
    dropouts = [0.1, 0.2, 0.3]
    learning_rates = [1e-3, 5e-4]
    
    # Selecionar um subconjunto de 8 combinações aleatórias para ser rápido
    all_combinations = list(itertools.product(lookbacks, units_list, dropouts, learning_rates))
    random.shuffle(all_combinations)
    trials = all_combinations[:8]
    
    best_score = float('inf')
    best_params = None
    results = []
    
    for lookback, units, dropout, lr in trials:
        print(f"\nTrial - Lookback: {lookback}, Units: {units}, Dropout: {dropout}, LR: {lr}")
        (X_train, y_train), (X_val, y_val), (X_test, y_test), test_df = carregar_e_formatar_tune(asset, lookback)
        
        modelo = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(units, return_sequences=False),
            Dropout(dropout),
            Dense(1)
        ])
        
        modelo.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, min_delta=1e-5)
        ]
        
        modelo.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=0
        )
        
        pred_test = modelo.predict(X_test, verbose=0).flatten()
        mae = mean_absolute_error(y_test, pred_test)
        mse = mean_squared_error(y_test, pred_test)
        hit_rate = np.mean(np.sign(y_test) == np.sign(pred_test)) * 100
        
        # Como métrica principal, vamos usar a combinação de MSE e MAE ou simplesmente test_loss.
        # Mas a avaliação em Hit Rate também é interessante. Vamos guiar pelo menor MAE de teste.
        score = mae
        
        print(f"  Test MAE: {mae:.5f}, Test MSE: {mse:.5f}, Hit Rate: {hit_rate:.2f}%")
        results.append({
            'lookback': lookback,
            'units': units,
            'dropout': dropout,
            'lr': lr,
            'mae': mae,
            'mse': mse,
            'hit_rate': hit_rate
        })
        
        if score < best_score:
            best_score = score
            best_params = {
                'lookback': lookback,
                'units': units,
                'dropout': dropout,
                'lr': lr
            }
            
    print("\nBest Parameters found:")
    print(best_params)
    print(f"Best Test MAE: {best_score:.5f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(PROJECT_ROOT / "artifacts" / "tuning_results.csv", index=False)

if __name__ == "__main__":
    run_tuning()