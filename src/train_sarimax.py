from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR   = PROJECT_ROOT / "evaluation" / "predictions"

EXOGENOUS_COLS = ["ibov", "usdbrl", "selic"]
TARGET_COL     = "log_return"

def descobrir_ativos():
    ativos = []
    for arquivo in PROCESSED_DIR.glob("*_processed.csv"):
        name = arquivo.stem.replace("_processed", "")
        ativos.append(name)
    return sorted(ativos)

def carregar_splits(name):
    df = pd.read_csv(
        PROCESSED_DIR / f"{name}_processed.csv",
        index_col="Date",
        parse_dates=True
    )

    n     = len(df)
    train = df.iloc[:int(n * 0.70)]
    val   = df.iloc[int(n * 0.70):int(n * 0.85)]
    test  = df.iloc[int(n * 0.85):]

    return train, val, test

def separar_xy(split):
    y    = split[TARGET_COL]
    exog = split[EXOGENOUS_COLS]
    return y, exog

def treinar(name, train):
    print(f"    Rodando auto_arima para {name} — pode demorar alguns minutos...")

    y_train, exog_train = separar_xy(train)

    modelo = auto_arima(
        y                    = y_train,
        X                    = exog_train,
        seasonal             = True,
        m                    = 5,
        stepwise             = True,
        information_criterion= "aic",
        error_action         = "ignore",
        suppress_warnings    = True
    )

    print(f"    Melhor ordem encontrada: {modelo.order} sazonal: {modelo.seasonal_order}")
    return modelo

def salvar_modelo(modelo, name):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{name}_sarimax.pkl"
    with open(path, "wb") as f:
        pickle.dump(modelo, f)
    print(f"    Modelo salvo: {path}")

def salvar_previsoes(name, train, val, test, modelo):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    y_train, exog_train = separar_xy(train)
    y_val,   exog_val   = separar_xy(val)
    y_test,  exog_test  = separar_xy(test)

    # previsões in-sample (treino) — o modelo já viu esses dados
    pred_train = modelo.predict_in_sample(X=exog_train)

    # walk-forward one-step-ahead na validação:
    # prevê 1 dia, observa o real, atualiza o modelo e avança
    pred_val = []
    for i in range(len(val)):
        yhat = modelo.predict(n_periods=1, X=exog_val.iloc[[i]])
        pred_val.append(float(np.asarray(yhat)[0]))
        modelo.update([y_val.iloc[i]], X=exog_val.iloc[[i]])
    pred_val = np.array(pred_val)

    mae_val  = mean_absolute_error(y_val, pred_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, pred_val))
    print(f"    Validação (walk-forward) — MAE: {mae_val:.5f}  RMSE: {rmse_val:.5f}")

    # walk-forward one-step-ahead no teste
    pred_test = []
    for i in range(len(test)):
        yhat = modelo.predict(n_periods=1, X=exog_test.iloc[[i]])
        pred_test.append(float(np.asarray(yhat)[0]))
        modelo.update([y_test.iloc[i]], X=exog_test.iloc[[i]])
    pred_test = np.array(pred_test)

    df_result = pd.DataFrame({
        "real":    pd.concat([y_train, y_val, y_test]),
        "sarimax": np.concatenate([pred_train, pred_val, pred_test]),
        "split":   (["train"] * len(train) +
                    ["val"]   * len(val)   +
                    ["test"]  * len(test))
    })

    path = RESULTS_DIR / f"{name}_sarimax_predictions.csv"
    df_result.to_csv(path)
    print(f"    Previsões salvas: {path}")

def run():
    ativos = descobrir_ativos()
    print(f"Ativos encontrados: {ativos}")

    for name in ativos:
        print(f"\n  Treinando SARIMAX — {name}...")
        train, val, test = carregar_splits(name)
        modelo           = treinar(name, train)
        salvar_modelo(modelo, name)
        salvar_previsoes(name, train, val, test, modelo)
        print(f"  Concluído: {name}")


if __name__ == "__main__":
    run()