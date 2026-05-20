import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.regression.linear_model import OLS

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR   = PROJECT_ROOT / "evaluation" / "predictions"
METRICS_DIR   = PROJECT_ROOT / "evaluation"

def descobrir_ativos():
    ativos = []
    for arquivo in PROCESSED_DIR.glob("*_processed.csv"):
        name = arquivo.stem.replace("_processed", "")
        ativos.append(name)
    return sorted(ativos)

def diebold_mariano_test(y_true, y_pred1, y_pred2):
    """
    Teste Diebold-Mariano (DM test) com variância robusta a
    autocorrelação e heterocedasticidade (HAC / Newey-West).
    Compara as perdas (erros quadráticos) de dois modelos.
    Retorna o DM-stat e o p-valor.
    """
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2

    # Função de perda: erro quadrático
    d = (e1**2) - (e2**2)
    n = len(d)

    # Evita divisão por zero se ambos os modelos preverem exatamente igual
    if np.var(d, ddof=1) == 0:
        return 0.0, 1.0

    # Regressão de d sobre uma constante com erros-padrão HAC (Newey-West).
    # A estatística-t do intercepto é exatamente a estatística DM corrigida.
    res = OLS(d, np.ones(n)).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": int(n ** 0.25)}
    )
    dm_stat = res.tvalues[0]
    p_value = res.pvalues[0]

    return dm_stat, p_value

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # sMAPE — simétrico e bem-definido para valores próximos de zero (log_return)
    smape = np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100

    # Theil's U — desempenho relativo ao naive (prever retorno zero).
    # U < 1 supera o naive, U = 1 é equivalente, U > 1 é pior.
    denom = np.sqrt(np.mean(y_true ** 2))
    theil_u = np.sqrt(np.mean((y_true - y_pred) ** 2)) / denom if denom != 0 else np.nan

    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape, "Theil_U": theil_u, "R2": r2}

def run():
    ativos = descobrir_ativos()
    print(f"Ativos encontrados: {ativos}")

    resultados_finais = []

    for name in ativos:
        print(f"\n  Calculando métricas para {name}...")
        
        # O arquivo hybrid_predictions.csv contém as previsões reais, sarimax e hybrid
        # O arquivo lstm_predictions.csv contém real e lstm
        df_hybrid = pd.read_csv(
            RESULTS_DIR / f"{name}_hybrid_predictions.csv",
            index_col="Date", parse_dates=True
        )
        df_lstm = pd.read_csv(
            RESULTS_DIR / f"{name}_lstm_predictions.csv",
            index_col="Date", parse_dates=True
        )

        # Filtra apenas o conjunto de TESTE para avaliação rigorosa
        test_hybrid = df_hybrid[df_hybrid["split"] == "test"]
        test_lstm   = df_lstm[df_lstm["split"] == "test"]

        # Interseção de datas garante que só dias presentes em ambos são comparados
        datas_comuns = test_hybrid.index.intersection(test_lstm.index)
        y_true       = test_hybrid.loc[datas_comuns, "real"].values
        pred_sarimax = test_hybrid.loc[datas_comuns, "sarimax"].values
        pred_hybrid  = test_hybrid.loc[datas_comuns, "hybrid"].values
        pred_lstm    = test_lstm.loc[datas_comuns, "lstm"].values

        # 1. Métricas Base
        metrics_sarimax = calculate_metrics(y_true, pred_sarimax)
        metrics_lstm    = calculate_metrics(y_true, pred_lstm)
        metrics_hybrid  = calculate_metrics(y_true, pred_hybrid)
        
        # 2. Teste Diebold-Mariano (Comparação Par a Par - Base Sarimax)
        # Queremos saber se LSTM e Hybrid são ESTATISTICAMENTE melhores que o baseline (Sarimax)
        # Hipótese Nula (H0): Mesma acurácia preditiva.
        # Se p_value < 0.05 e DM-stat for positivo, o modelo testado vence o Sarimax
        # dm_stat > 0 significa que erro do Sarimax > erro do Modelo 2
        dm_lstm_stat, p_lstm     = diebold_mariano_test(y_true, pred_sarimax, pred_lstm)
        dm_hybrid_stat, p_hybrid = diebold_mariano_test(y_true, pred_sarimax, pred_hybrid)

        def registrar(modelo, metricas, dm_stat=None, p_value=None):
            resultados_finais.append({
                "Ativo": name,
                "Modelo": modelo,
                "MAE": metricas["MAE"],
                "RMSE": metricas["RMSE"],
                "sMAPE": metricas["sMAPE"],
                "Theil_U": metricas["Theil_U"],
                "R2": metricas["R2"],
                "DM_Stat_vs_Base": dm_stat,
                "DM_PValue": p_value
            })

        registrar("SARIMAX", metrics_sarimax) # Baseline
        registrar("LSTM", metrics_lstm, dm_lstm_stat, p_lstm)
        registrar("Hibrido", metrics_hybrid, dm_hybrid_stat, p_hybrid)
        
        print("    Métricas geradas com sucesso.")

    df_metrics = pd.DataFrame(resultados_finais)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    path = METRICS_DIR / "metrics.csv"
    df_metrics.to_csv(path, index=False)
    print(f"\n[+] Tabela de métricas salva em: {path}")

if __name__ == "__main__":
    run()