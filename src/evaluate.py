import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.regression.linear_model import OLS

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
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

def calculate_metrics(y_true, y_pred, y_true_unscaled=None, y_pred_unscaled=None):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # sMAPE — simétrico e bem-definido para valores próximos de zero (log_return)
    smape = np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100

    # Theil's U — desempenho relativo ao naive (prever retorno zero).
    denom = np.sqrt(np.mean(y_true ** 2))
    theil_u = np.sqrt(np.mean((y_true - y_pred) ** 2)) / denom if denom != 0 else np.nan

    r2 = r2_score(y_true, y_pred)

    # Hit Rate direcional — % de dias em que o modelo acertou a direção do movimento
    # Usa os valores despadronizados (se fornecidos) para ver a direção real em relação a 0
    if y_true_unscaled is not None and y_pred_unscaled is not None:
        hit_rate = np.mean(np.sign(y_true_unscaled) == np.sign(y_pred_unscaled)) * 100
    else:
        hit_rate = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100

    return {
        "MAE": mae, "RMSE": rmse, "sMAPE": smape,
        "Theil_U": theil_u, "R2": r2, "Hit_Rate": hit_rate
    }

def metricas_em_reais(name, datas, preds_por_modelo):
    """Reconstrói o preço (R$) a partir do retorno previsto e calcula métricas.

    O alvo está em retorno log padronizado; o ret_scaler salvo na Etapa 2 desfaz
    a padronização e o fechamento real do dia anterior reconstrói o preço:
        preco_previsto[t] = close_real[t-1] * exp(log_return_previsto[t])
    """
    raw   = pd.read_csv(RAW_DIR / f"{name}_raw.csv", index_col="Date", parse_dates=True)
    close = raw["Close"]
    real_price = close.reindex(datas).values
    prev_close = close.shift(1).reindex(datas).values

    with open(ARTIFACTS_DIR / f"{name}_ret_scaler.pkl", "rb") as f:
        ret_scaler = pickle.load(f)

    resultado = {}
    for modelo, pred_z in preds_por_modelo.items():
        log_ret    = ret_scaler.inverse_transform(np.asarray(pred_z).reshape(-1, 1)).ravel()
        pred_price = prev_close * np.exp(log_ret)
        mask = ~np.isnan(real_price) & ~np.isnan(pred_price)
        rp, pp = real_price[mask], pred_price[mask]
        resultado[modelo] = {
            "MAE_BRL":  mean_absolute_error(rp, pp),
            "RMSE_BRL": np.sqrt(mean_squared_error(rp, pp)),
            "MAPE_BRL": np.mean(np.abs((rp - pp) / rp)) * 100,
        }
    return resultado

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

        # Despadronizar para calcular a direção correta (Hit Rate)
        with open(ARTIFACTS_DIR / f"{name}_ret_scaler.pkl", "rb") as f:
            ret_scaler = pickle.load(f)
            
        y_true_unscaled = ret_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
        pred_sarimax_unscaled = ret_scaler.inverse_transform(pred_sarimax.reshape(-1, 1)).ravel()
        pred_hybrid_unscaled = ret_scaler.inverse_transform(pred_hybrid.reshape(-1, 1)).ravel()
        pred_lstm_unscaled = ret_scaler.inverse_transform(pred_lstm.reshape(-1, 1)).ravel()

        # 1. Métricas Base
        metrics_sarimax = calculate_metrics(y_true, pred_sarimax, y_true_unscaled, pred_sarimax_unscaled)
        metrics_lstm    = calculate_metrics(y_true, pred_lstm, y_true_unscaled, pred_lstm_unscaled)
        metrics_hybrid  = calculate_metrics(y_true, pred_hybrid, y_true_unscaled, pred_hybrid_unscaled)
        
        # 2. Teste Diebold-Mariano (Comparação Par a Par - Base Sarimax)
        # Queremos saber se LSTM e Hybrid são ESTATISTICAMENTE melhores que o baseline (Sarimax)
        # Hipótese Nula (H0): Mesma acurácia preditiva.
        # Se p_value < 0.05 e DM-stat for positivo, o modelo testado vence o Sarimax
        # dm_stat > 0 significa que erro do Sarimax > erro do Modelo 2
        dm_lstm_stat, p_lstm     = diebold_mariano_test(y_true, pred_sarimax, pred_lstm)
        dm_hybrid_stat, p_hybrid = diebold_mariano_test(y_true, pred_sarimax, pred_hybrid)

        # 3. Métricas em R$ (reconstrução de preço a partir do retorno previsto)
        try:
            precos = metricas_em_reais(
                name, datas_comuns,
                {"SARIMAX": pred_sarimax, "LSTM": pred_lstm, "Hibrido": pred_hybrid}
            )
        except Exception as e:
            print(f"    [aviso] não foi possível reconstruir preços em R$: {e}")
            precos = {}

        def registrar(modelo, metricas, dm_stat=None, p_value=None):
            linha = {
                "Ativo": name,
                "Modelo": modelo,
                "MAE": metricas["MAE"],
                "RMSE": metricas["RMSE"],
                "sMAPE": metricas["sMAPE"],
                "Theil_U": metricas["Theil_U"],
                "R2": metricas["R2"],
                "Hit_Rate": metricas["Hit_Rate"],
                "DM_Stat_vs_Base": dm_stat,
                "DM_PValue": p_value,
            }
            if modelo in precos:
                linha.update(precos[modelo])  # MAE_BRL, RMSE_BRL, MAPE_BRL
            resultados_finais.append(linha)

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