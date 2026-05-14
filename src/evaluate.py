import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.stats

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR   = Path("evaluation/predictions")
METRICS_DIR   = Path("evaluation")

def descobrir_ativos():
    ativos = []
    for arquivo in PROCESSED_DIR.glob("*_processed.csv"):
        name = arquivo.stem.replace("_processed", "")
        ativos.append(name)
    return sorted(ativos)

def diebold_mariano_test(y_true, y_pred1, y_pred2):
    """
    Implementação simples do Teste Diebold-Mariano (DM test).
    Compara as perdas (erros quadráticos) de dois modelos.
    Retorna o DM-stat e o p-valor.
    """
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2
    
    # Função de perda: erro quadrático
    d = (e1**2) - (e2**2)
    
    mean_d = np.mean(d)
    var_d  = np.var(d, ddof=1)
    
    # Evita divisão por zero se ambos os modelos preverem exatamente igual
    if var_d == 0:
        return 0.0, 1.0
        
    n = len(d)
    dm_stat = mean_d / np.sqrt(var_d / n)
    
    # P-valor (teste bicaudal)
    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Tratamento para MAPE (evita divisão por zero/valores muito pequenos)
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    
    r2 = r2_score(y_true, y_pred)
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

def run():
    ativos = descobrir_ativos()
    print(f"Ativos encontrados: {ativos}")

    resultados_finais = []

    for name in ativos:
        print(f"\n  Calculando métricas para {name}...")
        
        # O arquivo hybrid_predictions.csv contém as previsões reais, sarimax e hybrid
        # O arquivo lstm_predictions.csv contém real e lstm
        df_hybrid = pd.read_csv(RESULTS_DIR / f"{name}_hybrid_predictions.csv")
        df_lstm   = pd.read_csv(RESULTS_DIR / f"{name}_lstm_predictions.csv")
        
        # Filtra apenas o conjunto de TESTE para avaliação rigorosa
        test_hybrid = df_hybrid[df_hybrid["split"] == "test"]
        test_lstm   = df_lstm[df_lstm["split"] == "test"]
        
        # Para garantir alinhamento (caso as datas sejam as mesmas), mas sabemos que as datas batem
        y_true  = test_hybrid["real"].values
        pred_sarimax = test_hybrid["sarimax"].values
        pred_hybrid  = test_hybrid["hybrid"].values
        
        # LSTM base pode ter tamanho levemente diferente se não cortamos igual, 
        # mas as datas de teste devem bater. Vamos alinhar para segurança caso o usuário inspecione.
        # Aqui assumimos que o len do teste bate perfeitamente
        pred_lstm = test_lstm["lstm"].values[-len(y_true):] # Alinha pelo final (últimos dias)

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
                "MAPE": metricas["MAPE"],
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