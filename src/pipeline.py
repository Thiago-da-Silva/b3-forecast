import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import src.collect       as collect
import src.preprocess    as preprocess
import src.train_sarimax as train_sarimax
import src.tune_lstm     as tune_lstm
import src.train_lstm    as train_lstm
import src.train_hybrid  as train_hybrid
import src.evaluate      as evaluate

def run():
    print("="*50)
    print(" INICIANDO PIPELINE - PREVISÃO B3")
    print("="*50)

    print("\n[1/7] Coletando dados (yfinance + BCB)...")
    collect.run()

    print("\n[2/7] Pré-processando dados e calculando features técnicas...")
    preprocess.run()

    print("\n[3/7] Treinando modelo SARIMAX (Baseline)...")
    train_sarimax.run()

    print("\n[4/7] Tunando hiperparâmetros LSTM por ativo...")
    tune_lstm.run_tuning()

    print("\n[5/7] Treinando rede neural LSTM...")
    train_lstm.run()

    print("\n[6/7] Treinando modelo Híbrido (SARIMAX + LSTM Residual)...")
    train_hybrid.run()

    print("\n[7/7] Avaliando modelos e aplicando Teste Diebold-Mariano...")
    evaluate.run()

    print("\n" + "="*50)
    print(" PIPELINE CONCLUÍDO COM SUCESSO!")
    print(" Execute para visualizar:")
    print(" streamlit run dashboard/app.py")
    print("="*50)

if __name__ == "__main__":
    run()