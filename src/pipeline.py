import sys
from pathlib import Path

# Adiciona o diretório raiz do projeto ao sys.path para permitir importações absolutas
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Importando todas as etapas do nosso projeto
import src.collect as collect
import src.preprocess as preprocess
import src.train_sarimax as train_sarimax
import src.train_lstm as train_lstm
import src.train_hybrid as train_hybrid
import src.evaluate as evaluate

def run():
    print("="*50)
    print(" INICIANDO PIPELINE - PREVISÃO B3 (SARIMAX vs LSTM)")
    print("="*50)

    print("\n[1/6] Coletando dados (yfinance)...")
    collect.run()

    print("\n[2/6] Pré-processando dados e calculando features técnicas...")
    preprocess.run()

    print("\n[3/6] Treinando modelo SARIMAX (Baseline)...")
    train_sarimax.run()

    print("\n[4/6] Treinando rede neural LSTM...")
    train_lstm.run()

    print("\n[5/6] Treinando modelo Híbrido (Resíduos LSTM + SARIMAX)...")
    train_hybrid.run()

    print("\n[6/6] Avaliando modelos e aplicando Teste Diebold-Mariano...")
    evaluate.run()

    print("\n" + "="*50)
    print(" PIPELINE CONCLUÍDO COM SUCESSO! 🎉")
    print(" Para visualizar os resultados, execute no seu terminal:")
    print(" .\\.venv\\Scripts\\streamlit run dashboard/app.py")
    print("="*50)

if __name__ == "__main__":
    run()