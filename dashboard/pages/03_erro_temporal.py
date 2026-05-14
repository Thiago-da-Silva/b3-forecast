import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Erro Temporal", page_icon="⏳", layout="wide")
st.title("Evolução do Erro no Tempo (Janela Móvel)")

RESULTS_DIR = Path("evaluation/predictions")

st.markdown("Esta página exibe a evolução do Erro Absoluto ao longo do período de **Teste**. Útil para identificar se o modelo se degradou com alguma mudança brusca no mercado.")

col1, col2 = st.columns(2)
with col1:
    ativo = st.selectbox("Selecione o Ativo:", ["PETR4", "VALE3", "ITUB4"])
with col2:
    modelo = st.selectbox("Selecione o Modelo:", ["SARIMAX", "LSTM", "Híbrido"])

try:
    if modelo == "Híbrido":
        coluna_pred = "hybrid"
        path = RESULTS_DIR / f"{ativo}_hybrid_predictions.csv"
    elif modelo == "LSTM":
        coluna_pred = "lstm"
        path = RESULTS_DIR / f"{ativo}_lstm_predictions.csv"
    else:
        coluna_pred = "sarimax"
        path = RESULTS_DIR / f"{ativo}_sarimax_predictions.csv"

    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df_test = df[df["split"] == "test"].copy()
    
    # Calcula erro absoluto para cada dia
    df_test["Erro Absoluto"] = abs(df_test["real"] - df_test[coluna_pred])
    
    # Aplica uma média móvel suave de 5 dias (1 semana útil) para remover ruído
    df_test["Erro Suavizado (SMA 5)"] = df_test["Erro Absoluto"].rolling(5).mean()
    
    fig = px.line(
        df_test,
        y=["Erro Absoluto", "Erro Suavizado (SMA 5)"],
        labels={"value": "Erro (log return)", "Date": "Data"},
        title=f"Evolução do Erro Absoluto - {ativo} ({modelo})"
    )
    
    fig.data[0].line.color = 'rgba(255, 0, 0, 0.3)' # Vermelho transparente para os picos
    fig.data[1].line.color = 'rgba(0, 0, 255, 1.0)' # Azul sólido para a média móvel
    
    st.plotly_chart(fig, use_container_width=True)

except FileNotFoundError:
    st.warning("O arquivo de previsão não foi encontrado. Certifique-se de ter rodado o pipeline.")