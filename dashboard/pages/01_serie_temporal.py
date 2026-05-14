import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Série Temporal", page_icon="📈", layout="wide")
st.title("Série Temporal: Real vs Previsto")

RESULTS_DIR = Path("evaluation/predictions")

@st.cache_data
def carregar_dados(ativo, modelo):
    if modelo == "Híbrido":
        path = RESULTS_DIR / f"{ativo}_hybrid_predictions.csv"
        coluna_pred = "hybrid"
    elif modelo == "LSTM":
        path = RESULTS_DIR / f"{ativo}_lstm_predictions.csv"
        coluna_pred = "lstm"
    else:
        path = RESULTS_DIR / f"{ativo}_sarimax_predictions.csv"
        coluna_pred = "sarimax"
        
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df, coluna_pred

col1, col2 = st.columns(2)
with col1:
    ativo = st.selectbox("Selecione o Ativo:", ["PETR4", "VALE3", "ITUB4"])
with col2:
    modelo = st.selectbox("Selecione o Modelo:", ["SARIMAX", "LSTM", "Híbrido"])

try:
    df, col_pred = carregar_dados(ativo, modelo)
    
    # Filtro opcional por split (Geralmente nos interessa mais o Teste)
    split_choice = st.radio("Período de visualização:", ["Apenas Teste (Recomendado)", "Validacao e Teste", "Completo (Treino, Val, Teste)"])
    
    if split_choice == "Apenas Teste (Recomendado)":
        df = df[df["split"] == "test"]
    elif split_choice == "Validacao e Teste":
        df = df[df["split"].isin(["val", "test"])]
        
    fig = px.line(
        df, 
        y=["real", col_pred], 
        labels={"value": "Retorno Logarítmico", "Date": "Data", "variable": "Legenda"},
        title=f"Desempenho do modelo {modelo} no ativo {ativo}"
    )
    
    fig.data[0].name = "Valor Real"
    fig.data[1].name = f"Previsão ({modelo})"
    
    st.plotly_chart(fig, use_container_width=True)
    
except FileNotFoundError:
    st.error(f"Arquivo de previsão não encontrado para o ativo {ativo} e modelo {modelo}. Execute as etapas anteriores do pipeline.")