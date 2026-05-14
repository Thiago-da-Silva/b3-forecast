import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Resíduos", page_icon="🎯", layout="wide")
st.title("Análise de Resíduos")

RESIDUALS_DIR = Path("evaluation/residuals")

st.markdown("Um bom modelo de previsão tem resíduos que se assemelham a um ruído branco (distribuição normal, centrada em zero). O **Histograma** e o **Q-Q Plot** ajudam nessa verificação.")

ativo = st.selectbox("Selecione o Ativo:", ["PETR4", "VALE3", "ITUB4"])

try:
    # O arquivo gerado no híbrido contém os resíduos do sarimax (res_real) 
    # e os resíduos previstos pelo LSTM (res_previsto_lstm)
    path = RESIDUALS_DIR / f"{ativo}_residuals.csv"
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df_test = df[df["split"] == "test"]
    
    # Resíduo do SARIMAX já existe no CSV ("res_real")
    # O Resíduo Final do Híbrido = Resíduo Sarimax - Previsão do LSTM
    res_sarimax = df_test["res_real"]
    res_hybrid  = df_test["res_real"] - df_test["res_previsto_lstm"]
    
    col_grafico, col_qq = st.columns(2)
    
    with col_grafico:
        st.subheader("Histograma de Resíduos (SARIMAX vs Híbrido)")
        
        hist_df = pd.DataFrame({
            "SARIMAX": res_sarimax,
            "Híbrido": res_hybrid
        }).melt(var_name="Modelo", value_name="Resíduo")
        
        fig_hist = px.histogram(
            hist_df, 
            x="Resíduo", 
            color="Modelo", 
            barmode="overlay",
            nbins=50,
            opacity=0.7
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_qq:
        st.subheader("Q-Q Plot (Híbrido)")
        # Criação manual do QQ Plot via plotly
        osm, osr = stats.probplot(res_hybrid, dist="norm", fit=True)
        
        fig_qq = go.Figure()
        # Dispersão teórica vs observada
        fig_qq.add_trace(go.Scatter(x=osm[0], y=osm[1], mode='markers', name='Quantis Observados'))
        
        # Linha teórica ideal (vermelha)
        linha_x = np.array([min(osm[0]), max(osm[0])])
        linha_y = osr[0] * linha_x + osr[1]
        fig_qq.add_trace(go.Scatter(x=linha_x, y=linha_y, mode='lines', name='Linha Teórica (Normal)', line=dict(color='red')))
        
        st.plotly_chart(fig_qq, use_container_width=True)

except FileNotFoundError:
    st.warning("Arquivo de resíduos não encontrado. Rode o script de treinamento do Modelo Híbrido primeiro.")