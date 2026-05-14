import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Métricas", page_icon="📊", layout="wide")
st.title("Tabela de Métricas e Comparação Estatística")

METRICS_FILE = Path("evaluation/metrics.csv")

try:
    df = pd.read_csv(METRICS_FILE)
    
    st.markdown("### Resumo de Erros no Conjunto de Teste")
    st.markdown("Aqui comparamos o Erro Absoluto Médio (**MAE**), Raiz do Erro Quadrático (**RMSE**), Erro Percentual (**MAPE**) e o Teste **Diebold-Mariano**.")
    
    ativo_filter = st.selectbox("Filtrar por Ativo (ou Todos):", ["Todos", "PETR4", "VALE3", "ITUB4"])
    
    if ativo_filter != "Todos":
        df_view = df[df["Ativo"] == ativo_filter]
    else:
        df_view = df.copy()
        
    # Formatação visual da tabela
    st.dataframe(
        df_view.style.format({
            "MAE": "{:.5f}",
            "RMSE": "{:.5f}",
            "MAPE": "{:.2f}%",
            "R2": "{:.4f}",
            "DM_Stat_vs_Base": "{:.4f}",
            "DM_PValue": "{:.5f}"
        }),
        use_container_width=True
    )
    
    st.info("💡 **Dica sobre Diebold-Mariano:** Um `DM_PValue` menor que `0.05` indica que o modelo testado é ESTATISTICAMENTE DIFERENTE do SARIMAX. Se a Estatística DM (`DM_Stat_vs_Base`) for POSITIVA, o erro do SARIMAX é maior, o que significa que o modelo testado (LSTM ou Híbrido) **venceu** o baseline.")

except FileNotFoundError:
    st.warning("O arquivo `metrics.csv` não foi encontrado na pasta `evaluation/`.")