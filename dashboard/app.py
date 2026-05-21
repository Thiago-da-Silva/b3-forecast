import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(
    page_title="Previsão B3 - Dashboard",
    page_icon="📈",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "predictions"
METRICS_FILE = PROJECT_ROOT / "evaluation" / "metrics.csv"

@st.cache_data
def carregar_dados():
    # Descobre os ativos a partir dos arquivos de previsão, em vez de fixar a lista
    ativos = sorted({
        p.stem.replace("_hybrid_predictions", "")
        for p in RESULTS_DIR.glob("*_hybrid_predictions.csv")
    })
    modelos = {
        "SARIMAX": "sarimax",
        "LSTM": "lstm",
        "Híbrido": "hybrid"
    }
    
    df_all = []
    
    for ativo in ativos:
        # Carrega o CSV que tem real, sarimax, hybrid
        try:
            df_h = pd.read_csv(RESULTS_DIR / f"{ativo}_hybrid_predictions.csv", index_col="Date", parse_dates=True)
            df_l = pd.read_csv(RESULTS_DIR / f"{ativo}_lstm_predictions.csv", index_col="Date", parse_dates=True)
            
            # Realinhar e unificar
            df_temp = pd.DataFrame(index=df_h.index)
            df_temp["Ativo"] = ativo
            df_temp["real"] = df_h["real"]
            df_temp["split"] = df_h["split"]
            df_temp["SARIMAX"] = df_h["sarimax"]
            df_temp["Híbrido"] = df_h["hybrid"]
            
            # O LSTM pode ter tamanho diferente por causa do lookback, vamos alinhar pelo índice
            df_temp = df_temp.join(df_l["lstm"].rename("LSTM"), how="left")
            
            df_all.append(df_temp)
        except Exception as e:
            continue
            
    if not df_all:
        return pd.DataFrame()
        
    return pd.concat(df_all)

@st.cache_data
def carregar_metricas():
    try:
        return pd.read_csv(METRICS_FILE)
    except FileNotFoundError:
        return pd.DataFrame()

df_completo = carregar_dados()
df_metricas = carregar_metricas()

st.title("📈 Previsão de Ações da B3")

if df_completo.empty:
    st.error("Dados de previsão não encontrados. Execute o pipeline primeiro.")
    st.stop()

# --- Configuração do Sidebar ---
st.sidebar.header("Filtros")

ativos_disponiveis = df_completo["Ativo"].unique().tolist()
ativos_selecionados = st.sidebar.multiselect("Selecione os Ativos:", ativos_disponiveis, default=ativos_disponiveis)

modelos_disponiveis = ["SARIMAX", "LSTM", "Híbrido"]
modelos_selecionados = st.sidebar.multiselect("Selecione os Modelos:", modelos_disponiveis, default=["Híbrido"])

# --- Filtro de Datas ---
min_date = df_completo.index.min().date()
max_date = df_completo.index.max().date()

start_date, end_date = st.sidebar.date_input(
    "Selecione o Período:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# --- Filtro de Splits ---
splits = st.sidebar.multiselect("Splits dos Dados:", ["train", "val", "test"], default=["test"])

# --- Aplicar Filtros ---
mask = (
    (df_completo.index >= start_date) & 
    (df_completo.index <= end_date) &
    (df_completo["Ativo"].isin(ativos_selecionados)) &
    (df_completo["split"].isin(splits))
)

df_filtrado = df_completo[mask]

# --- Info Card de Datas ---
st.markdown("### 📅 Informações da Divisão dos Dados")
cols_info = st.columns(3)
splits_info = ["train", "val", "test"]
nomes_splits = ["Treino (70%)", "Validação (15%)", "Teste (15%)"]

for col, split_code, nome in zip(cols_info, splits_info, nomes_splits):
    df_split = df_completo[df_completo["split"] == split_code]
    if not df_split.empty:
        d_inicio = df_split.index.min().strftime('%d/%m/%Y')
        d_fim = df_split.index.max().strftime('%d/%m/%Y')
        qtd_dias = len(df_split.index.unique())
    else:
        d_inicio, d_fim, qtd_dias = "N/A", "N/A", 0
        
    with col:
        st.info(f"**{nome}**\n\nPeríodo: {d_inicio} a {d_fim}\n\nDias Úteis: {qtd_dias}")

# --- Gráficos ---
st.markdown("---")
st.markdown("### 📈 Série Temporal: Real vs Previsto")

if df_filtrado.empty:
    st.warning("Nenhum dado encontrado para os filtros selecionados.")
else:
    for ativo in ativos_selecionados:
        df_ativo = df_filtrado[df_filtrado["Ativo"] == ativo]
        if df_ativo.empty:
            continue
            
        fig = go.Figure()
        
        # Adiciona a linha real
        fig.add_trace(go.Scatter(x=df_ativo.index, y=df_ativo["real"], name="Real", line=dict(color='black', width=2)))
        
        # Adiciona os modelos selecionados
        cores = {"SARIMAX": "blue", "LSTM": "orange", "Híbrido": "red"}
        for modelo in modelos_selecionados:
            if modelo in df_ativo.columns:
                fig.add_trace(go.Scatter(x=df_ativo.index, y=df_ativo[modelo], name=modelo, line=dict(color=cores.get(modelo, 'green'))))
                
        fig.update_layout(
            title=f"Ativo: {ativo}",
            xaxis_title="Data",
            yaxis_title="Retorno log (padronizado)",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Métricas ---
if not df_metricas.empty:
    st.markdown("---")
    st.markdown("### 📊 Tabela de Métricas (Baseada no conjunto de Teste)")
    
    df_metricas_view = df_metricas[df_metricas["Ativo"].isin(ativos_selecionados)]
    if not df_metricas_view.empty:
        st.dataframe(
            df_metricas_view.style.format({
                "MAE": "{:.5f}",
                "RMSE": "{:.5f}",
                "sMAPE": "{:.2f}%",
                "Theil_U": "{:.4f}",
                "R2": "{:.4f}",
                "Hit_Rate": "{:.2f}%",
                "DM_Stat_vs_Base": "{:.4f}",
                "DM_PValue": "{:.5f}"
            }),
            use_container_width=True
        )
