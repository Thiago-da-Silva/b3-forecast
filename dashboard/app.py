import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
METRICS_FILE = PROJECT_ROOT / "evaluation" / "metrics.csv"

def unscale_to_pct(ativo, series):
    """Desfaz a padronização e converte de log return para variação percentual"""
    try:
        with open(ARTIFACTS_DIR / f"{ativo}_ret_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        # Ignora NaNs se houver
        valid_mask = ~series.isna()
        if not valid_mask.any():
            return series
            
        unscaled = scaler.inverse_transform(series[valid_mask].values.reshape(-1, 1)).ravel()
        pct_change = (np.exp(unscaled) - 1) * 100
        
        result = series.copy()
        result.loc[valid_mask] = pct_change
        return result
    except Exception as e:
        return series

@st.cache_data
def carregar_dados():
    ativos = sorted({
        p.stem.replace("_hybrid_predictions", "")
        for p in RESULTS_DIR.glob("*_hybrid_predictions.csv")
    })
    
    df_all = []
    
    for ativo in ativos:
        try:
            df_h = pd.read_csv(RESULTS_DIR / f"{ativo}_hybrid_predictions.csv", index_col="Date", parse_dates=True)
            df_l = pd.read_csv(RESULTS_DIR / f"{ativo}_lstm_predictions.csv", index_col="Date", parse_dates=True)
            
            df_temp = pd.DataFrame(index=df_h.index)
            df_temp["Ativo"] = ativo
            df_temp["real"] = df_h["real"]
            df_temp["split"] = df_h["split"]
            df_temp["SARIMAX"] = df_h["sarimax"]
            df_temp["Híbrido"] = df_h["hybrid"]
            df_temp = df_temp.join(df_l["lstm"].rename("LSTM"), how="left")
            
            # Criar as versões em percentual diário
            df_temp["real_pct"] = unscale_to_pct(ativo, df_temp["real"])
            df_temp["SARIMAX_pct"] = unscale_to_pct(ativo, df_temp["SARIMAX"])
            df_temp["Híbrido_pct"] = unscale_to_pct(ativo, df_temp["Híbrido"])
            df_temp["LSTM_pct"] = unscale_to_pct(ativo, df_temp["LSTM"])
            
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

tipo_grafico = st.sidebar.radio(
    "Tipo de Gráfico:",
    ["Variação Percentual Diária (%)", "Retorno Acumulado (%)", "Retorno Log Padronizado"]
)

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
st.markdown(f"### 📈 Série Temporal: {tipo_grafico}")

if df_filtrado.empty:
    st.warning("Nenhum dado encontrado para os filtros selecionados.")
else:
    for ativo in ativos_selecionados:
        df_ativo = df_filtrado[df_filtrado["Ativo"] == ativo].copy()
        if df_ativo.empty:
            continue
            
        fig = go.Figure()
        
        # Mapeamento das colunas dependendo do tipo de gráfico
        if tipo_grafico == "Variação Percentual Diária (%)":
            col_real = "real_pct"
            y_title = "Variação Diária (%)"
        elif tipo_grafico == "Retorno Acumulado (%)":
            # Para o retorno acumulado, calculamos a soma cumulativa dos retornos percentuais logarítmicos
            # Ou de forma mais simples: (prod(1 + retorno_diario) - 1) * 100
            df_ativo["real_cum"] = ((1 + df_ativo["real_pct"]/100).cumprod() - 1) * 100
            for mod in modelos_disponiveis:
                if f"{mod}_pct" in df_ativo.columns:
                    df_ativo[f"{mod}_cum"] = ((1 + df_ativo[f"{mod}_pct"]/100).cumprod() - 1) * 100
            col_real = "real_cum"
            y_title = "Acumulado no Período Selecionado (%)"
        else:
            col_real = "real"
            y_title = "Retorno log (padronizado)"

        if tipo_grafico == "Variação Percentual Diária (%)":
            fig.add_trace(go.Bar(x=df_ativo.index, y=df_ativo[col_real], name="Real", marker_color='black'))
        else:
            fig.add_trace(go.Scatter(x=df_ativo.index, y=df_ativo[col_real], name="Real", line=dict(color='black', width=2)))
        
        cores = {"SARIMAX": "blue", "LSTM": "orange", "Híbrido": "red"}
        for modelo in modelos_selecionados:
            col_modelo = f"{modelo}_pct" if tipo_grafico == "Variação Percentual Diária (%)" else (f"{modelo}_cum" if tipo_grafico == "Retorno Acumulado (%)" else modelo)
            if col_modelo in df_ativo.columns:
                if tipo_grafico == "Variação Percentual Diária (%)":
                    fig.add_trace(go.Bar(x=df_ativo.index, y=df_ativo[col_modelo], name=modelo, marker_color=cores.get(modelo, 'green')))
                else:
                    fig.add_trace(go.Scatter(x=df_ativo.index, y=df_ativo[col_modelo], name=modelo, line=dict(color=cores.get(modelo, 'green'))))
                
        if tipo_grafico == "Variação Percentual Diária (%)":
            fig.update_layout(barmode='group')

        fig.update_layout(
            title=f"Ativo: {ativo}",
            xaxis_title="Data",
            yaxis_title=y_title,
            hovermode="x unified"
        )
        
        if "%)" in y_title:
            fig.update_yaxes(ticksuffix="%", tickformat="+.2f")

        st.plotly_chart(fig, use_container_width=True)

# --- Métricas ---
if not df_metricas.empty:
    st.markdown("---")
    st.markdown("### 📊 Tabela de Métricas (Baseada no conjunto de Teste)")
    
    df_metricas_view = df_metricas[df_metricas["Ativo"].isin(ativos_selecionados)]
    if not df_metricas_view.empty:
        col_fmt = {
            "MAE": "{:.5f}",
            "RMSE": "{:.5f}",
            "sMAPE": "{:.2f}%",
            "Theil_U": "{:.4f}",
            "R2": "{:.4f}",
            "Hit_Rate": "{:.2f}%",
            "MAE_BRL": "R$ {:.2f}",
            "RMSE_BRL": "R$ {:.2f}",
            "MAPE_BRL": "{:.2f}%",
            "DM_Stat_vs_Base": "{:.4f}",
            "DM_PValue": "{:.5f}",
        }
        col_fmt = {k: v for k, v in col_fmt.items() if k in df_metricas_view.columns}
        st.dataframe(
            df_metricas_view.style.format(col_fmt),
            use_container_width=True
        )

    # --- Glossário de Métricas ---
    with st.expander("📖 Entenda as Métricas"):
        st.markdown("""
        * **MAE (Erro Médio Absoluto):** A média da distância entre a previsão e o real na escala original matemática.
        * **RMSE (Raiz do Erro Quadrático Médio):** Similar ao MAE, mas pune erros muito grandes (pontos muito fora da curva).
        * **sMAPE:** Erro Percentual Absoluto Médio Simétrico. É o MAE em versão de porcentagem.
        * **Theil_U:** Métrica de "Inteligência". Compara o modelo com um palpite ingênuo (chutar sempre zero). Se for menor que 1.0, o modelo é útil.
        * **R2 (R-Quadrado):** Mostra quanto da variação real o modelo explica. Em bolsa de valores, quase sempre é negativo para retornos diários.
        * **Hit_Rate (%):** **Taxa de Acerto de Direção.** Se o modelo acertar se vai subir ou cair, conta ponto. Acima de 55% é excelente no mercado financeiro.
        * **MAE_BRL / RMSE_BRL:** O mesmo erro calculado, mas convertido e traduzido para dinheiro real (R$). Ex: "Erra a previsão por R$ 0,50".
        * **MAPE_BRL:** O erro percentual de precisão quando tentamos adivinhar o preço em R$ de fechamento da ação para amanhã.
        * **DM_Stat_vs_Base / DM_PValue:** Teste científico de Diebold-Mariano. Avalia se a vitória do modelo contra o SARIMAX foi por mérito ou apenas "sorte". Se PValue < 0.05, a vitória é estatisticamente comprovada.
        """)