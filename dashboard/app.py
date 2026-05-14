import streamlit as st

st.set_page_config(
    page_title="Previsão B3 - Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Previsão de Ações da B3")
st.markdown("""
Este painel interativo apresenta os resultados do estudo comparativo entre modelos estatísticos (**SARIMAX**) e redes neurais (**LSTM** e **Híbrido**) para previsão de ativos da B3.

### 📌 Sobre o Projeto
O mercado de ações é complexo e ruidoso. Este projeto testa a hipótese de que a combinação da inteligência estatística clássica para lidar com a sazonalidade e tendência, unida à capacidade de aprendizado profundo (Deep Learning) para padrões não lineares, melhora a precisão preditiva.

### 🧭 Como navegar
Utilize o menu lateral para visualizar os resultados:
- **Série Temporal:** Compare visualmente os dados reais de fechamento com a predição de cada modelo.
- **Métricas:** Tabela consolidada com cálculo de erros (MAE, RMSE) e o Teste de Significância de Diebold-Mariano.
- **Erro Temporal:** Análise de quão estável foi o erro do modelo ao longo da janela de teste.
- **Resíduos:** Verificação das premissas estatísticas (distribuição do erro) usando Histogramas e Q-Q Plots.

---
**Autores:** Henrique Kuroda, Thiago Silva — Engenharia da Computação, Fundação Hermínio Ometto, Araras, Brasil.
""")
