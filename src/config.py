"""Constantes compartilhadas entre as etapas do pipeline.

Centralizar aqui evita que valores como a proporção do split temporal ou o
LOOKBACK divirjam entre os arquivos — um bug silencioso e difícil de achar.
"""

# --- Coleta de dados ---
START_DATE = "2019-01-01"
END_DATE   = "2026-05-22"
TICKERS = {
    "PETR4":  "PETR4.SA",
    "VALE3":  "VALE3.SA",
    "ITUB4":  "ITUB4.SA",
    "IBOV":   "^BVSP",
    "USDBRL": "BRL=X",
}

# --- Variáveis exógenas ---
EXOGENOUS      = ["IBOV", "USDBRL", "SELIC"]   # nomes brutos (arquivos *_raw.csv)
EXOGENOUS_COLS = ["ibov", "usdbrl", "selic"]   # nomes das colunas já processadas

# --- Alvo ---
TARGET_COL = "log_return"

# --- Split temporal cronológico (70% treino / 15% validação / 15% teste) ---
TRAIN_END = 0.70
VAL_END   = 0.85

# --- Hiperparâmetros base do LSTM / Híbrido ---
LOOKBACK   = 45
EPOCHS     = 300
BATCH_SIZE = 32
