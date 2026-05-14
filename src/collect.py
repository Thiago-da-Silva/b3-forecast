import yfinance as yf
from pathlib import Path

START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
TICKERS = {
    "PETR4": "PETR4.SA",
    "VALE3": "VALE3.SA",
    "ITUB4": "ITUB4.SA",
    "IBOV":  "^BVSP",
    "USDBRL":"BRL=X",
    "SELIC": "LFTS11.SA", # ETF atrelado à taxa Selic
}

def run():
    for name, symbol in TICKERS.items():
        path = Path(f"data/raw/{name}_raw.csv")
        df = yf.Ticker(symbol).history(
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            auto_adjust=True,
        )
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index = df.index.tz_localize(None)
        df.to_csv(path)
        print(f"  Salvo: {path} ({len(df)} linhas)")

