import yfinance as yf
from bcb import sgs
from pathlib import Path

START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
TICKERS = {
    "PETR4": "PETR4.SA",
    "VALE3": "VALE3.SA",
    "ITUB4": "ITUB4.SA",
    "IBOV":  "^BVSP",
    "USDBRL":"BRL=X",
}

def run():
    project_root = Path(__file__).resolve().parent.parent
    
    # 1. Coleta dados via yfinance
    for name, symbol in TICKERS.items():
        path = project_root / f"data/raw/{name}_raw.csv"
        df = yf.Ticker(symbol).history(
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            auto_adjust=True,
        )
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index = df.index.tz_localize(None)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        print(f"  Salvo: {path} ({len(df)} linhas)")

    # 2. Coleta SELIC via python-bcb (SGS código 11)
    selic_path = project_root / "data/raw/SELIC_raw.csv"
    df_selic = sgs.get({"Close": 11}, start=START_DATE, end=END_DATE)
    selic_path.parent.mkdir(parents=True, exist_ok=True)
    df_selic.to_csv(selic_path, index_label="Date")
    print(f"  Salvo: {selic_path} ({len(df_selic)} linhas)")

