# src/data.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/commodities.csv"

def load_dataset() -> pd.DataFrame:
    "Load dataset with date as index"
    df = pd.read_csv(DATA_PATH, delimiter=';', header=0, usecols=[0,1,2,3,4,5,6,7])
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')
    df.set_index('Date', inplace=True)
    df = df.asfreq('MS')
    return df

def load_commodity(commodity: str) -> pd.DataFrame:
    """Load each serie by commodity"""
    df = load_dataset()
    return df[[commodity]].dropna()