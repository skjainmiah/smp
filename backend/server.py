# backend/server.py
"""
Tiny FastAPI backend to serve live stock recommendations
from live_recommendations.csv file for the website frontend.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path

app = FastAPI(title="Stock Recommendations API", version="1.0")

# Allow Bolt or other web origins to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your bolt domain later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = Path("results/live_recommendations.csv")

def load_data():
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="CSV file not found")
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

@app.get("/")
def root():
    return {"message": "Stock Recommendations API running ✅"}

@app.get("/api/recommendations/today")
def get_all_recommendations():
    """Return all current stock recommendations."""
    df = load_data()
    data = df.to_dict(orient="records")
    return {"count": len(data), "data": data}

@app.get("/api/stocks/{symbol}")
def get_stock(symbol: str):
    """Return single stock details."""
    df = load_data()
    df_symbol = df[df["symbol"].str.upper() == symbol.upper()]
    if df_symbol.empty:
        raise HTTPException(status_code=404, detail="Symbol not found")
    return df_symbol.to_dict(orient="records")[0]
