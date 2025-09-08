from __future__ import annotations
import pandas as pd

REQUIRED_COLS = [
    "ADDRESS", "POSTCODE", "LAT", "LON",
    "EPC_CURRENT", "EPC_POTENTIAL", "UNITS_PER_BUILDING"
]

def load_epc_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # normalize columns
    df.columns = [c.strip().upper() for c in df.columns]
    # coerce numerics
    for col in ["LAT", "LON", "UNITS_PER_BUILDING"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # ensure required
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

