from __future__ import annotations
import pandas as pd

# We accept several common header names and normalize them
ALIASES = {
    "ADDRESS": {"ADDRESS", "ADDRESS1", "FULL_ADDRESS"},
    "POSTCODE": {"POSTCODE", "POST CODE", "ZIP", "ZIPCODE"},
    "LAT": {"LAT", "LATITUDE"},
    "LON": {"LON", "LONG", "LNG", "LONGITUDE"},
    "EPC_CURRENT": {"EPC_CURRENT", "CURRENT_ENERGY_RATING", "EPC_RATING", "CURRENT_RATING"},
    "EPC_POTENTIAL": {"EPC_POTENTIAL", "POTENTIAL_ENERGY_RATING", "POTENTIAL_RATING"},
    "UNITS_PER_BUILDING": {"UNITS_PER_BUILDING", "UNITS", "NUM_UNITS", "FLATS_PER_BUILDING"},
}

REQUIRED = ["ADDRESS", "POSTCODE", "LAT", "LON"]  # hard requirements

def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]

    # Map aliases to canonical names
    existing = set(df.columns)
    for canon, alts in ALIASES.items():
        if canon in existing:
            continue
        for alt in alts:
            if alt in existing:
                df.rename(columns={alt: canon}, inplace=True)
                break
    return df

def load_epc_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file, dtype=str)  # read as string first to avoid coercion issues
    df = _normalize_headers(df)

    # Check hard-required columns
    missing_req = [c for c in REQUIRED if c not in df.columns]
    if missing_req:
        raise ValueError(f"Missing required columns: {missing_req}")

    # Numeric coercions (best-effort)
    for col in ["LAT", "LON"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Optional columns: if absent, create empties (filled later in app if needed)
    if "EPC_CURRENT" not in df.columns:
        df["EPC_CURRENT"] = ""
    if "EPC_POTENTIAL" not in df.columns:
        df["EPC_POTENTIAL"] = ""
    if "UNITS_PER_BUILDING" in df.columns:
        df["UNITS_PER_BUILDING"] = pd.to_numeric(df["UNITS_PER_BUILDING"], errors="coerce")

    return df
