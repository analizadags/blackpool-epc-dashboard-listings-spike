from __future__ import annotations
import pandas as pd

# Canonical columns the app expects
CANON = [
    "ADDRESS", "POSTCODE", "LAT", "LON",
    "EPC_CURRENT", "EPC_POTENTIAL", "UNITS_PER_BUILDING",
]

# Broad alias map (uppercased header names)
ALIASES = {
    "ADDRESS": {
        "ADDRESS", "ADDRESS1", "ADDRESS 1", "ADDRESS_LINE_1", "ADDRESS LINE 1",
        "FULL_ADDRESS", "PROPERTY_ADDRESS", "STREET_ADDRESS", "LINE1",
        "PRIMARY_ADDRESS", "PAON_SAON", "PAON", "SAON", "BUILDING_NAME",
        "STREET", "STREET_NAME", "PROPERTY", "LOCATION",
    },
    "POSTCODE": {"POSTCODE", "POST CODE", "POSTAL_CODE", "ZIP", "ZIPCODE", "OUTCODE+INCODE"},
    "LAT": {"LAT", "LATITUDE", "Y"},
    "LON": {"LON", "LONG", "LNG", "LONGITUDE", "X"},
    "EPC_CURRENT": {
        "EPC_CURRENT", "CURRENT_ENERGY_RATING", "EPC_RATING", "CURRENT_RATING",
        "CURRENT-ENERGY-RATING", "CURRENT_RR",
    },
    "EPC_POTENTIAL": {
        "EPC_POTENTIAL", "POTENTIAL_ENERGY_RATING", "POTENTIAL_RATING",
        "POTENTIAL-ENERGY-RATING",
    },
    "UNITS_PER_BUILDING": {
        "UNITS_PER_BUILDING", "UNITS", "NUM_UNITS", "NUMBER_OF_UNITS",
        "FLATS_PER_BUILDING", "NUM_FLATS", "NUMBER_OF_FLATS",
    },
}

# Helper lists to synthesize ADDRESS if missing
ADDRESS_PARTS_PRIORITY = [
    "ADDRESS", "ADDRESS1", "ADDRESS 1", "ADDRESS_LINE_1", "ADDRESS LINE 1",
    "SAON", "PAON", "BUILDING_NAME", "STREET", "STREET_NAME",
    "LOCALITY", "TOWN", "CITY",
]

REQUIRED_MIN = ["POSTCODE"]  # we'll try to build ADDRESS if it's missing


def _uc(s: str) -> str:
    return (s or "").strip().upper()


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_uc(c) for c in df.columns]
    return df


def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known aliases to canonical names (first match wins)."""
    existing = set(df.columns)
    for canon, alts in ALIASES.items():
        if canon in existing:
            continue
        for alt in alts:
            if alt in existing:
                df.rename(columns={alt: canon}, inplace=True)
                existing.add(canon)
                break
    return df


def _synthesize_address(df: pd.DataFrame) -> pd.DataFrame:
    """If ADDRESS is missing, try to build it from common parts."""
    if "ADDRESS" in df.columns:
        return df
    parts = [p for p in ADDRESS_PARTS_PRIORITY if p in df.columns]
    if not parts:
        return df  # we'll error-check later if still missing
    df["ADDRESS"] = (
        df[parts]
        .astype(str)
        .apply(lambda row: ", ".join([v for v in row if v and v.strip() and v != "nan"]), axis=1)
        .str.strip(", ")
    )
    return df


def load_epc_csv(file) -> pd.DataFrame:
    # Read as strings first (safer), we’ll coerce numerics after
    df = pd.read_csv(file, dtype=str, keep_default_na=False)
    df = _normalize_headers(df)
    df = _apply_aliases(df)
    df = _synthesize_address(df)

    # Minimal required columns
    missing_req = [c for c in REQUIRED_MIN if c not in df.columns]
    if missing_req:
        raise ValueError(f"Missing required columns: {missing_req}. Found columns: {list(df.columns)}")

    # Ensure canonical columns exist (create empty if absent)
    for col in CANON:
        if col not in df.columns:
            df[col] = ""

    # Coerce numerics
    for col in ["LAT", "LON"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "UNITS_PER_BUILDING" in df.columns:
        df["UNITS_PER_BUILDING"] = pd.to_numeric(df["UNITS_PER_BUILDING"], errors="coerce")

    return df
