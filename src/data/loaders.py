"""
Data loading utilities for Ag IQ ML project.
Adapted to actual file schemas.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_auction_results(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load and combine all auction result files.
    """
    data_path = Path(data_dir)
    
    # Find auction files
    auction_files = sorted(data_path.glob("auction_results_part*.xlsx"))
    
    if not auction_files:
        raise FileNotFoundError(f"No auction result files found in {data_dir}")
    
    print(f"Found {len(auction_files)} auction result files")
    
    dfs = []
    for f in auction_files:
        print(f"  Loading {f.name}...")
        df = pd.read_excel(f)
        dfs.append(df)
        print(f"    → {len(df):,} rows, {len(df.columns)} columns")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal: {len(combined):,} rows")
    
    return combined


def load_ag_barometer(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load Purdue Ag Economy Barometer data."""
    path = Path(data_dir) / "ag_economy_barometers.csv"
    df = pd.read_csv(path, sep='\t')  # Tab-delimited
    
    # Parse date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded ag_barometer: {len(df)} rows")
    return df


def load_diesel_prices(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load diesel price data."""
    path = Path(data_dir) / "diesel_prices.csv"
    df = pd.read_csv(path, sep='\t')  # Tab-delimited
    
    if 'month_date' in df.columns:
        df['month_date'] = pd.to_datetime(df['month_date'])
    
    print(f"Loaded diesel_prices: {len(df)} rows")
    return df


def load_el_nino(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load El Niño readings."""
    path = Path(data_dir) / "el_nino_readings.csv"
    df = pd.read_csv(path, sep='\t')  # Tab-delimited
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded el_nino: {len(df)} rows")
    return df


def load_future_prices(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load commodity futures prices."""
    path = Path(data_dir) / "future_prices.csv"
    df = pd.read_csv(path, sep='\t')  # Tab-delimited
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded future_prices: {len(df)} rows")
    print(f"  Symbols: {df['future_symbol'].unique().tolist() if 'future_symbol' in df.columns else 'N/A'}")
    return df


def load_makes(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load makes reference data."""
    data_path = Path(data_dir)
    makes_files = list(data_path.glob("makes_export*.xlsx"))
    
    if not makes_files:
        raise FileNotFoundError("No makes export file found")
    
    df = pd.read_excel(makes_files[0])
    print(f"Loaded makes: {len(df)} rows")
    return df


def load_auctioneers(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load auctioneers reference data."""
    data_path = Path(data_dir)
    auct_files = list(data_path.glob("auctioneers_export*.xlsx"))
    
    if not auct_files:
        raise FileNotFoundError("No auctioneers export file found")
    
    df = pd.read_excel(auct_files[0])
    print(f"Loaded auctioneers: {len(df)} rows")
    return df


def load_all_data(data_dir: str = "data/raw") -> dict:
    """Load all data files into a dictionary."""
    print("=" * 60)
    print("LOADING ALL DATA FILES")
    print("=" * 60 + "\n")
    
    return {
        'auctions': load_auction_results(data_dir),
        'barometer': load_ag_barometer(data_dir),
        'diesel': load_diesel_prices(data_dir),
        'el_nino': load_el_nino(data_dir),
        'futures': load_future_prices(data_dir),
        'makes': load_makes(data_dir),
        'auctioneers': load_auctioneers(data_dir),
    }

