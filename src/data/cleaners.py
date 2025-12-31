"""
Data cleaning utilities with actual schema column names.
"""

import pandas as pd
import numpy as np
from typing import Optional


def clean_auction_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean auction results data.
    """
    df = df.copy()
    print(f"Starting with {len(df):,} records")
    
    # Parse dates
    df['sold_date'] = pd.to_datetime(df['sold_date'], errors='coerce')
    
    # Clean price
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    invalid_price = (df['price'] <= 0) | df['price'].isna()
    print(f"  Invalid prices: {invalid_price.sum():,}")
    
    # Clean hours
    df['hours'] = pd.to_numeric(df['hours'], errors='coerce')
    df.loc[df['hours'] < 0, 'hours'] = np.nan
    df.loc[df['hours'] > 100000, 'hours'] = np.nan
    
    # Clean year (model year)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    current_year = pd.Timestamp.now().year
    df.loc[df['year'] < 1950, 'year'] = np.nan
    df.loc[df['year'] > current_year + 1, 'year'] = np.nan
    
    # Standardize region
    if 'region' in df.columns:
        df['region'] = df['region'].astype(str).str.lower().str.strip()
        df.loc[df['region'].isin(['nan', 'none', '']), 'region'] = np.nan
    
    # Standardize state
    if 'state' in df.columns:
        df['state'] = df['state'].astype(str).str.upper().str.strip()
        df.loc[df['state'].isin(['NAN', 'NONE', '']), 'state'] = np.nan
    
    # Standardize make_key
    if 'make_key' in df.columns:
        df['make_key'] = df['make_key'].astype(str).str.lower().str.strip()
        df.loc[df['make_key'].isin(['nan', 'none', '']), 'make_key'] = np.nan
    
    print(f"Cleaning complete: {len(df):,} records")
    return df


def filter_training_data(
    df: pd.DataFrame,
    min_price: int = 1000,
    max_price: int = 2_000_000,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    require_fields: list = None
) -> pd.DataFrame:
    """
    Filter data to records suitable for training.
    """
    df = df.copy()
    initial = len(df)
    
    # Price filters
    df = df[(df['price'] >= min_price) & (df['price'] <= max_price)]
    print(f"After price filter: {len(df):,} (removed {initial - len(df):,})")
    
    # Date filters
    if min_date:
        before = len(df)
        df = df[df['sold_date'] >= pd.to_datetime(min_date)]
        print(f"After min_date filter: {len(df):,} (removed {before - len(df):,})")
    
    if max_date:
        before = len(df)
        df = df[df['sold_date'] <= pd.to_datetime(max_date)]
        print(f"After max_date filter: {len(df):,} (removed {before - len(df):,})")
    
    # Required fields
    if require_fields:
        for field in require_fields:
            if field in df.columns:
                before = len(df)
                df = df[df[field].notna()]
                print(f"After requiring {field}: {len(df):,} (removed {before - len(df):,})")
    
    print(f"\nFiltered from {initial:,} to {len(df):,} records")
    return df

