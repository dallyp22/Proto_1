"""
Data processing and merging with actual schema column names.
"""

import pandas as pd
import numpy as np
from typing import Optional


def merge_with_makes(
    auctions: pd.DataFrame,
    makes: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge auction data with makes reference using make_key.
    """
    # Standardize make_key in makes table
    makes = makes.copy()
    makes['make_key'] = makes['make_key'].astype(str).str.lower().str.strip()
    
    # Select relevant columns
    make_cols = [
        'make_key', 'name', 'still_in_business', 
        'headquarters_country_iso_2',
        'production_year_start', 'production_year_end'
    ]
    make_cols = [c for c in make_cols if c in makes.columns]
    makes_subset = makes[make_cols].copy()
    
    # Rename to avoid conflicts
    makes_subset = makes_subset.rename(columns={
        'name': 'make_name',
        'still_in_business': 'make_still_in_business',
        'headquarters_country_iso_2': 'make_country',
        'production_year_start': 'make_production_year_start',
        'production_year_end': 'make_production_year_end'
    })
    
    # Merge
    merged = auctions.merge(makes_subset, on='make_key', how='left')
    print(f"Merged with makes: {merged['make_name'].notna().sum():,} matches")
    
    return merged


def merge_with_macro_data(
    auctions: pd.DataFrame,
    barometer: pd.DataFrame,
    diesel: pd.DataFrame,
    el_nino: pd.DataFrame,
    futures: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge auction data with macro indicators by month.
    """
    df = auctions.copy()
    
    # Create month key
    df['sale_month'] = df['sold_date'].dt.to_period('M').dt.to_timestamp()
    
    # --- Merge Barometer ---
    if 'date' in barometer.columns:
        baro = barometer.copy()
        baro['month_key'] = pd.to_datetime(baro['date']).dt.to_period('M').dt.to_timestamp()
        
        baro_cols = ['month_key', 'barometer', 'current_conditions', 
                     'future_expectations', 'capital_investment_index']
        baro_cols = [c for c in baro_cols if c in baro.columns]
        
        df = df.merge(baro[baro_cols], left_on='sale_month', right_on='month_key', how='left')
        df = df.drop(columns=['month_key'], errors='ignore')
        print(f"Barometer matches: {df['barometer'].notna().sum():,}")
    
    # --- Merge Diesel ---
    if 'month_date' in diesel.columns:
        dies = diesel.copy()
        dies['month_key'] = pd.to_datetime(dies['month_date']).dt.to_period('M').dt.to_timestamp()
        
        df = df.merge(
            dies[['month_key', 'price_usd_per_gallon']].rename(
                columns={'price_usd_per_gallon': 'diesel_price'}
            ),
            left_on='sale_month', 
            right_on='month_key', 
            how='left'
        )
        df = df.drop(columns=['month_key'], errors='ignore')
        print(f"Diesel matches: {df['diesel_price'].notna().sum():,}")
    
    # --- Merge El Nino ---
    if 'year' in el_nino.columns and 'month' in el_nino.columns:
        el = el_nino.copy()
        el['month_key'] = pd.to_datetime(
            el['year'].astype(str) + '-' + el['month'].astype(str).str.zfill(2) + '-01'
        )
        
        el_subset = el[['month_key', 'value', 'phase']].rename(columns={
            'value': 'el_nino_value',
            'phase': 'el_nino_phase'
        })
        
        df = df.merge(el_subset, left_on='sale_month', right_on='month_key', how='left')
        df = df.drop(columns=['month_key'], errors='ignore')
        print(f"El Nino matches: {df['el_nino_phase'].notna().sum():,}")
    
    # --- Merge Futures (pivot to columns) ---
    if 'future_symbol' in futures.columns:
        fut = futures.copy()
        fut['month_key'] = pd.to_datetime(fut['date']).dt.to_period('M').dt.to_timestamp()
        
        # Monthly average per symbol
        fut_monthly = fut.groupby(['month_key', 'future_symbol'])['price'].mean().reset_index()
        fut_pivot = fut_monthly.pivot(index='month_key', columns='future_symbol', values='price').reset_index()
        
        # Clean column names
        fut_pivot.columns = ['month_key'] + [f'{c.lower().replace(" ", "_")}_price' for c in fut_pivot.columns[1:]]
        
        df = df.merge(fut_pivot, left_on='sale_month', right_on='month_key', how='left')
        df = df.drop(columns=['month_key'], errors='ignore')
        
        # Report matches for first commodity
        commodity_cols = [c for c in df.columns if c.endswith('_price') and c != 'diesel_price']
        if commodity_cols:
            print(f"Futures matches: {df[commodity_cols[0]].notna().sum():,}")
    
    return df


def create_training_dataset(
    auctions: pd.DataFrame,
    makes: pd.DataFrame,
    barometer: pd.DataFrame,
    diesel: pd.DataFrame,
    el_nino: pd.DataFrame,
    futures: pd.DataFrame,
    min_date: str = '2018-01-01',
    max_date: Optional[str] = None,
    min_price: int = 1000,
    max_price: int = 2_000_000
) -> pd.DataFrame:
    """
    Create complete training dataset.
    """
    from .cleaners import clean_auction_results, filter_training_data
    
    print("=" * 60)
    print("CREATING TRAINING DATASET")
    print("=" * 60)
    
    # Clean
    print("\n--- Cleaning ---")
    df = clean_auction_results(auctions)
    
    # Filter
    print("\n--- Filtering ---")
    df = filter_training_data(
        df,
        min_price=min_price,
        max_price=max_price,
        min_date=min_date,
        max_date=max_date,
        require_fields=['price', 'sold_date', 'make_key']
    )
    
    # Merge makes
    print("\n--- Merging Makes ---")
    df = merge_with_makes(df, makes)
    
    # Merge macro
    print("\n--- Merging Macro Data ---")
    df = merge_with_macro_data(df, barometer, diesel, el_nino, futures)
    
    print("\n" + "=" * 60)
    print(f"FINAL DATASET: {len(df):,} records")
    print(f"Date range: {df['sold_date'].min().date()} to {df['sold_date'].max().date()}")
    print("=" * 60)
    
    return df

