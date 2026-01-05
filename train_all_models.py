"""
Train category-specific FMV models.
This trains 7 models: 6 dedicated + 1 generic 'other' model.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loaders import load_all_data
from src.data.cleaners import clean_auction_results
from src.data.processors import merge_with_makes, merge_with_macro_data
from src.models.fmv import FMVModel
from src.models.multi_model_config import CATEGORY_MODELS, GENERIC_MODEL, get_category_key, get_model_display_name


def filter_by_category(df, category_filters, min_price, max_price):
    """Filter dataset to specific category."""
    # Filter by category
    mask = df['raw_category'].str.lower().str.contains('|'.join(category_filters), na=False)
    filtered = df[mask].copy()
    
    # Apply price filters
    filtered = filtered[
        (filtered['price'] >= min_price) & 
        (filtered['price'] <= max_price)
    ]
    
    return filtered


def create_category_dataset(auctions, makes, barometer, diesel, el_nino, futures, 
                            category_name, category_filters, min_price, max_price):
    """Create training dataset for a specific category."""
    
    print(f"\n{'='*60}")
    print(f"CREATING DATASET: {category_name}")
    print(f"{'='*60}")
    
    # Clean all data first
    df = clean_auction_results(auctions)
    
    # Filter to recent data
    df = df[df['sold_date'] >= pd.to_datetime('2018-01-01')]
    print(f"After date filter (2018+): {len(df):,}")
    
    # Filter by category
    df = filter_by_category(df, category_filters, min_price, max_price)
    print(f"After category filter: {len(df):,}")
    
    # Require critical fields (year, hours, region, make_model_key)
    initial = len(df)
    
    # Use make_model_key if available, otherwise fall back to make_key
    make_field = 'make_model_key' if 'make_model_key' in df.columns else 'make_key'
    
    df = df[
        df['price'].notna() & 
        df['sold_date'].notna() & 
        df[make_field].notna() &
        df['year'].notna() &
        df['hours'].notna() &
        df['region'].notna()
    ]
    print(f"After requiring complete data (including {make_field}): {len(df):,} (removed {initial - len(df):,})")
    
    # Merge makes
    df = merge_with_makes(df, makes)
    
    # Merge macro
    df = merge_with_macro_data(df, barometer, diesel, el_nino, futures)
    
    print(f"\nFinal dataset: {len(df):,} records")
    print(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    print(f"Median price: ${df['price'].median():,.0f}")
    
    return df


def train_category_model(df, category_key, category_name):
    """Train a model for a specific category."""
    
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL: {category_name}")
    print(f"{'='*60}")
    
    if len(df) < 1000:
        print(f"⚠️ Warning: Only {len(df):,} records. May not be enough for good performance.")
        print("   Skipping this category.")
        return None
    
    # Initialize and train model
    model = FMVModel()
    metrics = model.train(df)
    
    # Save model
    model_path = Path(__file__).parent / "models" / f"fmv_{category_key}"
    model.save(str(model_path))
    
    print(f"\n✓ {category_name} model saved")
    print(f"  MAPE: {metrics['test_mape']:.2f}%")
    print(f"  R²: {metrics['test_r2']:.4f}")
    
    return metrics


def main():
    print("="*60)
    print("TRAIN ALL CATEGORY-SPECIFIC MODELS")
    print("="*60)
    
    base_path = Path(__file__).parent
    
    # Load raw data
    print("\n📦 Loading raw data...")
    data = load_all_data(str(base_path / "data" / "raw"))
    
    results = {}
    
    # Train dedicated category models
    for category_key, config in CATEGORY_MODELS.items():
        try:
            # Create category dataset
            category_df = create_category_dataset(
                auctions=data['auctions'],
                makes=data['makes'],
                barometer=data['barometer'],
                diesel=data['diesel'],
                el_nino=data['el_nino'],
                futures=data['futures'],
                category_name=config['name'],
                category_filters=config['filters'],
                min_price=config['min_price'],
                max_price=config['max_price']
            )
            
            # Train model
            if len(category_df) >= 1000:
                metrics = train_category_model(category_df, category_key, config['name'])
                if metrics:
                    results[category_key] = metrics
            
        except Exception as e:
            print(f"\n❌ Error training {config['name']}: {e}")
            continue
    
    # Train generic "other" model
    print(f"\n{'='*60}")
    print("CREATING DATASET: Other Equipment (Generic)")
    print(f"{'='*60}")
    
    try:
        df = clean_auction_results(data['auctions'])
        df = df[df['sold_date'] >= pd.to_datetime('2018-01-01')]
        
        # Get records that DON'T match any dedicated category
        df['detected_category'] = df['raw_category'].apply(get_category_key)
        other_df = df[df['detected_category'] == 'other'].copy()
        
        # Price filter for generic model
        other_df = other_df[
            (other_df['price'] >= GENERIC_MODEL['min_price']) &
            (other_df['price'] <= GENERIC_MODEL['max_price'])
        ]
        
        # Require complete data
        other_df = other_df[
            other_df['price'].notna() &
            other_df['sold_date'].notna() &
            other_df['make_key'].notna() &
            other_df['year'].notna() &
            other_df['hours'].notna() &
            other_df['region'].notna()
        ]
        
        print(f"Other equipment records: {len(other_df):,}")
        
        # Merge
        other_df = merge_with_makes(other_df, data['makes'])
        other_df = merge_with_macro_data(
            other_df, data['barometer'], data['diesel'], 
            data['el_nino'], data['futures']
        )
        
        # Train
        if len(other_df) >= 1000:
            metrics = train_category_model(other_df, 'other', 'Other Equipment')
            if metrics:
                results['other'] = metrics
    
    except Exception as e:
        print(f"\n❌ Error training Other model: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    
    print(f"\nModels trained: {len(results)}")
    
    if results:
        print("\nPerformance by Category:")
        print(f"{'Category':<25} {'Records':<10} {'MAPE':<10} {'R²':<10}")
        print("-"*60)
        
        for cat_key, metrics in results.items():
            name = get_model_display_name(cat_key)
            # We'd need to track record counts, for now show metrics
            print(f"{name:<25} {'N/A':<10} {metrics['test_mape']:>7.2f}%  {metrics['test_r2']:>7.4f}")
        
        avg_mape = sum(m['test_mape'] for m in results.values()) / len(results)
        avg_r2 = sum(m['test_r2'] for m in results.values()) / len(results)
        
        print("-"*60)
        print(f"{'AVERAGE':<25} {'':<10} {avg_mape:>7.2f}%  {avg_r2:>7.4f}")
        
        print(f"\n🎉 Next step: streamlit run app.py")
    else:
        print("\n❌ No models were successfully trained.")
        print("Check the error messages above.")

if __name__ == "__main__":
    main()

