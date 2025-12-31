"""
Train log-price models for all categories.
Run this after train_all_models.py to create log-price versions.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loaders import load_all_data
from src.data.cleaners import clean_auction_results
from src.data.processors import merge_with_makes, merge_with_macro_data
from src.models.fmv_log import FMVLogModel
from src.models.multi_model_config import CATEGORY_MODELS, GENERIC_MODEL, get_category_key, get_model_display_name


def filter_by_category(df, category_filters, min_price, max_price):
    """Filter dataset to specific category."""
    mask = df['raw_category'].str.lower().str.contains('|'.join(category_filters), na=False)
    filtered = df[mask].copy()
    
    filtered = filtered[
        (filtered['price'] >= min_price) & 
        (filtered['price'] <= max_price)
    ]
    
    return filtered


def create_category_dataset(auctions, makes, barometer, diesel, el_nino, futures, 
                            category_filters, min_price, max_price):
    """Create training dataset for a specific category."""
    
    df = clean_auction_results(auctions)
    df = df[df['sold_date'] >= pd.to_datetime('2018-01-01')]
    df = filter_by_category(df, category_filters, min_price, max_price)
    
    df = df[
        df['price'].notna() & 
        df['sold_date'].notna() & 
        df['make_key'].notna() &
        df['year'].notna() &
        df['hours'].notna() &
        df['region'].notna()
    ]
    
    df = merge_with_makes(df, makes)
    df = merge_with_macro_data(df, barometer, diesel, el_nino, futures)
    
    return df


def main():
    print("="*60)
    print("TRAIN LOG-PRICE MODELS")
    print("="*60)
    print("\nThis creates log-transformed versions of all category models.")
    print("Log transformation often improves MAPE on skewed price data.\n")
    
    base_path = Path(__file__).parent
    
    # Load raw data
    print("📦 Loading raw data...")
    data = load_all_data(str(base_path / "data" / "raw"))
    
    results = {}
    
    # Train dedicated category models
    for category_key, config in CATEGORY_MODELS.items():
        print(f"\n{'='*60}")
        print(f"TRAINING LOG-PRICE MODEL: {config['name']}")
        print(f"{'='*60}")
        
        try:
            # Create category dataset
            category_df = create_category_dataset(
                auctions=data['auctions'],
                makes=data['makes'],
                barometer=data['barometer'],
                diesel=data['diesel'],
                el_nino=data['el_nino'],
                futures=data['futures'],
                category_filters=config['filters'],
                min_price=config['min_price'],
                max_price=config['max_price']
            )
            
            print(f"Dataset: {len(category_df):,} records")
            
            if len(category_df) < 1000:
                print(f"⚠️ Skipping {config['name']} - insufficient data")
                continue
            
            # Train LOG-PRICE model
            model = FMVLogModel()
            metrics = model.train(category_df)
            
            # Save with _log suffix
            model_path = base_path / "models" / f"fmv_{category_key}_log"
            model.save(str(model_path))
            
            results[category_key] = metrics
            
            print(f"\n✓ {config['name']} LOG model saved")
            print(f"  MAPE: {metrics['test_mape']:.2f}%")
            print(f"  R²: {metrics['test_r2']:.4f}")
            
        except Exception as e:
            print(f"\n❌ Error training {config['name']}: {e}")
            continue
    
    # Train generic "other" model
    print(f"\n{'='*60}")
    print("TRAINING LOG-PRICE MODEL: Other Equipment")
    print(f"{'='*60}")
    
    try:
        df = clean_auction_results(data['auctions'])
        df = df[df['sold_date'] >= pd.to_datetime('2018-01-01')]
        
        df['detected_category'] = df['raw_category'].apply(get_category_key)
        other_df = df[df['detected_category'] == 'other'].copy()
        
        other_df = other_df[
            (other_df['price'] >= GENERIC_MODEL['min_price']) &
            (other_df['price'] <= GENERIC_MODEL['max_price']) &
            other_df['price'].notna() &
            other_df['sold_date'].notna() &
            other_df['make_key'].notna() &
            other_df['year'].notna() &
            other_df['hours'].notna() &
            other_df['region'].notna()
        ]
        
        other_df = merge_with_makes(other_df, data['makes'])
        other_df = merge_with_macro_data(
            other_df, data['barometer'], data['diesel'], 
            data['el_nino'], data['futures']
        )
        
        print(f"Dataset: {len(other_df):,} records")
        
        if len(other_df) >= 1000:
            model = FMVLogModel()
            metrics = model.train(other_df)
            
            model_path = base_path / "models" / "fmv_other_log"
            model.save(str(model_path))
            
            results['other'] = metrics
            
            print(f"\n✓ Other Equipment LOG model saved")
            print(f"  MAPE: {metrics['test_mape']:.2f}%")
            print(f"  R²: {metrics['test_r2']:.4f}")
    
    except Exception as e:
        print(f"\n❌ Error training Other model: {e}")
    
    # Summary with comparison
    print("\n" + "="*60)
    print("LOG-PRICE TRAINING COMPLETE")
    print("="*60)
    
    print(f"\nLog-price models trained: {len(results)}")
    
    if results:
        print("\n📊 Performance Summary (Log-Price Models):")
        print(f"{'Category':<25} {'MAPE':<12} {'R²':<10}")
        print("-"*50)
        
        for cat_key, metrics in results.items():
            name = get_model_display_name(cat_key)
            print(f"{name:<25} {metrics['test_mape']:>8.2f}%  {metrics['test_r2']:>8.4f}")
        
        avg_mape = sum(m['test_mape'] for m in results.values()) / len(results)
        avg_r2 = sum(m['test_r2'] for m in results.values()) / len(results)
        
        print("-"*50)
        print(f"{'AVERAGE':<25} {avg_mape:>8.2f}%  {avg_r2:>8.4f}")
        
        print(f"\n🎉 Next step:")
        print(f"   Run: streamlit run app.py")
        print(f"   Toggle between Regular and Log-Price models in the app!")

if __name__ == "__main__":
    main()

