"""
Train Make + Category specific models for major brands.
Expected MAPE: 18-30% for John Deere, Ford, Case IH equipment.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loaders import load_all_data
from src.data.cleaners import clean_auction_results
from src.data.processors import merge_with_makes, merge_with_macro_data
from src.models.fmv_log import FMVLogModel
from src.models.make_category_config import MAKE_CATEGORY_MODELS, CATEGORY_FILTERS


def create_make_category_dataset(auctions, makes, barometer, diesel, el_nino, futures,
                                  make_key, category_name, min_price=5000, max_price=500000):
    """Create dataset for specific make-category combination."""
    
    print(f"\n{'='*60}")
    print(f"CREATING DATASET: {make_key.upper()} - {category_name.title()}")
    print(f"{'='*60}")
    
    # Clean
    df = clean_auction_results(auctions)
    
    # Filter to recent data
    df = df[df['sold_date'] >= pd.to_datetime('2018-01-01')]
    
    # Filter by make
    df = df[df['make_key'] == make_key]
    print(f"After make filter ({make_key}): {len(df):,}")
    
    # Filter by category
    cat_filters = CATEGORY_FILTERS.get(category_name, [category_name])
    mask = df['raw_category'].str.lower().str.contains('|'.join(cat_filters), na=False)
    df = df[mask]
    print(f"After category filter ({category_name}): {len(df):,}")
    
    # Price filter
    df = df[(df['price'] >= min_price) & (df['price'] <= max_price)]
    
    # Require critical fields
    make_field = 'make_model_key' if 'make_model_key' in df.columns else 'make_key'
    df = df[
        df['price'].notna() &
        df['sold_date'].notna() &
        df[make_field].notna() &
        df['year'].notna() &
        df['hours'].notna() &
        df['region'].notna()
    ]
    print(f"After requiring complete data: {len(df):,}")
    
    # Merge reference data
    df = merge_with_makes(df, makes)
    df = merge_with_macro_data(df, barometer, diesel, el_nino, futures)
    
    print(f"Final dataset: {len(df):,} records")
    if len(df) > 0:
        print(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
        print(f"Median price: ${df['price'].median():,.0f}")
    
    return df


def main():
    print("="*60)
    print("TRAIN MAKE + CATEGORY SPECIFIC MODELS")
    print("="*60)
    print("\nTraining dedicated models for major brands:")
    print("- John Deere (Tractors, Harvesting, Planting)")
    print("- Ford (Trucks, Tractors)")
    print("- Case IH (Tractors, Harvesting)")
    print("- Chevrolet (Trucks)")
    print("- New Holland (Tractors, Hay)")
    print("\nExpected MAPE: 18-30% for these models!\n")
    
    base_path = Path(__file__).parent
    
    # Load raw data
    print("📦 Loading raw data...")
    data = load_all_data(str(base_path / "data" / "raw"))
    
    results = {}
    models_trained = 0
    
    # Train each make-category model
    for model_key, config in MAKE_CATEGORY_MODELS.items():
        try:
            # Create dataset
            dataset = create_make_category_dataset(
                auctions=data['auctions'],
                makes=data['makes'],
                barometer=data['barometer'],
                diesel=data['diesel'],
                el_nino=data['el_nino'],
                futures=data['futures'],
                make_key=config['make'],
                category_name=config['category']
            )
            
            # Check if we have enough data
            if len(dataset) < config['min_records']:
                print(f"⚠️ Skipping - only {len(dataset):,} records (need {config['min_records']:,})")
                continue
            
            # Train model
            print(f"\n🤖 Training {model_key} model...")
            model = FMVLogModel()
            metrics = model.train(dataset)
            
            # Save
            model_path = base_path / "models" / f"fmv_{model_key}_log"
            model.save(str(model_path))
            
            results[model_key] = {
                'records': len(dataset),
                **metrics
            }
            models_trained += 1
            
            print(f"\n✓ {model_key} saved")
            print(f"  Records: {len(dataset):,}")
            print(f"  MAPE: {metrics['test_mape']:.2f}%")
            print(f"  R²: {metrics['test_r2']:.4f}")
            
        except Exception as e:
            print(f"\n❌ Error training {model_key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "="*60)
    print("MAKE-CATEGORY TRAINING COMPLETE")
    print("="*60)
    
    print(f"\nModels trained: {models_trained}")
    
    if results:
        print(f"\n📊 Performance Summary:")
        print(f"{'Make-Category':<35} {'Records':<10} {'MAPE':<10} {'R²':<10}")
        print("-"*70)
        
        for key, data in sorted(results.items(), key=lambda x: x[1]['test_mape']):
            make_cat_display = key.replace('_', ' ').replace('-', ' ').title()
            print(f"{make_cat_display:<35} {data['records']:<10,} {data['test_mape']:>7.2f}%  {data['test_r2']:>7.4f}")
        
        avg_mape = np.mean([d['test_mape'] for d in results.values()])
        avg_r2 = np.mean([d['test_r2'] for d in results.values()])
        
        print("-"*70)
        print(f"{'AVERAGE':<35} {'':<10} {avg_mape:>7.2f}%  {avg_r2:>7.4f}")
        
        # Show best performers
        best = sorted(results.items(), key=lambda x: x[1]['test_mape'])[:5]
        print(f"\n🏆 Top 5 Best Performing Make-Category Models:")
        for idx, (key, data) in enumerate(best, 1):
            print(f"  {idx}. {key}: {data['test_mape']:.1f}% MAPE, R² {data['test_r2']:.3f}")
        
        print(f"\n🎉 Next: streamlit run app.py")
        print("   App will automatically use make-category models when available!")
    else:
        print("\n❌ No models successfully trained")

if __name__ == "__main__":
    main()
