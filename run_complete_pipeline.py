"""
Complete pipeline: Load raw data → Clean → Train → Save model
Run this if processed data doesn't exist yet.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loaders import load_all_data
from src.data.processors import create_training_dataset
from src.models.fmv import FMVModel

def main():
    print("=" * 60)
    print("AG IQ COMPLETE PIPELINE")
    print("=" * 60)
    
    base_path = Path(__file__).parent
    
    # Check if we can skip to training
    processed_path = base_path / "data" / "processed" / "training_data.parquet"
    
    if processed_path.exists():
        print("\n✓ Processed data already exists!")
        print(f"  Loading from: {processed_path}")
        training_df = pd.read_parquet(processed_path)
        print(f"  Loaded {len(training_df):,} records")
    else:
        print("\n📦 STEP 1: Load Raw Data")
        print("-" * 60)
        
        raw_data_path = base_path / "data" / "raw"
        if not raw_data_path.exists():
            print(f"❌ Error: Raw data not found at {raw_data_path}")
            print("Please ensure data files are in the data/raw/ directory")
            return
        
        print("Loading all data sources...")
        data = load_all_data(str(raw_data_path))
        
        print("\n🧹 STEP 2: Clean and Merge Data")
        print("-" * 60)
        
        print("Creating training dataset...")
        print("This will:")
        print("  - Clean auction data")
        print("  - Filter to valid records (2018-2025)")
        print("  - Merge with macro indicators")
        
        training_df = create_training_dataset(
            auctions=data['auctions'],
            makes=data['makes'],
            barometer=data['barometer'],
            diesel=data['diesel'],
            el_nino=data['el_nino'],
            futures=data['futures'],
            min_date='2018-01-01',
            max_date=None,
            min_price=1000,
            max_price=2_000_000
        )
        
        # Save processed data
        print(f"\nSaving processed data to: {processed_path}")
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        training_df.to_parquet(processed_path, index=False)
        print(f"✓ Saved {len(training_df):,} records")
    
    print("\n🤖 STEP 3: Train FMV Model")
    print("-" * 60)
    
    print("Initializing model...")
    model = FMVModel()
    
    print("\nTraining (this takes 2-5 minutes)...")
    print("Progress:")
    
    metrics = model.train(training_df)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nTest Set Performance:")
    print(f"  RMSE:  ${metrics['test_rmse']:,.0f}")
    print(f"  MAPE:  {metrics['test_mape']:.2f}%")
    print(f"  R²:    {metrics['test_r2']:.4f}")
    
    # Check goals
    print(f"\nGoal Achievement:")
    goals_met = 0
    
    if metrics['test_mape'] < 10:
        print(f"  ✓ MAPE < 10%: YES ({metrics['test_mape']:.2f}%)")
        goals_met += 1
    else:
        print(f"  ✗ MAPE < 10%: NO ({metrics['test_mape']:.2f}%)")
    
    if metrics['test_r2'] > 0.85:
        print(f"  ✓ R² > 0.85: YES ({metrics['test_r2']:.4f})")
        goals_met += 1
    else:
        print(f"  ✗ R² > 0.85: NO ({metrics['test_r2']:.4f})")
    
    if metrics['test_rmse'] < 15000:
        print(f"  ✓ RMSE < $15K: YES (${metrics['test_rmse']:,.0f})")
        goals_met += 1
    else:
        print(f"  ✗ RMSE < $15K: NO (${metrics['test_rmse']:,.0f})")
    
    print(f"\n🎯 Goals Met: {goals_met}/3")
    
    print("\n💾 STEP 4: Save Model")
    print("-" * 60)
    
    model_path = base_path / "models" / "fmv_model"
    model.save(str(model_path))
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)
    
    print(f"\nModel saved to: {model_path}/")
    print("  - model.lgb")
    print("  - pipeline.joblib")
    print("  - metadata.json")
    
    print("\n🎉 Next Step: Run the Streamlit App")
    print("   Command: streamlit run app.py")
    print()

if __name__ == "__main__":
    main()

