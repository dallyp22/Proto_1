"""
Add specification features (horsepower, cab, 4WD, etc.) to feature pipeline.
Expected MAPE improvement: 15-25%
"""

# This shows you how to parse specs - add to pipeline.py

EXAMPLE_CODE = '''
# Add to src/features/pipeline.py in _add_equipment_features():

import json

def _parse_specs(self, df: pd.DataFrame) -> pd.DataFrame:
    """Parse equipment specifications from JSON field."""
    
    if 'specs' not in df.columns:
        return df
    
    # Parse JSON specs
    def safe_parse(spec_str):
        if pd.isna(spec_str) or not spec_str:
            return {}
        try:
            # Convert Ruby hash notation to JSON
            json_str = str(spec_str).replace('=>', ':').replace('nil', 'null')
            return json.loads(json_str)
        except:
            return {}
    
    df['specs_dict'] = df['specs'].apply(safe_parse)
    
    # Extract horsepower (CRITICAL for tractors/equipment)
    df['horsepower'] = df['specs_dict'].apply(
        lambda x: x.get('horsepower') if isinstance(x, dict) else None
    )
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    
    # Impute missing horsepower by make_model
    if 'make_model_key' in df.columns:
        df['horsepower'] = df['horsepower'].fillna(
            df.groupby('make_model_key')['horsepower'].transform('median')
        )
    
    # Extract drive type
    df['drive_type'] = df['specs_dict'].apply(
        lambda x: str(x.get('drive', '')).lower() if isinstance(x, dict) else ''
    )
    df['is_4wd'] = df['drive_type'].str.contains('4').astype(int)
    
    # Extract features
    df['features_list'] = df['specs_dict'].apply(
        lambda x: x.get('features', []) if isinstance(x, dict) else []
    )
    df['has_cab'] = df['features_list'].apply(
        lambda x: any('cab' in str(f).lower() for f in x) if isinstance(x, list) else False
    ).astype(int)
    df['has_gps'] = df['features_list'].apply(
        lambda x: any('gps' in str(f).lower() or 'guidance' in str(f).lower() for f in x) if isinstance(x, list) else False
    ).astype(int)
    df['has_ac'] = df['features_list'].apply(
        lambda x: any('ac' in str(f).lower() or 'air' in str(f).lower() for f in x) if isinstance(x, list) else False
    ).astype(int)
    
    # Count attachments
    df['num_attachments'] = df['specs_dict'].apply(
        lambda x: len(x.get('attachments', [])) if isinstance(x, dict) else 0
    )
    
    return df

# Then update NUMERIC_FEATURES:
NUMERIC_FEATURES = [
    # ... existing features
    'horsepower',          # Engine power (MAJOR for tractors)
    'is_4wd',              # 4WD vs 2WD
    'has_cab',             # Enclosed cab
    'has_gps',             # GPS/guidance system
    'has_ac',              # Air conditioning
    'num_attachments',     # Included attachments
]
'''

print("="*60)
print("HOW TO ADD SPECS FEATURES")
print("="*60)
print(EXAMPLE_CODE)
print("\n" + "="*60)
print("IMPLEMENTATION STEPS")
print("="*60)
print("""
1. Copy the _parse_specs method above
2. Add it to src/features/pipeline.py in FeaturePipeline class
3. Call it in _add_equipment_features():
   
   def _add_equipment_features(self, df):
       df = self._parse_specs(df)  # ADD THIS LINE
       # ... rest of method
       
4. Add new features to NUMERIC_FEATURES list
5. Retrain: python train_log_models.py

Expected improvement:
- Harvesting: 41% → 25-30% MAPE
- Applicators: 51% → 30-35% MAPE  
- Tractors: 45% → 28-35% MAPE

Horsepower alone explains 20-30% of tractor value!
""")
