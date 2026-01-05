# How to Improve Model Accuracy (Reduce MAPE)

**Current Performance:**
- Best: Harvesting 41% MAPE, R² 0.91
- Good: Applicators 51% MAPE, R² 0.79
- Target: <20% MAPE for production

---

## 🎯 Prioritized Improvement Strategies

### ✅ DONE - Data Imputation
**Status**: Implemented in `train_with_imputation.py`  
**Result**: 2-16x more training data  
**Impact**: Trucks improved 82% → 66% MAPE  

---

## Priority 1: Parse Equipment Specifications ⭐⭐⭐⭐⭐

**Impact**: **15-25% MAPE reduction**  
**Effort**: 3-4 hours  
**Data Available**: 99.6% of records have specs  

### What's in Specs

```json
{
  "horsepower": 455,           // Engine power - HUGE price driver
  "drive": "4 Wheel Drive",    // 4WD vs 2WD
  "features": ["cab", "AC"],   // Enclosed cab, AC
  "engine": "Paccar MX13",     // Engine type
  "attachments": [...]         // Included equipment
}
```

### Implementation

**Add to `src/features/pipeline.py`:**

```python
def _parse_specs(self, df):
    """Extract features from specs JSON."""
    import json
    
    def safe_parse(s):
        if pd.isna(s): return {}
        try:
            return json.loads(str(s).replace('=>', ':').replace('nil', 'null'))
        except: return {}
    
    df['specs_dict'] = df['specs'].apply(safe_parse)
    
    # Horsepower - CRITICAL!
    df['horsepower'] = df['specs_dict'].apply(
        lambda x: x.get('horsepower') if isinstance(x, dict) else None
    )
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    df['horsepower'] = df['horsepower'].fillna(
        df.groupby('make_model_key')['horsepower'].transform('median')
    ).fillna(150)  # Overall default
    
    # Drive type
    df['is_4wd'] = df['specs_dict'].apply(
        lambda x: 1 if '4' in str(x.get('drive', '')) else 0
    )
    
    # Features
    df['has_cab'] = df['specs_dict'].apply(
        lambda x: int(any('cab' in str(f).lower() for f in x.get('features', [])))
    )
    df['has_gps'] = df['specs_dict'].apply(
        lambda x: int(any('gps' in str(f).lower() for f in x.get('features', [])))
    )
    df['has_ac'] = df['specs_dict'].apply(
        lambda x: int(any('ac' in str(f).lower() or 'air' in str(f).lower() for f in x.get('features', [])))
    )
    
    # Attachments count
    df['num_attachments'] = df['specs_dict'].apply(
        lambda x: len(x.get('attachments', [])) if isinstance(x, dict) else 0
    )
    
    return df

# Add to NUMERIC_FEATURES:
NUMERIC_FEATURES = [
    ...existing features...,
    'horsepower',      # Power rating (100-500 typical)
    'is_4wd',          # 4WD = premium
    'has_cab',         # Enclosed cab = 20-30% value
    'has_gps',         # GPS guidance = 15-25% value
    'has_ac',          # Air conditioning = comfort
    'num_attachments', # Included equipment
]

# Call in _add_equipment_features():
def _add_equipment_features(self, df):
    df = self._parse_specs(df)  # ADD THIS
    # ... rest of method
```

**Expected Results:**
- **Harvesting**: 41% → **25-30%** MAPE
- **Applicators**: 51% → **30-38%** MAPE
- **Tractors**: 45% → **28-35%** MAPE

**Why Huge Impact:**
- 200HP tractor worth 2x more than 100HP
- Cab adds $15-30K value
- GPS adds $10-20K value
- Model currently has NO IDEA about these!

---

## Priority 2: Numeric Condition Score ⭐⭐⭐⭐

**Impact**: **5-10% MAPE reduction**  
**Effort**: 30 minutes  

**Current**: Categorical (Excellent/Good/Fair/Poor)  
**Better**: Numeric scale (5/4/3/2)  

```python
# In _add_equipment_features():

condition_map = {
    'excellent': 5.0,
    'good': 4.0,
    'fair': 3.0,
    'poor': 2.0,
}

df['condition_score'] = df['condition'].map(condition_map).fillna(4.0)

# Add to NUMERIC_FEATURES
'condition_score',  # 1-5 rating
```

**Why Better:**
- Captures magnitude (Excellent is 25% better than Good)
- Allows interpolation
- Better for gradient boosting

---

## Priority 3: Segment Tractors by HP Class ⭐⭐⭐

**Impact**: **8-15% MAPE reduction** for tractors  
**Effort**: 2 hours  

**Current**: All tractors in one model  
**Problem**: $5K compact vs $500K row-crop  

**Solution**: 3 tractor models

```python
TRACTOR_CATEGORIES = {
    'tractors_compact': {
        'name': 'Compact Tractors',
        'hp_range': (0, 60),
        'price_range': (5000, 50000),
    },
    'tractors_utility': {
        'name': 'Utility Tractors', 
        'hp_range': (60, 150),
        'price_range': (15000, 150000),
    },
    'tractors_rowcrop': {
        'name': 'Row-Crop Tractors',
        'hp_range': (150, 600),
        'price_range': (50000, 500000),
    },
}
```

**Expected**: Each segment achieves 25-35% MAPE

---

## Priority 4: Add State-Level Features ⭐⭐⭐

**Impact**: **5-8% MAPE reduction**  
**Effort**: 30 minutes  

```python
# Use 'state' instead of 'region'
CATEGORICAL_FEATURES = [
    'make_model_key',
    'state',  # 50 states vs 8 regions
    ...
]
```

**Why**: Iowa ≠ Kansas even though both "Midwest"

---

## Priority 5: Add Price Trends ⭐⭐⭐

**Impact**: **5-10% MAPE reduction**  
**Effort**: 3 hours  

```python
# Add rolling price averages for make_model

# 6-month rolling average for this make_model
df['make_model_price_6mo'] = df.groupby('make_model_key')['price'].transform(
    lambda x: x.rolling(window=6, min_periods=1).mean()
)

# Price momentum (trending up or down)
df['price_trend'] = (
    df['make_model_price_6mo'] - 
    df.groupby('make_model_key')['make_model_price_6mo'].shift(6)
)
```

---

## 📊 Combined Impact Projection

**Starting Point** (With Imputation):
- Harvesting: 41.1% MAPE
- Applicators: 50.7% MAPE

**After Specs Parsing:**
- Harvesting: **26-32%** MAPE
- Applicators: **32-40%** MAPE

**After Condition Score + State:**
- Harvesting: **22-28%** MAPE ✅
- Applicators: **28-35%** MAPE ✅

**After HP Segmentation (Tractors):**
- Compact Tractors: **25-30%** MAPE ✅
- Utility Tractors: **22-28%** MAPE ✅
- Row-Crop Tractors: **20-25%** MAPE ✅

---

## 🚀 Implementation Roadmap

### Week 1: Parse Specs (Biggest Win!)
```bash
# 1. Update pipeline.py with _parse_specs method
# 2. Add 6 new features from specs
# 3. Retrain: python train_with_imputation.py
```

**Expected**: Drop to 25-40% MAPE across categories

### Week 2: Refine Features
```bash
# 1. Add condition_score (numeric)
# 2. Change region → state
# 3. Retrain
```

**Expected**: Drop to 20-35% MAPE

### Week 3: Segment Tractors
```bash
# 1. Create 3 tractor categories by HP
# 2. Train separate models
# 3. Update Streamlit to route correctly
```

**Expected**: Tractors drop to 20-28% MAPE

---

## 💡 Why Current MAPE is High

**Missing Information:**
- ❌ No horsepower (200HP tractor worth 2x 100HP)
- ❌ No cab info ($20K value difference)
- ❌ No GPS ($15K value difference)
- ❌ Coarse condition (Good could be 70% or 90%)
- ❌ Wide category (utility vs row-crop mixed)

**After fixes:**
- ✅ Have horsepower
- ✅ Know cab/GPS/AC
- ✅ Numeric condition
- ✅ Segmented categories

**Result**: Model can accurately differentiate equipment!

---

## 🎯 Fastest Path to <25% MAPE

**Single action with biggest impact:**

### Parse Horsepower (1 hour work)

```python
# Just add this ONE feature - it's that important!

# In pipeline.py:
import json

df['specs_dict'] = df['specs'].apply(
    lambda x: json.loads(str(x).replace('=>', ':').replace('nil', 'null'))
    if pd.notna(x) else {}
)

df['horsepower'] = df['specs_dict'].apply(lambda x: x.get('horsepower'))
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df['horsepower'] = df['horsepower'].fillna(
    df.groupby('make_model_key')['horsepower'].transform('median')
).fillna(150)

# Add to NUMERIC_FEATURES
'horsepower',
```

**Retrain:**
```bash
python train_with_imputation.py
```

**Expected**:
- **Tractors**: 45% → **30-35%** MAPE
- **Harvesting**: 41% → **28-32%** MAPE

**Just horsepower alone could reduce MAPE by 15-20 percentage points!**

---

## 📈 Long-Term Optimizations

### Advanced Techniques (After basics above)

1. **Ensemble Models** - Combine multiple models
2. **XGBoost** - Try alongside LightGBM
3. **Neural Networks** - For very large datasets
4. **Image Analysis** - CNN on equipment photos
5. **Market Timing** - Seasonal adjustment factors
6. **Hyperparameter Tuning** - Optuna optimization

---

## 🎯 Recommendation

**Do These 3 Things (Total: 4-5 hours):**

1. ✅ **Imputation** (already done!) - 2-16x more data
2. 🔜 **Parse horsepower** (1 hour) - 15-20% MAPE drop
3. 🔜 **Add condition_score + specs features** (2 hours) - 10-15% more

**Result**: **Harvesting at 20-25% MAPE, Applicators at 25-30%** 

**That's production quality for asset valuation!** 🎉

---

**Want me to implement the specs parsing for you?** It's the single biggest opportunity for improvement!
