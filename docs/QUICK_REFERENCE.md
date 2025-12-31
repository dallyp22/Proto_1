# Ag IQ Quick Reference Guide

## 🚀 Quick Start

### First Time Setup
```bash
cd /Users/dallas/AssetManager/ag_iq_ml
source venv/bin/activate
```

### Train Models
```bash
# Regular models (required, ~15 min)
python train_all_models.py

# Log-price models (recommended, ~15 min)
python train_log_models.py
```

### Run App
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web interface |
| `train_all_models.py` | Train 7 regular models |
| `train_log_models.py` | Train 7 log-price models |
| `run_complete_pipeline.py` | Full pipeline from scratch |
| `src/models/fmv.py` | Regular price model |
| `src/models/fmv_log.py` | Log-price model |
| `src/features/pipeline.py` | Feature engineering |
| `src/data/loaders.py` | Data loading |

---

## 🎯 Model Categories

1. **Tractors** (best data: 19K records)
2. **Trucks & Trailers** (limited: 2K records)
3. **Harvesting** (good R²: 0.89)
4. **Loaders & Lifts** (moderate: 8K records)
5. **Construction** (moderate: 5K records)
6. **Applicators** (good R²: 0.85)
7. **Other** (generic: 5K records)

---

## 🔬 Prediction Methods

### Regular Price
- **Current MAPE**: 55-135%
- **Best for**: Testing, baseline
- **Command**: Select "Regular Price" in app

### Log-Price
- **Expected MAPE**: 10-25%
- **Best for**: Production use
- **Command**: Select "Log-Price" in app

---

## 📊 Current Performance

| Category | Regular MAPE | R² | Records |
|----------|-------------|-----|---------|
| Harvesting | 55.7% | 0.892 ⭐ | 3,731 |
| Applicators | 61.6% | 0.848 ⭐ | 1,802 |
| Tractors | 85.4% | 0.619 | 13,619 |
| Other | 60-135% | 0.23-0.51 | Varies |

**After log-price training, expect MAPE to drop 50-80%!**

---

## 💡 Usage Tips

### Getting Best Results

1. ✅ **Use Log-Price models** - Much better accuracy
2. ✅ **Select specific model** - More precise than just make
3. ✅ **Enter accurate hours** - Critical for valuation
4. ✅ **Use Harvesting or Applicators** - Best performing categories
5. ✅ **Check comparison** - If both methods available

### Interpreting Results

**Confidence Range:**
- Based on model's test MAPE
- ±61% means: actual could be 39%-161% of prediction
- Narrower = more confident

**Feature Importance:**
- Shows what drives the prediction
- Top features: year, hours, make usually
- Category-specific patterns

**Utilization:**
- Light (<300 hrs/yr): Premium value
- Normal (300-600): Standard value
- Heavy (600-1000): Reduced value
- Extreme (>1000): Significant reduction

---

## 🔧 Common Commands

### Training
```bash
# All regular models
python train_all_models.py

# All log-price models
python train_log_models.py

# Single category (edit script)
python -c "from src.models.fmv_log import FMVLogModel; ..."
```

### Data Exploration
```bash
# Launch Jupyter
jupyter notebook

# Open notebooks in order:
# 01_data_exploration.ipynb
# 02_data_cleaning.ipynb
# 03_feature_engineering.ipynb
# 04_fmv_model_training.ipynb
```

### Diagnostics
```bash
# Check data quality
python diagnose_data.py

# Check available models
ls -la models/
```

---

## ⚠️ Troubleshooting

### "Module not found"
```bash
# Make sure venv is activated
source venv/bin/activate
# You should see (venv) in prompt
```

### "Model not found"
```bash
# Train the models first
python train_all_models.py
python train_log_models.py
```

### "File not found: training_data.parquet"
```bash
# Run complete pipeline
python run_complete_pipeline.py
```

### "libomp.dylib not found" (Mac)
```bash
brew install libomp
```

### High MAPE results
```bash
# Try log-price models instead
python train_log_models.py
# Then select "Log-Price" in app
```

---

## 📈 Improvement Priorities

### Priority 1 (Do Now)
- [ ] Train log-price models
- [ ] Test both methods
- [ ] Validate against known values

### Priority 2 (Next Week)
- [ ] Parse condition field
- [ ] Impute missing hours/year
- [ ] Hyperparameter tuning

### Priority 3 (Next Month)
- [ ] Deploy to Streamlit Cloud
- [ ] Add batch upload
- [ ] Create PDF reports

---

## 📞 Support

### Documentation
- `PROJECT_OVERVIEW.md` - Complete technical docs
- `README.md` - Project setup
- `MULTIMODEL_SUMMARY.md` - Model performance
- `PHASE[1-6]_COMPLETE.md` - Implementation phases

### Key Metrics to Track
- **MAPE**: Mean Absolute Percentage Error (goal: <15%)
- **R²**: Variance explained (goal: >0.85)
- **RMSE**: Root Mean Squared Error (goal: <$15K)

### Model Locations
- Regular: `models/fmv_{category}/`
- Log-price: `models/fmv_{category}_log/`
- Each contains: `model.lgb`, `pipeline.joblib`, `metadata.json`

---

## 🎯 Success Criteria

**Minimum Viable Product** (Current):
- [x] 7 category models trained
- [x] Streamlit interface working
- [x] Model + Make + Category selection
- [x] Real-time predictions

**Production Ready** (Next):
- [ ] Log models achieving <20% MAPE
- [ ] Validated against 100+ known sales
- [ ] User acceptance testing complete
- [ ] Deployed to accessible URL

**Industry Leading** (Future):
- [ ] <10% MAPE across all categories
- [ ] Mobile app available
- [ ] API for integrations
- [ ] Automated retraining pipeline

---

**Current Status**: ✅ MVP Complete, Ready for Log-Price Training

**Next Action**: `python train_log_models.py`

