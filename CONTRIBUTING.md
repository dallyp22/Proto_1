# Contributing to Ag IQ Equipment Valuation

## Development Setup

```bash
# Clone and setup
git clone <repository-url>
cd ag_iq_ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Mac users: Install OpenMP
brew install libomp
```

## Project Workflow

### 1. Data Preparation
- Place raw data files in `data/raw/`
- Run `run_complete_pipeline.py` to process data

### 2. Model Training
- Train regular models: `python train_all_models.py`
- Train log-price models: `python train_log_models.py`

### 3. Testing
- Launch app: `streamlit run app.py`
- Test predictions against known values
- Validate confidence ranges

## Code Standards

### Python Style
- Follow PEP 8
- Use type hints where appropriate
- Document functions with docstrings
- Keep functions focused and modular

### File Organization
- `src/data/` - Data loading and processing
- `src/features/` - Feature engineering
- `src/models/` - Model training and inference
- `notebooks/` - Analysis and exploration
- `docs/` - Documentation

### Testing
- Test new features with sample data
- Validate model performance on test set
- Check Streamlit UI for errors

## Adding New Features

### To Add a New Feature:

1. **Update Feature Pipeline** (`src/features/pipeline.py`):
```python
# Add to NUMERIC_FEATURES or CATEGORICAL_FEATURES
NUMERIC_FEATURES = [
    # ... existing features
    'your_new_feature',
]

# Add feature engineering logic
def _add_your_feature(self, df):
    df['your_new_feature'] = ...  # Your logic
    return df
```

2. **Retrain Models**:
```bash
python train_log_models.py
```

3. **Test in Streamlit**:
```bash
streamlit run app.py
```

## Adding New Categories

### To Add a New Equipment Category:

1. **Update Config** (`src/models/multi_model_config.py`):
```python
CATEGORY_MODELS = {
    # ... existing categories
    'your_category': {
        'name': 'Your Category Name',
        'filters': ['keyword1', 'keyword2'],
        'min_price': 5000,
        'max_price': 500000,
        'color': '#hexcolor',
    },
}
```

2. **Train Models**:
```bash
python train_all_models.py
python train_log_models.py
```

## Improving Model Performance

### Priority Improvements:

1. **Parse Equipment Specifications**
   - Extract from `specs` JSON field
   - Add: GPS, cab, AC, horsepower

2. **Impute Missing Data**
   - Use make/model medians for missing hours
   - Estimate missing years from age patterns

3. **Hyperparameter Tuning**
   - Use Optuna for systematic search
   - Tune per category

4. **Add More Data Sources**
   - Weather patterns
   - Farm income data
   - Crop yields

## Pull Request Guidelines

1. **Branch naming**: `feature/description` or `fix/description`
2. **Commit messages**: Clear, descriptive
3. **Testing**: Verify models still train
4. **Documentation**: Update relevant docs
5. **Performance**: Report MAPE/R² changes

## Questions?

Contact the development team or open an issue.
