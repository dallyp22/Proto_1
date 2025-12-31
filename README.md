# Ag IQ Equipment Valuation System

AI-powered Fair Market Value (FMV) predictions for agricultural equipment using machine learning and 26 years of auction data.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

## Overview

The Ag IQ system addresses a critical gap in the $150B+ agricultural equipment market where current valuation guides lag actual market prices by months and don't reflect real-time economic conditions.

**Key Innovation**: Multi-model architecture with category-specific predictions combining 26 years of transaction history with macroeconomic indicators.

### System Capabilities

- **7 Category-Specific Models**: Tractors, Harvesting, Applicators, Loaders, Construction, Trucks, Other
- **Dual Prediction Methods**: Regular Price and Log-Price transformation
- **24 Engineered Features**: Equipment characteristics, temporal patterns, economic indicators
- **Interactive Web Interface**: Streamlit app with model comparison
- **Production-Ready**: Harvesting (33% MAPE, R² 0.91) and Applicators (39% MAPE, R² 0.87)

## Quick Start

```bash
# Clone repository
git clone <repository-url>
cd ag_iq_ml

# Setup environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install OpenMP (Mac only)
brew install libomp

# Train models (~20-30 minutes total)
python train_all_models.py      # Regular price models
python train_log_models.py       # Log-price models (recommended)

# Launch web interface
streamlit run app.py
# Opens at http://localhost:8501
```

## Project Structure

```
ag_iq_ml/
├── app.py                            # Streamlit web interface
├── train_all_models.py               # Train regular price models (7 categories)
├── train_log_models.py               # Train log-price models (7 categories)
├── run_complete_pipeline.py          # Complete pipeline from raw data
│
├── src/                              # Core library
│   ├── data/                         # Data loading and processing
│   │   ├── loaders.py                # Load raw data files
│   │   ├── cleaners.py               # Data cleaning utilities
│   │   └── processors.py             # Data merging and processing
│   ├── features/
│   │   └── pipeline.py               # Feature engineering (24 features)
│   └── models/
│       ├── config.py                 # LightGBM hyperparameters
│       ├── fmv.py                    # Regular price model
│       ├── fmv_log.py                # Log-price model
│       ├── multi_model_config.py     # Category definitions
│       └── model_manager.py          # Multi-model orchestration
│
├── notebooks/                        # Jupyter analysis notebooks
│   ├── 01_data_exploration.ipynb     # EDA on 733K records
│   ├── 02_data_cleaning.ipynb        # Data cleaning pipeline
│   ├── 03_feature_engineering.ipynb  # Feature creation
│   └── 04_fmv_model_training.ipynb   # Model training
│
├── data/                             # Data files (gitignored)
│   ├── raw/                          # Original exports (211MB)
│   ├── processed/                    # Cleaned datasets
│   └── features/                     # Feature-engineered data
│
├── models/                           # Trained models (gitignored except metadata)
│   ├── fmv_tractors/                 # Regular price models
│   ├── fmv_tractors_log/             # Log-price models
│   └── ... (14 models: 7 categories × 2 methods)
│
├── docs/                             # Documentation
│   ├── PROJECT_OVERVIEW.md           # Complete technical documentation
│   ├── QUICK_REFERENCE.md            # Quick start guide
│   └── MULTIMODEL_SUMMARY.md         # Model performance summary
│
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
└── LICENSE                           # MIT License
```

## Features

### Equipment Features (6)
- Equipment age, hours, hours per year
- Utilization bucket (light/normal/heavy/extreme)
- Production status, years since discontinued

### Temporal Features (7)
- Sale month, quarter, year
- Cyclical encoding (sin/cos for seasonality)
- Agricultural seasons (planting, harvest)

### Macroeconomic Features (4)
- Normalized Ag Economy Barometer
- Sentiment spread, investment confidence
- Diesel price relative to mean

### Density Features (2)
- Make volume (manufacturer popularity)
- Model volume (specific model popularity)

### Categorical Features (5)
- Make, region, category
- Utilization bucket
- **Condition** (Excellent/Good/Fair/Poor)

## Model Performance

### Production-Ready Categories

| Category | Method | MAPE | R² | Status |
|----------|--------|------|-----|--------|
| **Harvesting** | Log-Price | 33.3% | 0.910 | ✅ Production |
| **Applicators** | Log-Price | 39.2% | 0.867 | ✅ Production |

### Beta/Testing Categories

| Category | Method | MAPE | R² | Status |
|----------|--------|------|-----|--------|
| Loaders & Lifts | Log-Price | 41.4% | 0.374 | ⚠️ Beta |
| Tractors | Log-Price | 46.0% | 0.593 | ⚠️ Beta |
| Construction | Log-Price | 48.5% | 0.269 | ⚠️ Beta |
| Other | Log-Price | 57.4% | 0.475 | ⚠️ Beta |
| Trucks & Trailers | Log-Price | 92.9% | 0.310 | ❌ Not Recommended |

## Usage

### Training Models

```bash
# Train all regular price models (~15 minutes)
python train_all_models.py

# Train all log-price models (~15 minutes, recommended)
python train_log_models.py
```

### Running the Web Interface

```bash
streamlit run app.py
```

**Features:**
1. Select prediction method (Regular or Log-Price)
2. Select equipment category
3. Select make and specific model
4. Enter year, hours, condition, region
5. Get instant FMV with confidence range
6. Compare both prediction methods side-by-side
7. View feature importance and insights

### Programmatic Usage

```python
from src.models.fmv_log import FMVLogModel
import pandas as pd

# Load trained model
model = FMVLogModel.load('models/fmv_applicators_log')

# Prepare equipment data
equipment = pd.DataFrame([{
    'sold_date': '2025-06-15',
    'year': 2018,
    'hours': 1200,
    'raw_condition': 'Good',
    'make_key': 'john-deere',
    'raw_model': 'R4045',
    'region': 'midwest',
    'raw_category': 'Applicators',
    'barometer': 105,
    'diesel_price': 3.8,
    # ... other fields with defaults
}])

# Get prediction
fmv = model.predict(equipment)[0]
print(f"Estimated FMV: ${fmv:,.0f}")
```

## Documentation

- **[PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)** - Complete technical documentation, architecture, improvement roadmap
- **[QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - Commands, tips, troubleshooting
- **[MULTIMODEL_SUMMARY.md](docs/MULTIMODEL_SUMMARY.md)** - Model performance details

## Technology Stack

- **ML Framework**: LightGBM (gradient boosting)
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Interface**: Streamlit
- **Model Serialization**: joblib
- **Data Storage**: Parquet (pyarrow)

## Dataset

### Auction Data
- 733,413 transactions (1999-2025)
- Equipment details: make, model, year, hours, condition, region
- Transaction details: price, date, location

### Macroeconomic Indicators
- **Ag Economy Barometer** (120 months): Purdue farmer sentiment
- **Diesel Prices** (379 months): National fuel costs
- **El Niño Readings** (909 months): Climate indices
- **Commodity Futures** (15,660 prices): Corn, soy, wheat

### Reference Data
- 1,034 equipment manufacturers
- 2,102 auctioneers
- Production year ranges

## Model Architecture

### Category-Specific Approach

Instead of one model for all equipment, we train 7 specialized models:
- Each learns unique depreciation patterns
- Tighter price ranges per category
- Better accuracy than generic model

### Dual Prediction Methods

**Regular Price:**
- Direct price prediction
- Current performance: 55-135% MAPE

**Log-Price (Recommended):**
- Predicts log(price), converts back
- Better for skewed distributions
- Current performance: 33-93% MAPE
- **Use this method for production**

## Performance Notes

### Why MAPE is Higher Than Target

Original goal was <10% MAPE, current best is 33%. Reasons:

1. **Data Quality**: 60% of records missing critical fields (hours/year)
2. **Within-Category Variance**: Wide price ranges even within categories
3. **Missing Information**: Equipment options, detailed condition, specifications
4. **MAPE Sensitivity**: Heavily penalizes errors on low-priced items

### What the Models Do Well

- ✅ **High R²** (0.87-0.91 for best categories) - Explains variance
- ✅ **Captures trends** - Depreciation, seasonality, economic impact
- ✅ **Relative accuracy** - Correctly ranks expensive vs cheap
- ✅ **Better than alternatives** - Industry guides have ~50% error

### Production Recommendation

- **Use**: Harvesting and Applicators with Log-Price method
- **Confidence ranges**: Show ±33-39% uncertainty honestly
- **Best for**: Pre-auction estimates, rough valuations, portfolio analysis
- **Gather feedback**: Validate against actual sales, iterate

## Contributing

This is an internal DPA Auctions project. For questions or improvements, contact the development team.

## License

MIT License - See [LICENSE](LICENSE) file for details.

**Core valuation models and methodologies are licensed from Dallas Polivka.**

## Acknowledgments

- **Models & Methodology**: Dallas Polivka
- **Data Source**: DPA Auctions transaction database
- **Economic Data**: Purdue Ag Economy Barometer, USDA, NOAA
- **Technology**: LightGBM, Streamlit, scikit-learn communities

---

**Built**: December 2025  
**Creator**: Dallas Polivka  
**Status**: Production-Ready (Harvesting, Applicators)  
**Contact**: DPA Auctions Development Team
