# Data Schema Documentation

## Overview

This document describes the data schemas for all files used in the Ag IQ system.

## Auction Results (Primary Transaction Data)

**Files**: `auction_results_part[1-3]_*.xlsx` (3 files, 733,413 total records)

### Target Variable
- `price` - Sale price in USD (what we predict)

### Equipment Identity
- `make_id` - UUID foreign key to makes table
- `make_key` - Normalized make string (used for joining)
- `model_id` - UUID (mostly empty)
- `raw_model` - Model name/number (100% coverage)
- `year` - Model year (58% coverage)
- `raw_make` - Original make string
- `raw_category` - Equipment category (100% coverage)

### Condition/Usage
- `hours` - Equipment hours (40% coverage)
- `miles` - Miles if applicable
- `raw_condition` - Condition: Excellent/Good/Fair/Poor (100% coverage)
- `raw_hours` - Original hours string

### Transaction Details
- `sold_date` - Date of sale (100% coverage)
- `raw_sold_date` - Original date string
- `raw_price` - Original price string
- `auctioneer_id` - UUID foreign key
- `auctioneer_key` - Normalized auctioneer string
- `raw_auctioneer` - Original auctioneer string

### Geography
- `region` - Region enum: midwest, great_plains, southeast, west, etc.
- `state` - State code (2-letter)
- `city` - City name
- `zip` - ZIP code
- `latitude` - Coordinate
- `longitude` - Coordinate

### Other Fields
- `serial_number` - Equipment serial
- `vin` - VIN if applicable
- `specs` - JSON specifications (not currently parsed)
- `raw_specs` - Original specs string

## Macroeconomic Indicators

### Ag Economy Barometer

**File**: `ag_economy_barometers.csv` (120 records, tab-delimited)

- `date` - Timestamp
- `month` - Month string
- `year` - Year integer
- `barometer` - Overall index (typically 80-180)
- `current_conditions` - Present assessment
- `future_expectations` - Forward outlook
- `capital_investment_index` - Equipment buying intent

### Diesel Prices

**File**: `diesel_prices.csv` (379 records, tab-delimited)

- `month_date` - Month timestamp
- `price_usd_per_gallon` - Diesel price

### El Niño Readings

**File**: `el_nino_readings.csv` (909 records, tab-delimited)

- `date` - Timestamp
- `year` - Year integer
- `month` - Month integer
- `value` - ENSO index value
- `phase` - Phase indicator

### Commodity Futures

**File**: `future_prices.csv` (15,660 records, tab-delimited)

- `date` - Date
- `future_symbol` - Contract symbol (ZCZ5, ZSF6, etc.)
- `price` - Futures price
- `vol` - Volume

**Symbols**: Corn, Soybean, Wheat, Crude Oil, Live Cattle, Feeder Cattle, Lean Hogs

## Reference Data

### Makes

**File**: `makes_export_*.xlsx` (1,034 records)

- `name` - Display name
- `make_key` - Normalized key (JOIN KEY with auction_results)
- `still_in_business` - Boolean
- `headquarters_country_iso_2` - Country code
- `production_year_start` - First production year
- `production_year_end` - Last production year (null if current)
- Additional enrichment fields (CoreSignal, social media, etc.)

### Auctioneers

**File**: `auctioneers_export_*.xlsx` (2,102 records)

- Various auctioneer details
- Not currently used in modeling

## Processed Datasets

### Training Data

**File**: `data/processed/training_data.parquet`

**Created by**: `run_complete_pipeline.py` or `train_all_models.py`

**Contents**:
- Cleaned auction results (2018-2025)
- Merged with makes reference
- Merged with macro indicators (by month)
- Filtered to valid records with required fields

**Record count**: ~204,532 (before category filtering)

**Key filters applied**:
- Date range: 2018-01-01 to present
- Price range: $1,000 to $2,000,000
- Required fields: price, sold_date, make_key

### Feature-Engineered Data

**File**: `data/features/training_features.parquet`

**Created by**: Feature pipeline during model training

**Contents**:
- All fields from training_data.parquet
- Plus 24 engineered features
- Ready for model input

## Data Quality Notes

### Coverage Statistics

| Field | Coverage | Notes |
|-------|----------|-------|
| price | 100% | Required for training |
| sold_date | 100% | Required for training |
| make_key | 100% | Required for training |
| raw_condition | 100% | Excellent/Good/Fair/Poor |
| raw_model | 100% | Model names/numbers |
| region | 100% | After filtering |
| **year** | **58%** | Major limitation |
| **hours** | **40%** | Major limitation |
| barometer | 98% | After merging |
| diesel_price | 99% | After merging |

### Data Quality Issues

1. **Missing Hours (60%)**: Limits utilization calculations
2. **Missing Year (42%)**: Limits age calculations
3. **Wide Price Ranges**: $1K parts to $500K equipment
4. **Heterogeneous Categories**: "Tractors" includes utility and row-crop

### Filtering for High-Quality Models

Category-specific training applies:
- Price filters per category ($5K-$500K typical)
- **Requires**: year, hours, region (not null)
- Results in ~20K-40K records per major category

## File Formats

### Excel (.xlsx)
- Auction results (3 parts)
- Makes export
- Auctioneers export

### CSV (Tab-Delimited)
- Ag economy barometers
- Diesel prices
- El Niño readings
- Commodity futures

**Note**: CSVs use `\t` (tab) delimiter, not comma!

### Parquet
- Processed datasets
- Feature-engineered data
- Efficient columnar storage
- Fast loading with pandas

## Data Pipeline Flow

```
Raw Data (733K records)
    ↓
Clean & Validate
  - Parse dates, prices
  - Standardize text fields
  - Remove invalid values
    ↓
Filter (204K records)
  - Date range: 2018-2025
  - Price range: $1K-$2M
  - Has make_key
    ↓
Merge Reference Data
  - Join makes (production years)
  - Join macro indicators by month
    ↓
Category-Specific Filtering (20K-40K per category)
  - Category-specific price ranges
  - Require: year, hours, region
  - Major equipment only
    ↓
Feature Engineering (24 features)
  - Equipment: age, utilization, condition
  - Temporal: seasonality, cycles
  - Macro: normalized indicators
  - Density: make/model volume
    ↓
Model Training
  - Time-based splits (70/15/15)
  - LightGBM gradient boosting
  - Early stopping
    ↓
Trained Models (14 total)
  - 7 regular + 7 log-price
  - Saved with metadata
```

## Adding New Data Sources

To add new data sources:

1. **Create loader** in `src/data/loaders.py`:
```python
def load_your_data(data_dir: str = "data/raw") -> pd.DataFrame:
    path = Path(data_dir) / "your_file.csv"
    df = pd.read_csv(path)
    # Parse dates, clean data
    return df
```

2. **Add to merger** in `src/data/processors.py`:
```python
def merge_with_your_data(auctions, your_data):
    # Merge logic (typically by date/month)
    return merged_df
```

3. **Update pipeline** in `run_complete_pipeline.py`:
```python
data = load_all_data(...)
training_df = create_training_dataset(
    ...,
    your_data=data['your_data']
)
```

## Data Privacy & Security

- **No PII**: Data contains equipment details only, no personal information
- **Anonymized**: Auctioneer IDs are UUIDs
- **Aggregated**: Individual transactions combined for modeling
- **Internal Use**: Not for public distribution

## Questions?

For data schema questions or issues, contact the development team.
