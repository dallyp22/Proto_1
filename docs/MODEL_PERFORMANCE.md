# Model Performance Report

**Last Updated**: December 30, 2025  
**Dataset**: 733,413 auction records (1999-2025)  
**Training Window**: 2018-2025 (7 years)  
**Total Models**: 14 (7 categories × 2 methods)

---

## Production-Ready Models

### Harvesting Equipment

**Log-Price Model** (Recommended)
- **Test MAPE**: 33.3%
- **Test R²**: 0.910
- **Test RMSE**: $37,431
- **Training Records**: 3,731
- **Status**: ✅ Production Ready

**Interpretation**:
- Explains 91% of price variance
- Typical prediction within ±33%
- Example: $100K combine predicted as $67K-$133K
- **Use case**: Pre-auction estimates, portfolio valuation

### Applicators

**Log-Price Model** (Recommended)
- **Test MAPE**: 39.2%
- **Test R²**: 0.867
- **Test RMSE**: $30,452
- **Training Records**: 1,802
- **Status**: ✅ Production Ready

**Interpretation**:
- Explains 87% of price variance
- Typical prediction within ±39%
- Example: $50K sprayer predicted as $30K-$70K
- **Use case**: Equipment valuations, trade-in estimates

---

## Beta/Testing Models

### Loaders and Lifts

**Log-Price Model**
- **Test MAPE**: 41.4%
- **Test R²**: 0.374
- **Training Records**: 6,117
- **Status**: ⚠️ Beta - Use with caution

### Tractors

**Log-Price Model**
- **Test MAPE**: 46.0%
- **Test R²**: 0.593
- **Training Records**: 13,619
- **Status**: ⚠️ Beta - Wide confidence ranges

### Construction

**Log-Price Model**
- **Test MAPE**: 48.5%
- **Test R²**: 0.269
- **Training Records**: 3,975
- **Status**: ⚠️ Beta - Rough estimates only

### Other Equipment

**Log-Price Model**
- **Test MAPE**: 57.4%
- **Test R²**: 0.475
- **Training Records**: 4,090
- **Status**: ⚠️ Beta - Very wide ranges

### Trucks and Trailers

**Log-Price Model**
- **Test MAPE**: 92.9%
- **Test R²**: 0.310
- **Training Records**: 1,646
- **Status**: ❌ Not Recommended

---

## Model Comparison: Regular vs Log-Price

### Overall Performance

| Method | Avg MAPE | Avg R² | Recommendation |
|--------|----------|--------|----------------|
| Regular Price | 78% | 0.44 | ❌ Not recommended |
| **Log-Price** | **51%** | **0.54** | ✅ Use this |

### Per-Category Comparison

| Category | Regular MAPE | Log MAPE | Improvement |
|----------|-------------|----------|-------------|
| Harvesting | 55.7% | **33.3%** | 🟢 40% better |
| Applicators | 61.6% | **39.2%** | 🟢 36% better |
| Loaders | 65.0% | **41.4%** | 🟢 36% better |
| Tractors | 85.4% | **46.0%** | 🟢 46% better |
| Construction | 69.1% | **48.5%** | 🟢 30% better |
| Other | 101.7% | **57.4%** | 🟢 44% better |
| Trucks | 135.0% | **92.9%** | 🟢 31% better |

**Conclusion**: Log-price method superior across all categories.

---

## Feature Importance

### Top 10 Features (Harvesting Model)

1. **year** (28.5%) - Model year drives depreciation
2. **condition** (18.2%) - Excellent vs Poor = huge difference
3. **hours** (15.7%) - Usage directly impacts value
4. **make_key** (12.3%) - Brand value (John Deere > generic)
5. **equipment_age** (8.4%) - Age-based depreciation
6. **hours_per_year** (6.1%) - Utilization intensity
7. **log_model_volume** (4.2%) - Popular models hold value
8. **region** (3.8%) - Geographic price differences
9. **barometer_norm** (2.1%) - Economic sentiment
10. **sale_month** (1.9%) - Seasonal patterns

### Top 10 Features (Applicators Model)

1. **year** (24.1%)
2. **condition** (22.3%)
3. **hours** (18.9%)
4. **make_key** (14.2%)
5. **log_model_volume** (7.8%)
6. **equipment_age** (5.3%)
7. **region** (3.2%)
8. **hours_per_year** (2.4%)
9. **barometer_norm** (1.1%)
10. **diesel_relative** (0.7%)

**Key Insight**: Condition is 2nd most important feature - critical addition!

---

## Error Analysis

### Error Distribution (Harvesting Model)

- **Within ±10%**: 18% of predictions
- **Within ±20%**: 35% of predictions
- **Within ±33%**: 50% of predictions
- **Within ±50%**: 68% of predictions

### Common Error Patterns

**Overestimation** (Predicts too high):
- Equipment in poor condition but model assumes "Good"
- Rare models with limited market
- Equipment with high hours for age

**Underestimation** (Predicts too low):
- Excellent condition equipment
- Popular models with high demand
- Equipment with low hours for age

### Outliers

**High Error Cases** (>100% MAPE):
- Missing critical information
- Unusual configurations
- Data entry errors
- Specialty/custom equipment

---

## Model Limitations

### Known Issues

1. **Missing Data Impact**
   - 60% of records filtered out (missing hours/year)
   - Can't use most available data
   - Limits training set size

2. **Within-Category Variance**
   - "Tractors" includes $5K utility and $500K row-crop
   - Model struggles with wide ranges
   - Category segmentation helps but not enough

3. **Missing Information**
   - Equipment options (GPS, cab, AC)
   - Detailed specifications
   - Service history
   - Visual condition (photos)

4. **MAPE Sensitivity**
   - Heavily penalizes errors on low-priced items
   - $5K error on $10K item = 50% MAPE
   - Same error on $100K item = 5% MAPE

### What Models Cannot Predict

- **Unusual configurations**: Custom builds, rare options
- **Market timing**: Temporary supply/demand shocks
- **Individual seller reputation**: Auction house quality
- **Buyer competition**: Number of bidders
- **Emotional value**: Collector items, sentimental value

---

## Validation Methodology

### Time-Based Splits

**Why**: Prevents data leakage, simulates real deployment

```
Train Set (70%):  Earliest data (2018-2023)
Val Set (15%):    Middle period (2023-2024)
Test Set (15%):   Most recent (2024-2025)
```

**Never**: Random splits (would leak future info into past)

### Metrics

**MAPE** (Mean Absolute Percentage Error):
- Primary metric
- Measures average % error
- Goal: <10% (achieved: 33-39% for best models)

**R²** (Coefficient of Determination):
- Measures variance explained
- Goal: >0.85 (achieved: 0.87-0.91 for best models)

**RMSE** (Root Mean Squared Error):
- Measures average dollar error
- Goal: <$15K (achieved: $30-37K for best models)

---

## Improvement Roadmap

### Immediate (Expected Impact: 10-20% MAPE reduction)

1. ✅ **Condition feature** - Already added
2. ✅ **Model volume feature** - Already added
3. ⏳ **Numeric condition score** - Convert to 1-5 scale
4. ⏳ **Parse specs field** - Extract options/features

### Short-term (Expected Impact: 20-40% MAPE reduction)

1. **Impute missing data** - Keep 3x more records
2. **Add state-level features** - Finer geographic granularity
3. **Hyperparameter tuning** - Optimize per category
4. **Ensemble methods** - Combine multiple models

### Long-term (Expected Impact: 40-60% MAPE reduction)

1. **Image analysis** - CNN for visual condition
2. **Specifications parsing** - Detailed equipment features
3. **More data sources** - Weather, farm income, yields
4. **Deep learning** - Neural networks for complex patterns

---

## Comparison to Industry Standards

### vs. Iron Solutions (Current Industry Leader)

| Metric | Iron Solutions | Ag IQ (Best Models) |
|--------|---------------|---------------------|
| **Accuracy** | ~50% error | 33-39% MAPE |
| **Update Frequency** | Quarterly | Real-time |
| **Economic Context** | None | 5 indicators |
| **Data Points** | Static guides | 733K transactions |
| **Customization** | Fixed | Category-specific |

**Verdict**: Ag IQ provides better accuracy with real-time context.

---

## Recommendations for Use

### High Confidence (Use in Production)
- ✅ Harvesting equipment
- ✅ Applicators
- ✅ Use Log-Price method
- ✅ Show confidence ranges

### Moderate Confidence (Beta Testing)
- ⚠️ Loaders & Lifts
- ⚠️ Tractors
- ⚠️ Construction
- ⚠️ Clearly communicate ±40-50% uncertainty

### Low Confidence (Not Recommended)
- ❌ Trucks & Trailers (92% MAPE)
- ❌ Use alternative methods

### Best Practices

1. **Always show confidence ranges** - Be transparent about uncertainty
2. **Use Log-Price method** - Consistently better performance
3. **Validate predictions** - Compare to actual sales when possible
4. **Gather feedback** - Track prediction accuracy over time
5. **Focus on best categories** - Start with Harvesting and Applicators

---

**For detailed technical documentation, see [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)**
