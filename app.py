"""
Ag IQ Equipment Valuation - Streamlit Interface
Multi-Model Version: Category-specific models with Make + Model selection

Core valuation models and methodologies are licensed from Dallas Polivka.
Copyright (c) 2025 Dallas Polivka
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.fmv import FMVModel
from src.models.fmv_log import FMVLogModel
from src.models.multi_model_config import get_model_display_name, CATEGORY_MODELS
from src.models.smart_router import SmartModelRouter

# Page configuration
st.set_page_config(
    page_title="Ag IQ Equipment Valuation",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f8f9fa;
        border-left: 5px solid #28a745;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .price-display {
        font-size: 3rem;
        font-weight: bold;
        color: #28a745;
    }
    .confidence-text {
        font-size: 1.1rem;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_router():
    """Get the smart model router (cached)."""
    return SmartModelRouter()


def check_available_models():
    """Check which models are available."""
    available = {'regular': [], 'log': []}
    
    for cat_key in list(CATEGORY_MODELS.keys()) + ['other']:
        # Check regular model
        if (Path(__file__).parent / "models" / f"fmv_{cat_key}").exists():
            available['regular'].append(cat_key)
        
        # Check log model
        if (Path(__file__).parent / "models" / f"fmv_{cat_key}_log").exists():
            available['log'].append(cat_key)
    
    return available


@st.cache_data
def load_reference_data():
    """Load reference data for dropdowns with category mapping."""
    try:
        # Try to load any available processed data
        data_options = [
            Path(__file__).parent / "data" / "processed" / "training_data.parquet",
            Path(__file__).parent / "data" / "processed" / "training_data_high_quality.parquet",
        ]
        
        df = None
        for data_path in data_options:
            if data_path.exists():
                df = pd.read_parquet(data_path)
                break
        
        if df is None:
            # If no processed data, return empty reference data
            return {
                'john-deere': {'makes': ['john-deere'], 'models': {}, 'regions': ['midwest']},
                'ford': {'makes': ['ford'], 'models': {}, 'regions': ['midwest']},
                'case-ih': {'makes': ['case-ih'], 'models': {}, 'regions': ['midwest']},
            }
        
        # Get unique values per category
        category_data = {}
        
        for cat_key, config in CATEGORY_MODELS.items():
            # Filter to this category
            mask = df['raw_category'].str.lower().str.contains('|'.join(config['filters']), na=False)
            cat_df = df[mask]
            
            category_data[cat_key] = {
                'makes': sorted([m for m in cat_df['make_key'].unique() if pd.notna(m)]),
                'models': {},  # Will populate when make is selected
                'regions': sorted([r for r in cat_df['region'].unique() if pd.notna(r)]),
            }
            
            # Get models per make for this category
            for make in category_data[cat_key]['makes']:
                make_models = cat_df[cat_df['make_key'] == make]['raw_model'].dropna().unique()
                category_data[cat_key]['models'][make] = sorted([str(m) for m in make_models if m])[:100]  # Limit to 100 models
        
        # Add "other" category
        category_data['other'] = {
            'makes': sorted([m for m in df['make_key'].unique() if pd.notna(m)]),
            'models': {},
            'regions': sorted([r for r in df['region'].unique() if pd.notna(r)]),
        }
        
        return category_data
    
    except Exception as e:
        st.error(f"Error loading reference data: {e}")
        return {}


def create_input_dataframe(inputs):
    """Create a DataFrame from user inputs for prediction."""
    current_date = datetime.now()
    
    # Create make_model_key from make and model
    make = inputs['make']
    model = inputs.get('model', '')
    
    if model and model != '(Any Model)':
        # Normalize model name
        model_normalized = str(model).lower().replace(' ', '-').replace('_', '-')
        make_model_key = f"{make}-{model_normalized}"
    else:
        make_model_key = f"{make}-na"
    
    return pd.DataFrame([{
        'sold_date': current_date,
        'year': float(inputs['year']),
        'hours': float(inputs['hours']),
        'raw_condition': inputs.get('condition', 'Good'),
        'make_key': make,  # Keep for backward compatibility
        'make_model_key': make_model_key,  # NEW: Combined identifier
        'raw_model': model if model and model != '(Any Model)' else 'Unknown',
        'region': inputs['region'],
        'raw_category': inputs['category_raw'],
        'barometer': float(inputs.get('barometer', 100)),
        'current_conditions': float(inputs.get('current_conditions', 95)),
        'future_expectations': float(inputs.get('future_expectations', 105)),
        'capital_investment_index': float(inputs.get('capital_investment', 80)),
        'diesel_price': float(inputs.get('diesel_price', 3.5)),
        'el_nino_phase': float(inputs.get('el_nino', 0.0)),
        'make_production_year_end': pd.NA,
    }])


def main():
    # Header
    st.markdown('<div class="main-header">🚜 Ag IQ Equipment Valuation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Category-Specific AI Models for Precision Valuations</div>', unsafe_allow_html=True)
    
    # Load reference data and smart router
    with st.spinner("Loading valuation system..."):
        router = get_router()
        ref_data = load_reference_data()
        available_models = router.list_available_models()
    
    # Show available models
    with st.expander("📚 Available Models"):
        st.write("**Make-Category Models:**", len(available_models.get('make_category', [])), "models")
        st.write("**Category Models:**", len(available_models.get('category', [])), "models")
        st.write("**Generic Models:**", len(available_models.get('generic', [])), "models")
        
        total = sum(len(v) for v in available_models.values())
        if total == 0:
            st.warning("No models trained yet. Run: python train_make_category_models.py")
    
    st.markdown("---")
    
    # Sidebar for inputs
    st.sidebar.header("Equipment Selection")
    
    # Model Method Selection
    st.sidebar.subheader("🔬 Prediction Method")
    model_method = st.sidebar.radio(
        "Choose prediction approach:",
        options=['regular', 'log'],
        format_func=lambda x: 'Regular Price' if x == 'regular' else 'Log-Price (Often Better)',
        help="Log-price transformation often improves accuracy on skewed data"
    )
    
    if model_method == 'log':
        st.sidebar.info("💡 Log-price models typically have lower MAPE")
    
    st.sidebar.markdown("---")
    
    # Step 1: Select Category
    st.sidebar.subheader("1️⃣ Select Category")
    
    category_options = {k: v['name'] for k, v in CATEGORY_MODELS.items()}
    category_options['other'] = 'Other Equipment'
    
    selected_category_key = st.sidebar.selectbox(
        "Equipment Category",
        options=list(category_options.keys()),
        format_func=lambda x: category_options[x],
        help="Select the type of equipment"
    )
    
    category_display_name = category_options[selected_category_key]
    
    # Step 2: Select Make
    st.sidebar.subheader("2️⃣ Select Make")
    
    if selected_category_key in ref_data and ref_data[selected_category_key]['makes']:
        available_makes = ref_data[selected_category_key]['makes']
    else:
        available_makes = ['john-deere', 'case-ih', 'new-holland', 'kubota']  # Fallback
    
    selected_make = st.sidebar.selectbox(
        "Manufacturer",
        options=available_makes,
        format_func=lambda x: x.replace('-', ' ').replace('_', ' ').title(),
        help="Select equipment manufacturer"
    )
    
    # Step 3: Select Model (optional but recommended)
    st.sidebar.subheader("3️⃣ Select Model (Optional)")
    
    available_models_for_make = []
    if (selected_category_key in ref_data and 
        selected_make in ref_data[selected_category_key]['models']):
        available_models_for_make = ref_data[selected_category_key]['models'][selected_make]
    
    if available_models_for_make:
        selected_model = st.sidebar.selectbox(
            "Specific Model",
            options=['(Any Model)'] + available_models_for_make,
            help="Select specific model for more accurate valuation"
        )
        if selected_model == '(Any Model)':
            selected_model = None
    else:
        selected_model = st.sidebar.text_input(
            "Model Name (Optional)",
            placeholder="e.g., 8320R, MX285",
            help="Enter specific model if known"
        )
        if not selected_model:
            selected_model = None
    
    # Step 4: Equipment Details
    st.sidebar.subheader("4️⃣ Equipment Details")
    
    year = st.sidebar.number_input(
        "Model Year",
        min_value=1980,
        max_value=datetime.now().year + 1,
        value=2018,
        help="Year manufactured"
    )
    
    hours = st.sidebar.number_input(
        "Equipment Hours",
        min_value=0,
        max_value=50000,
        value=1000,
        step=100,
        help="Total operating hours"
    )
    
    condition = st.sidebar.select_slider(
        "Condition",
        options=['Poor', 'Fair', 'Good', 'Excellent'],
        value='Good',
        help="Equipment condition - MAJOR price driver!"
    )
    
    # Get regions for this category
    if selected_category_key in ref_data and ref_data[selected_category_key]['regions']:
        available_regions = ref_data[selected_category_key]['regions']
    else:
        available_regions = ['midwest', 'great_plains', 'southeast', 'west']
    
    region = st.sidebar.selectbox(
        "Region",
        options=available_regions,
        help="Geographic region"
    )
    
    # Advanced options
    with st.sidebar.expander("⚙️ Economic Indicators"):
        st.caption("Optional - defaults provided")
        
        barometer = st.number_input(
            "Ag Economy Barometer",
            min_value=50,
            max_value=200,
            value=100,
            help="Current farmer sentiment (100 = neutral)"
        )
        
        diesel_price = st.number_input(
            "Diesel Price ($/gal)",
            min_value=1.0,
            max_value=10.0,
            value=3.5,
            step=0.1
        )
    
    # Get best available model using smart router (after make is selected)
    model, model_type_used = router.get_best_model(selected_make, category_display_name, model_method)
    
    if model is None:
        method_name = "Log-Price" if model_method == 'log' else "Regular"
        st.sidebar.error(f"⚠️ No {method_name} model found for {selected_make} - {category_display_name}")
        st.info(f"Train models using: `python train_make_category_models.py` or `python train_log_models.py`")
        return
    
    # Show which model is being used
    method_display = "Log-Price 📊" if model_method == 'log' else "Regular"
    
    if model_type_used == 'make_category':
        st.sidebar.success(f"✓ {selected_make.replace('-', ' ').title()} {category_display_name} ({method_display}) 🎯")
        st.sidebar.caption("Using brand-specific model for best accuracy!")
    elif model_type_used == 'category':
        st.sidebar.info(f"✓ Generic {category_display_name} ({method_display})")
        st.sidebar.caption(f"No {selected_make.replace('-', ' ').title()}-specific model available")
    else:
        st.sidebar.warning(f"✓ Generic Model ({method_display})")
        st.sidebar.caption("Using fallback model")
    
    # Get raw category value for the selected category
    if selected_category_key in CATEGORY_MODELS:
        category_raw = CATEGORY_MODELS[selected_category_key]['name']
    else:
        category_raw = 'Other'
    
    # Collect inputs
    inputs = {
        'category_key': selected_category_key,
        'category_raw': category_raw,
        'make': selected_make,
        'model': selected_model,
        'year': year,
        'hours': hours,
        'condition': condition,
        'region': region,
        'barometer': barometer,
        'diesel_price': diesel_price,
    }
    
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🔮 Get Valuation", type="primary", use_container_width=True)
    
    # Main content area
    if predict_button:
        with st.spinner("Analyzing equipment..."):
            input_df = create_input_dataframe(inputs)
            
            try:
                # Use category-specific model
                prediction = model.predict(input_df)[0]
                
                # Get model info from metadata
                model_info = {
                    'test_mape': model.metadata.get('metrics', {}).get('test_mape', 0),
                    'test_r2': model.metadata.get('metrics', {}).get('test_r2', 0),
                    'test_rmse': model.metadata.get('metrics', {}).get('test_rmse', 0),
                    'trained_at': model.metadata.get('trained_at'),
                    'n_train': model.metadata.get('n_train', 0),
                    'model_type': model.metadata.get('model_type', 'regular'),
                }
                
                # Try to load the OTHER method model for comparison
                other_method = 'log' if model_method == 'regular' else 'regular'
                other_model = load_model(selected_category_key, other_method)
                other_prediction = None
                other_info = None
                
                if other_model:
                    try:
                        other_prediction = other_model.predict(input_df)[0]
                        other_info = {
                            'test_mape': other_model.metadata.get('metrics', {}).get('test_mape', 0),
                            'test_r2': other_model.metadata.get('metrics', {}).get('test_r2', 0),
                        }
                    except:
                        pass
                
                # Display prediction
                method_name = "Log-Price" if model_method == 'log' else "Regular Price"
                
                if model_type_used == 'make_category':
                    model_desc = f"{selected_make.replace('-', ' ').title()} {category_display_name}"
                    badge = "🎯 Brand-Specific"
                elif model_type_used == 'category':
                    model_desc = f"{category_display_name}"
                    badge = "📊 Category"
                else:
                    model_desc = "Generic"
                    badge = "⚙️ Fallback"
                
                st.success(f"✅ Valuation Complete - {model_desc} ({method_name}) {badge}")
                
                # Comparison section if both models available
                if other_prediction and other_info:
                    st.info(f"💡 Comparison available! Both Regular and Log-Price models exist for {category_display_name}")
                
                # Main prediction display
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f'<div class="price-display">${prediction:,.0f}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="confidence-text">Estimated Fair Market Value</div>', unsafe_allow_html=True)
                
                with col2:
                    # Confidence range based on category model MAPE
                    mape = model_info['test_mape']
                    lower = max(0, prediction * (1 - mape/100))
                    upper = prediction * (1 + mape/100)
                    
                    st.markdown("**Confidence Range**")
                    st.markdown(f"${lower:,.0f} - ${upper:,.0f}")
                    st.caption(f"±{mape:.1f}% ({category_display_name} Model)")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Comparison view if other model available
                if other_prediction and other_info:
                    st.markdown("### ⚖️ Model Comparison")
                    
                    other_method_name = "Log-Price" if other_method == 'log' else "Regular Price"
                    current_method_name = "Log-Price" if model_method == 'log' else "Regular Price"
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            f"{current_method_name} (Current)",
                            f"${prediction:,.0f}",
                            f"MAPE: {model_info['test_mape']:.1f}%"
                        )
                    
                    with col2:
                        st.metric(
                            f"{other_method_name}",
                            f"${other_prediction:,.0f}",
                            f"MAPE: {other_info['test_mape']:.1f}%"
                        )
                    
                    with col3:
                        diff = abs(prediction - other_prediction)
                        diff_pct = (diff / prediction * 100) if prediction > 0 else 0
                        st.metric(
                            "Difference",
                            f"${diff:,.0f}",
                            f"{diff_pct:.1f}%"
                        )
                    
                    # Show which is better
                    if model_info['test_mape'] < other_info['test_mape']:
                        st.success(f"✅ {current_method_name} model has better accuracy ({model_info['test_mape']:.1f}% vs {other_info['test_mape']:.1f}% MAPE)")
                    else:
                        st.info(f"💡 {other_method_name} model has better accuracy ({other_info['test_mape']:.1f}% vs {model_info['test_mape']:.1f}% MAPE)")
                
                # Equipment summary
                st.markdown("### 📋 Equipment Summary")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Category", category_display_name)
                with col2:
                    st.metric("Make", selected_make.replace('-', ' ').replace('_', ' ').title())
                with col3:
                    st.metric("Model", selected_model if selected_model else "Generic")
                with col4:
                    st.metric("Year", year)
                with col5:
                    age = datetime.now().year - year
                    st.metric("Age", f"{age} yrs")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Hours", f"{hours:,}")
                with col2:
                    if age > 0:
                        hrs_per_year = hours / age
                        st.metric("Hours/Year", f"{hrs_per_year:,.0f}")
                    else:
                        st.metric("Hours/Year", "N/A")
                with col3:
                    st.metric("Region", region.replace('_', ' ').title())
                
                # Feature importance for this category
                st.markdown(f"### 📊 Key Price Drivers - {category_display_name}")
                
                importance_df = model.feature_importance()
                top_features = importance_df.head(10)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = [CATEGORY_MODELS.get(selected_category_key, {'color': '#28a745'})['color']] * len(top_features)
                ax.barh(range(len(top_features)), top_features['importance'], color=colors)
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features['feature'])
                ax.invert_yaxis()
                ax.set_xlabel('Importance (Gain)')
                ax.set_title(f'Top 10 Features for {category_display_name}')
                ax.grid(axis='x', alpha=0.3)
                
                st.pyplot(fig)
                
                # Insights
                st.markdown("### 💡 Valuation Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Utilization Analysis**")
                    if hours > 0 and age > 0:
                        hours_per_year = hours / age
                        if hours_per_year < 300:
                            usage = "Light usage - Premium value"
                            icon = "🟢"
                        elif hours_per_year < 600:
                            usage = "Normal usage - Good value"
                            icon = "🟡"
                        elif hours_per_year < 1000:
                            usage = "Heavy usage - Fair value"
                            icon = "🟠"
                        else:
                            usage = "Extreme usage - Reduced value"
                            icon = "🔴"
                        st.info(f"{icon} **{hours_per_year:.0f}** hrs/year: {usage}")
                    else:
                        st.info("Enter valid hours and year for analysis")
                
                with col2:
                    st.markdown("**Market Context**")
                    if barometer > 120:
                        sentiment = "Strong confidence - Prices trending up"
                        icon = "📈"
                    elif barometer > 90:
                        sentiment = "Stable market"
                        icon = "➡️"
                    else:
                        sentiment = "Weak sentiment - Softer prices"
                        icon = "📉"
                    st.info(f"{icon} Barometer: **{barometer}** - {sentiment}")
                
                # Model performance for this category
                with st.expander(f"📈 {category_display_name} Model Performance"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Test MAPE", f"{model_info['test_mape']:.2f}%")
                    with col2:
                        st.metric("Test R²", f"{model_info['test_r2']:.4f}")
                    with col3:
                        st.metric("Training Records", f"{model_info['n_train']:,}")
                    
                    st.caption(f"Model trained: {model_info['trained_at'][:10] if model_info['trained_at'] else 'N/A'}")
                
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.exception(e)
    
    else:
        # Initial state - show welcome message
        st.info("👈 Select equipment category, make, model, and details in the sidebar, then click **Get Valuation**")
        
        st.markdown("### 🎯 How It Works")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**1️⃣ Category**")
            st.write("Select equipment type")
        
        with col2:
            st.markdown("**2️⃣ Make & Model**")
            st.write("Choose manufacturer and model")
        
        with col3:
            st.markdown("**3️⃣ Details**")
            st.write("Year, hours, region")
        
        with col4:
            st.markdown("**4️⃣ Get Value**")
            st.write("Category-specific prediction")
        
        st.markdown("---")
        
        st.markdown("### 📈 Category-Specific Models")
        
        st.write("Each equipment category has its own dedicated model for maximum accuracy.")
        st.write("You can toggle between Regular and Log-Price prediction methods.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Make-Category Models:**")
            if available_models.get('make_category'):
                for model in available_models['make_category'][:5]:
                    display = model.replace('_', ' ').replace('-', ' ').title().replace(' Log', '')
                    st.write(f"- {display}")
                if len(available_models['make_category']) > 5:
                    st.write(f"- ... and {len(available_models['make_category']) - 5} more")
            else:
                st.write("*Run train_make_category_models.py*")
        
        with col2:
            st.markdown("**Category Models (Fallback):**")
            if available_models.get('category'):
                for model in available_models['category'][:5]:
                    display = model.replace('_', ' ').replace('-', ' ').title().replace(' Log', '')
                    st.write(f"- {display}")
                if len(available_models['category']) > 5:
                    st.write(f"- ... and {len(available_models['category']) - 5} more")
            else:
                st.write("*Run train_log_models.py*")
    
    # Footer
    st.markdown("---")
    st.caption("🚜 Ag IQ Equipment Valuation | Category-Specific Models | Built with Streamlit")
    st.caption("Core valuation models and methodologies licensed from Dallas Polivka © 2025")


if __name__ == "__main__":
    main()
