"""
Multi-model configuration for category-specific FMV models.
"""

# Category definitions for dedicated models
CATEGORY_MODELS = {
    'tractors': {
        'name': 'Tractors',
        'filters': ['tractors', 'tractor'],
        'min_price': 5000,
        'max_price': 500000,
        'color': '#2ecc71',  # Green
    },
    'trucks_and_trailers': {
        'name': 'Trucks and Trailers',
        'filters': ['trucks and trailers', 'truck', 'trailer'],
        'min_price': 2000,
        'max_price': 300000,
        'color': '#3498db',  # Blue
    },
    'harvesting': {
        'name': 'Harvesting',
        'filters': ['harvesting', 'harvest', 'combines', 'combine'],
        'min_price': 5000,
        'max_price': 600000,
        'color': '#f39c12',  # Orange
    },
    'loaders_and_lifts': {
        'name': 'Loaders and Lifts',
        'filters': ['loaders and lifts', 'loader', 'lift', 'skid steer'],
        'min_price': 3000,
        'max_price': 400000,
        'color': '#9b59b6',  # Purple
    },
    'construction': {
        'name': 'Construction',
        'filters': ['construction', 'excavator', 'dozer', 'backhoe'],
        'min_price': 5000,
        'max_price': 500000,
        'color': '#e74c3c',  # Red
    },
    'applicators': {
        'name': 'Applicators',
        'filters': ['applicators', 'applicator', 'sprayer', 'sprayers'],
        'min_price': 5000,
        'max_price': 500000,
        'color': '#1abc9c',  # Teal
    },
}

# Generic model for everything else
GENERIC_MODEL = {
    'name': 'Other Equipment',
    'min_price': 1000,
    'max_price': 500000,
    'color': '#95a5a6',  # Gray
}


def get_category_key(raw_category: str) -> str:
    """
    Map a raw_category value to a model category key.
    Returns 'other' if no match found.
    """
    if not raw_category or not isinstance(raw_category, str):
        return 'other'
    
    raw_lower = raw_category.lower()
    
    for key, config in CATEGORY_MODELS.items():
        for filter_term in config['filters']:
            if filter_term in raw_lower:
                return key
    
    return 'other'


def get_all_model_keys() -> list:
    """Get all model keys including 'other'."""
    return list(CATEGORY_MODELS.keys()) + ['other']


def get_model_display_name(model_key: str) -> str:
    """Get display name for a model key."""
    if model_key in CATEGORY_MODELS:
        return CATEGORY_MODELS[model_key]['name']
    return GENERIC_MODEL['name']

