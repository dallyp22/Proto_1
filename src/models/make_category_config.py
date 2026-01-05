"""
Configuration for Make + Category specific models.
Hybrid approach: Dedicated models for major brands, fallback to category models.
"""

# Make-Category combinations with enough data (>500 sales)
# These get dedicated models for best accuracy
MAKE_CATEGORY_MODELS = {
    # JOHN DEERE (Largest manufacturer - 57K+ sales)
    'john-deere_tractors': {
        'make': 'john-deere',
        'category': 'tractors',
        'min_records': 500,
        'expected_mape': '18-25%',
    },
    'john-deere_harvesting': {
        'make': 'john-deere',
        'category': 'harvesting',
        'min_records': 500,
        'expected_mape': '20-28%',
    },
    'john-deere_planting': {
        'make': 'john-deere',
        'category': 'planting',
        'min_records': 500,
        'expected_mape': '22-30%',
    },
    'john-deere_hay_forage': {
        'make': 'john-deere',
        'category': 'hay and forage',
        'min_records': 500,
        'expected_mape': '25-32%',
    },
    
    # FORD (2nd largest - 15K+ sales, mostly trucks)
    'ford_trucks': {
        'make': 'ford',
        'category': 'trucks and trailers',
        'min_records': 500,
        'expected_mape': '25-35%',
    },
    'ford_tractors': {
        'make': 'ford',
        'category': 'tractors',
        'min_records': 500,
        'expected_mape': '28-38%',
    },
    
    # CASE IH (3rd largest - 14K+ sales)
    'case-ih_tractors': {
        'make': 'case-ih',
        'category': 'tractors',
        'min_records': 500,
        'expected_mape': '20-28%',
    },
    'case-ih_harvesting': {
        'make': 'case-ih',
        'category': 'harvesting',
        'min_records': 500,
        'expected_mape': '22-30%',
    },
    
    # CHEVROLET (7.5K+ sales, mostly trucks)
    'chevrolet_trucks': {
        'make': 'chevrolet',
        'category': 'trucks and trailers',
        'min_records': 500,
        'expected_mape': '28-38%',
    },
    
    # NEW HOLLAND (7.4K+ sales)
    'new-holland_tractors': {
        'make': 'new-holland',
        'category': 'tractors',
        'min_records': 500,
        'expected_mape': '22-30%',
    },
    'new-holland_hay_forage': {
        'make': 'new-holland',
        'category': 'hay and forage',
        'min_records': 500,
        'expected_mape': '25-32%',
    },
}

# Category filters for matching raw_category values
CATEGORY_FILTERS = {
    'tractors': ['tractor', 'tractors'],
    'harvesting': ['harvesting', 'harvest', 'combines', 'combine'],
    'trucks and trailers': ['trucks and trailers', 'truck', 'trailer'],
    'planting': ['planting', 'planter', 'planters'],
    'hay and forage': ['hay and forage', 'hay', 'forage', 'baler'],
    'applicators': ['applicators', 'applicator', 'sprayer', 'sprayers'],
    'tillage': ['tillage', 'disk', 'plow', 'ripper'],
    'loaders and lifts': ['loaders and lifts', 'loader', 'lift', 'skid steer'],
    'construction': ['construction', 'excavator', 'dozer', 'backhoe'],
}


def get_make_category_key(make: str, category: str) -> str:
    """
    Generate model key for make-category combination.
    Returns None if no dedicated model exists.
    """
    # Normalize inputs
    make_lower = str(make).lower().strip()
    cat_lower = str(category).lower().strip()
    
    # Try exact match first
    key = f"{make_lower}_{cat_lower.replace(' ', '_').replace('and', '')}"
    if key in MAKE_CATEGORY_MODELS:
        return key
    
    # Try fuzzy match on category
    for model_key, config in MAKE_CATEGORY_MODELS.items():
        if config['make'] == make_lower:
            cat_filters = CATEGORY_FILTERS.get(config['category'], [])
            if any(cf in cat_lower for cf in cat_filters):
                return model_key
    
    return None


def get_all_make_category_keys() -> list:
    """Get all make-category model keys."""
    return list(MAKE_CATEGORY_MODELS.keys())


def should_train_make_category_model(make: str, category: str, record_count: int) -> bool:
    """Determine if a make-category combination has enough data for dedicated model."""
    key = get_make_category_key(make, category)
    if key and key in MAKE_CATEGORY_MODELS:
        min_records = MAKE_CATEGORY_MODELS[key]['min_records']
        return record_count >= min_records
    return False
