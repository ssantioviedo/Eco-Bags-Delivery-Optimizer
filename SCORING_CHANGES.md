# Priority Scoring System - Recent Changes

## Summary

The priority scoring system has been updated to use **dynamic scoring based on actual data** instead of hardcoded thresholds, and the **payment component has been simplified** to use a multiplier-based approach.

## Key Changes

### 1. Payment Scoring - From Split to Unified with Multiplier

**Before:**
- Payment split into two separate components:
  - `payment_status` (15% weight) - paid/partial/pending mapped to categorical scores
  - `payment_amount` (10% weight) - amount normalized between hardcoded min/max ($100 - $10,000)

**After:**
- Single `payment` component (25% weight)
- Formula: `payment_score = normalize(amount) × status_multiplier`
- Status multipliers:
  - `paid`: 1.0x (full amount score)
  - `partial`: 0.6x (60% of amount score)
  - `pending`: 0.3x (30% of amount score)

**Rationale:**
- Simpler, more intuitive approach
- Payment amount and status are inherently linked
- Higher amounts naturally get more weight, but status adjusts the score appropriately

### 2. Dynamic Ranges - No More Hardcoded Thresholds

**Before:**
- Hardcoded thresholds for all components:
  - Urgency: Fixed scores (100 for overdue, 95 for 1 day, 85 for 2 days, etc.)
  - Payment amount: Min $100, Max $10,000
  - Age: 14 days to reach 100 points

**After:**
- All ranges calculated from actual pending orders data:
  - `min_days_to_deadline` / `max_days_to_deadline`: Actual range in data (can include negative for overdue)
  - `min_amount` / `max_amount`: Actual amount range in current orders
  - `max_age_days`: Actual maximum age of orders

**Rationale:**
- Adapts to your real business conditions
- No arbitrary assumptions about ranges
- Scores remain meaningful even as order patterns change

### 3. Updated Scoring Formula

```
PRIORITY_SCORE = (w1 × URGENCY) + (w2 × PAYMENT) + (w3 × CLIENT) + (w4 × AGE)
```

Where:
- **URGENCY** (40% weight): Normalized from actual days_to_deadline range (inverted: sooner = higher)
- **PAYMENT** (25% weight): `normalize(amount) × status_multiplier`
- **CLIENT** (20% weight): Categorical scores based on client type (unchanged)
- **AGE** (15% weight): Normalized from actual age range (older = higher)

### 4. Function Signature Changes

#### `calculate_urgency_score()`
```python
# Before
calculate_urgency_score(deadline, config, reference_date=None)

# After  
calculate_urgency_score(deadline, reference_date=None, min_days=-30, max_days=30)
```

#### `calculate_payment_score()` (NEW - replaces two functions)
```python
# Before (two separate functions)
calculate_payment_status_score(payment_status, config)
calculate_payment_amount_score(total_amount, config)

# After (single unified function)
calculate_payment_score(total_amount, payment_status, config, min_amount=100.0, max_amount=10000.0)
```

#### `calculate_age_score()`
```python
# Before
calculate_age_score(issue_date, config, reference_date=None)

# After
calculate_age_score(issue_date, reference_date=None, max_days=30)
```

#### `calculate_priority_score()` (NEW parameter)
```python
# After
calculate_priority_score(order, client, historical_count, config, reference_date=None, data_ranges=None)
```

#### `get_scoring_breakdown()` (NEW parameter)
```python
# After
get_scoring_breakdown(order, client, historical_count, config, reference_date=None, data_ranges=None)
```

#### `calculate_data_ranges()` (NEW function)
```python
calculate_data_ranges(db, reference_date=None) -> dict
```
Returns:
```python
{
    'min_days_to_deadline': int,
    'max_days_to_deadline': int,
    'min_amount': float,
    'max_amount': float,
    'max_age_days': int,
}
```

## Configuration File Changes

### `config/scoring_weights.json`

**Before:**
```json
{
  "weights": {
    "urgency": 0.40,
    "payment_status": 0.15,
    "payment_amount": 0.10,
    "client": 0.20,
    "age": 0.15
  },
  "urgency_thresholds": {
    "overdue": 100,
    "1_day": 95,
    ...
  },
  "payment_amount_config": {
    "min_amount": 100,
    "max_amount": 10000
  },
  ...
}
```

**After:**
```json
{
  "weights": {
    "urgency": 0.40,
    "payment": 0.25,
    "client": 0.20,
    "age": 0.15
  },
  "payment_status_multipliers": {
    "paid": 1.0,
    "partial": 0.6,
    "pending": 0.3
  },
  ...
}
```

## Backward Compatibility

⚠️ **Breaking Changes:**
- Old notebooks/scripts that reference `payment_status` or `payment_amount` components separately will need updates
- Functions that imported `calculate_payment_status_score` or `calculate_payment_amount_score` need to use `calculate_payment_score`
- Scoring breakdowns now return single `payment` component instead of two

## Migration Guide

### For Notebooks
```python
# OLD
payment_status_score = calculate_payment_status_score(order['payment_status'], config)
payment_amount_score = calculate_payment_amount_score(order['total_amount'], config)

# NEW
payment_score = calculate_payment_score(
    order['total_amount'],
    order['payment_status'],
    config,
    min_amount=data_ranges['min_amount'],
    max_amount=data_ranges['max_amount']
)
```

### For Score Breakdown Parsing
```python
# OLD
breakdown['components']['payment_status']['raw']
breakdown['components']['payment_amount']['raw']

# NEW
breakdown['components']['payment']['raw']
```

## Benefits

1. **Simpler**: One payment component instead of two
2. **Adaptive**: Scoring adjusts to your actual data distribution
3. **More Accurate**: No guesswork about what threshold values to use
4. **Intuitive**: Multipliers are easier to understand than separate weights
5. **Maintainable**: Less configuration needed, system self-tunes to data

## Testing

The scoring system has been tested with:
- ✅ Imports work correctly
- ✅ Configuration loads successfully  
- ✅ Dynamic ranges calculation
- ✅ New payment score calculation with multipliers
- ✅ Backward-compatible default ranges when data unavailable

All functions maintain the same 0-100 score range for individual components.
