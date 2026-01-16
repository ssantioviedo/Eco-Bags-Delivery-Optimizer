# Priority Scoring System - January 2026 Updates

## Critical Fixes Implemented ✅

### Problem 1: Payment Scores Too Low Due to Outliers
**Issue**: Using actual min/max amounts ($757 to $29,040) caused most orders to score 1-16 points out of 100.

**Root Cause**: Extreme outliers in synthetic data distorted the normalization range.

**Solution**: Use 25th-75th percentile range instead of min/max for amount scoring.

```python
# OLD: Extreme range distortion
min_amount = $757 (outlier)
max_amount = $29,040 (outlier) 
score = ((amount - 757) / (29040 - 757)) * 100  # Most orders get <20

# NEW: Robust percentile-based scoring  
p25_amount = $2,266 (realistic minimum)
p75_amount = $5,269 (realistic maximum)
if amount <= p25: score = 20
elif amount >= p75: score = 100
else: score = 20 + ((amount - p25) / (p75 - p25)) * 80
```

**Result**: Payment scores now range 20-100 with meaningful differentiation.

### Problem 2: Insufficient Penalty for Overdue Orders
**Issue**: Orders 5 days past deadline only scored 80/100 urgency (should be maximum priority).

**Root Cause**: Linear normalization included overdue days in the same scale as future days.

**Solution**: Separate handling with aggressive overdue penalty.

```python
# OLD: Linear scaling
days_remaining = (deadline - today).days  # e.g., -5 days
normalized = (days - min_days) / (max_days - min_days)  # 0.2
score = 100 - (normalized * 100) = 80  # Too low!

# NEW: Overdue penalty system
if days_remaining < 0:
    score = min(150, 100 + abs(days_remaining) * 10)  # 150 for -5 days
else:
    score = 100 * (1 - days_remaining / max_days)
```

**Result**: Overdue orders get 110-150 urgency scores, properly prioritized.

### Problem 3: Illogical Payment Status Scoring
**Issue**: Sample order showed paid=1pt, pending=1pt, partial=3pts.

**Root Cause**: Base amount score was extremely low (1.6), so multipliers had minimal effect.

**Solution**: Fixed base scoring with percentiles + proper multiplier application.

```python
# Before: Base score too low
base_score = 1.6  # Due to outlier range
paid_score = 1.6 × 1.0 = 1.6
pending_score = 1.6 × 0.3 = 0.5  

# After: Realistic base scores
base_score = 100.0  # Order at 75th percentile
paid_score = 100.0 × 1.0 = 100.0
pending_score = 100.0 × 0.3 = 30.0
```

**Result**: Clear payment status differentiation.

## Updated Scoring Formula

```
PRIORITY = (0.40 × Urgency) + (0.25 × Payment) + (0.20 × Client) + (0.15 × Age)

Where:
• Urgency: 0-150 (overdue penalty: 100 + 10×days_overdue)
• Payment: (P25-P75 amount score 20-100) × status multiplier
• Client: Categorical scores based on type and history
• Age: Linear 0-100 based on days since order placed
```

## Data Range Improvements

| Component | Old Method | New Method | Benefit |
|-----------|------------|------------|---------|
| **Amount Range** | Min/Max ($757-$29,040) | P25/P75 ($2,266-$5,269) | ✅ Avoids outliers |
| **Urgency Range** | Linear -8 to +7 days | Overdue penalty system | ✅ Proper penalty |
| **Payment Base** | 0-100 linear | 20-100 with floor | ✅ Realistic minimum |

## Results Verification

### Score Distribution Analysis (41 pending orders):

**Overdue vs Non-Overdue**:
- Overdue (18 orders): 135.0 avg urgency, 84.5 avg final score
- Non-overdue (18 orders): 57.9 avg urgency, 42.0 avg final score
- ✅ **Clear differentiation**: Overdue orders score 2x higher

**Payment Status**:
- Paid (11 orders): 44.6 avg payment score, 79.0 avg final
- Partial (8 orders): 37.1 avg payment score, 62.4 avg final  
- Pending (17 orders): 17.5 avg payment score, 53.4 avg final
- ✅ **Logical progression**: paid > partial > pending

**Example Order Comparison**:
| Priority | Days to Deadline | Payment | Client | Score |
|----------|------------------|---------|---------|-------|
| 🔴 High | -6 (overdue) | partial | star | 104.0 |
| 🟡 Medium | 0 (due today) | paid | star | 66.2 |
| 🟢 Low | +7 (future) | partial | regular | 11.0 |

## Technical Implementation

### Functions Updated:
1. `calculate_urgency_score()`: Added overdue penalty logic
2. `calculate_payment_score()`: Percentile-based with 20-100 base range
3. `calculate_data_ranges()`: Returns P25/P75 instead of min/max

### Configuration Unchanged:
- Weights still configurable in `scoring_weights.json`
- Payment multipliers adjustable (paid=1.0, partial=0.6, pending=0.3)
- Client scoring thresholds remain flexible

## Database Updates

All 41 pending orders re-scored and updated with new priority scores:
- Mandatory orders: Still get infinite priority (999999)
- Regular orders: Now have realistic, differentiated scores
- Ready for optimizer phase (Phase 4)

## Key Takeaways

✅ **Outlier Handling**: Percentiles provide robust range estimation
✅ **Overdue Priority**: Critical orders get maximum attention  
✅ **Payment Logic**: Clear differentiation between payment statuses
✅ **Maintainable**: System adapts to actual data distribution
✅ **Flexible**: Configuration remains highly customizable

The scoring system now produces logical, defensible priorities that operators can trust for dispatch planning.