"""Priority scoring module for the Delivery Optimization System.

This module provides functions to calculate priority scores for orders based on
configurable weights and factors including urgency, payment status, client type,
and order age.
"""

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

from src.database import ClientModel, DatabaseManager, OrderModel


@dataclass
class ScoringConfig:
    """Container for scoring configuration loaded from JSON file."""

    # Weights (should sum to 1.0)
    weight_urgency: float = 0.40
    weight_payment: float = 0.25
    weight_client: float = 0.20
    weight_age: float = 0.15

    # Payment status multipliers
    payment_multiplier_paid: float = 1.0
    payment_multiplier_partial: float = 0.6
    payment_multiplier_pending: float = 0.3

    # Client scores
    client_star: int = 100
    client_new: int = 80
    client_frequent: int = 60
    client_regular: int = 40
    client_occasional: int = 20

    # Client order thresholds
    frequent_threshold: int = 5
    regular_threshold: int = 2

    # Mandatory score
    mandatory_score: float = 999999

    # Raw config for reference
    raw_config: dict = field(default_factory=dict)


def load_scoring_config(config_path: Path) -> ScoringConfig:
    """Load scoring configuration from JSON file.

    Args:
        config_path: Path to the scoring_weights.json file.

    Returns:
        ScoringConfig instance with loaded values.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        json.JSONDecodeError: If the config file is invalid JSON.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = ScoringConfig(
        # Weights
        weight_urgency=data["weights"]["urgency"],
        weight_payment=data["weights"]["payment"],
        weight_client=data["weights"]["client"],
        weight_age=data["weights"]["age"],
        # Payment status multipliers
        payment_multiplier_paid=data["payment_status_multipliers"]["paid"],
        payment_multiplier_partial=data["payment_status_multipliers"]["partial"],
        payment_multiplier_pending=data["payment_status_multipliers"]["pending"],
        # Client scores
        client_star=data["client_scores"]["star_client"],
        client_new=data["client_scores"]["new_client"],
        client_frequent=data["client_scores"]["frequent"],
        client_regular=data["client_scores"]["regular"],
        client_occasional=data["client_scores"]["occasional"],
        # Client thresholds
        frequent_threshold=data["client_order_thresholds"]["frequent"],
        regular_threshold=data["client_order_thresholds"]["regular"],
        # Mandatory
        mandatory_score=data["mandatory_score"],
        # Store raw config
        raw_config=data,
    )

    return config


def get_default_config() -> ScoringConfig:
    """Get default scoring configuration without loading from file.

    Returns:
        ScoringConfig instance with default values.
    """
    return ScoringConfig()


def calculate_urgency_score(
    delivery_deadline: date,
    reference_date: Optional[date] = None,
    min_days: int = -30,
    max_days: int = 30,
) -> float:
    """Calculate urgency score based on days until deadline.
    
    Uses actual data range for normalization. More urgent (closer/overdue) gets higher scores.
    Inverted scale: negative days (overdue) = high score, positive days (future) = lower score.
    
    Strong penalty for overdue orders: Each day overdue adds 10 bonus points.

    Args:
        delivery_deadline: The order's delivery deadline.
        reference_date: Reference date for calculation (defaults to today).
        min_days: Minimum days in the dataset (most urgent/overdue).
        max_days: Maximum days in the dataset (least urgent).

    Returns:
        Urgency score between 0 and 150 (can exceed 100 for overdue orders).
    """
    if reference_date is None:
        reference_date = date.today()

    days_remaining = (delivery_deadline - reference_date).days

    # Special handling for overdue orders
    if days_remaining < 0:
        # Overdue penalty: base score 100 + 10 points per day overdue
        overdue_penalty = abs(days_remaining) * 10
        return min(150.0, 100.0 + overdue_penalty)
    
    # For non-overdue orders, use normalized scoring
    days_remaining = max(0, min(days_remaining, max_days))
    
    # Scale: 0 days = 100 points, max_days = 0 points
    if max_days == 0:
        return 100.0
    
    score = 100.0 * (1 - (days_remaining / max_days))
    
    return max(0.0, min(100.0, score))


def calculate_payment_score(
    total_amount: float,
    payment_status: Optional[str],
    config: ScoringConfig,
    p15_amount: float = 1000.0,
    p85_amount: float = 5000.0,
) -> float:
    """Calculate payment score based on order amount and payment status.
    
    Formula: base_score × status_multiplier
    - base_score: Normalized from 25th to 75th percentile amounts (0-100)
    - status_multiplier: paid=1.0, partial=0.6, pending=0.3
    
    Using percentiles instead of min/max avoids outlier distortion.

    Args:
        total_amount: The order's total amount.
        payment_status: The order's payment status (paid, partial, pending).
        config: Scoring configuration.
        p15_amount: 15th percentile amount (robust minimum).
        p85_amount: 85th percentile amount (robust maximum).

    Returns:
        Payment score between 0 and 100.
    """
    # Calculate base score using interquartile range (0-100)
    range_size = p85_amount - p15_amount
    
    if range_size == 0:
        base_score = 100.0
    else:
        # Clamp to percentile range but allow scores outside for extreme values
        if total_amount <= p15_amount:
            base_score = 20.0  # Minimum score for low amounts
        elif total_amount >= p85_amount:
            base_score = 100.0  # Maximum score for high amounts
        else:
            # Linear interpolation between 15th and 85th percentile
            base_score = 20.0 + ((total_amount - p15_amount) / range_size) * 80.0
    
    # Get payment status multiplier
    if payment_status is None:
        payment_status = "pending"
    
    status = payment_status.lower().strip()
    if status == "paid":
        multiplier = config.payment_multiplier_paid
    elif status == "partial":
        multiplier = config.payment_multiplier_partial
    else:
        multiplier = config.payment_multiplier_pending
    
    final_score = base_score * multiplier
    
    return max(0.0, min(100.0, final_score))


def calculate_client_score(
    client: Optional[ClientModel],
    historical_order_count: int,
    config: ScoringConfig,
) -> float:
    """Calculate client score based on client type and history.

    The scoring priority is:
    1. Star clients get highest score
    2. New clients get second highest
    3. Frequent clients (> threshold orders) get third
    4. Regular clients (>= threshold orders) get fourth
    5. Occasional clients get lowest score

    Args:
        client: The client model (can be None for unknown clients).
        historical_order_count: Total number of orders from this client.
        config: Scoring configuration.

    Returns:
        Client score between 0 and 100.
    """
    if client is None:
        return float(config.client_occasional)

    # Star clients get highest priority
    if client.is_star_client:
        return float(config.client_star)

    # New clients get second highest
    if client.is_new_client:
        return float(config.client_new)

    # Check order history for frequency
    if historical_order_count > config.frequent_threshold:
        return float(config.client_frequent)
    elif historical_order_count >= config.regular_threshold:
        return float(config.client_regular)
    else:
        return float(config.client_occasional)


def get_client_type_label(
    client: Optional[ClientModel],
    historical_order_count: int,
    config: ScoringConfig,
) -> str:
    """Get human-readable client type label.

    Args:
        client: The client model.
        historical_order_count: Total number of orders from this client.
        config: Scoring configuration.

    Returns:
        String label describing client type.
    """
    if client is None:
        return "unknown"

    if client.is_star_client:
        return "star_client"
    elif client.is_new_client:
        return "new_client"
    elif historical_order_count > config.frequent_threshold:
        return "frequent"
    elif historical_order_count >= config.regular_threshold:
        return "regular"
    else:
        return "occasional"


def calculate_age_score(
    issue_date: date,
    reference_date: Optional[date] = None,
    max_days: int = 30,
) -> float:
    """Calculate age score based on days since order issue.
    
    Uses actual data range for normalization. Older orders get higher scores.

    Args:
        issue_date: The order's issue date.
        reference_date: Reference date for calculation (defaults to today).
        max_days: Maximum age in days found in the dataset.

    Returns:
        Age score between 0 and 100.
    """
    if reference_date is None:
        reference_date = date.today()

    days_old = (reference_date - issue_date).days
    
    # Clamp to range
    days_old = max(0, min(days_old, max_days))
    
    # Normalize: older = higher score
    if max_days == 0:
        return 100.0
    
    score = (days_old / max_days) * 100.0
    
    return max(0.0, min(100.0, score))


def calculate_priority_score(
    order: OrderModel,
    client: Optional[ClientModel],
    historical_order_count: int,
    config: ScoringConfig,
    reference_date: Optional[date] = None,
    data_ranges: Optional[dict] = None,
) -> float:
    """Calculate final priority score for an order.

    Formula: PRIORITY_SCORE = (w1 × URGENCY) + (w2 × PAYMENT) + (w3 × CLIENT) + (w4 × AGE)

    Where:
    - URGENCY: Based on days to deadline (dynamic range from data)
    - PAYMENT: Amount score × payment status multiplier (dynamic range from data)
    - CLIENT: Based on client type and history (categorical)
    - AGE: Based on days since issue (dynamic range from data)

    Exception: If is_mandatory = True, returns mandatory_score (999999).

    Args:
        order: The order model.
        client: The client model.
        historical_order_count: Total orders from this client.
        config: Scoring configuration.
        reference_date: Reference date for calculation (defaults to today).
        data_ranges: Dict with actual data ranges {
            'min_days_to_deadline': int,
            'max_days_to_deadline': int,
            'p15_amount': float,
            'p85_amount': float,
            'max_age_days': int
        }. If None, uses default values.

    Returns:
        Final priority score.
    """
    # Mandatory orders get maximum score
    if order.is_mandatory:
        return config.mandatory_score

    # Get data ranges (use defaults if not provided)
    if data_ranges is None:
        data_ranges = {
            'min_days_to_deadline': -30,
            'max_days_to_deadline': 30,
            'min_amount': 100.0,
            'max_amount': 10000.0,
            'max_age_days': 30,
        }

    # Calculate component scores
    urgency_score = calculate_urgency_score(
        order.delivery_deadline,
        reference_date=reference_date,
        min_days=data_ranges['min_days_to_deadline'],
        max_days=data_ranges['max_days_to_deadline'],
    )
    
    payment_score = calculate_payment_score(
        order.total_amount,
        order.payment_status,
        config,
        p15_amount=data_ranges['p15_amount'],
        p85_amount=data_ranges['p85_amount'],
    )
    
    client_score = calculate_client_score(client, historical_order_count, config)
    
    age_score = calculate_age_score(
        order.issue_date,
        reference_date=reference_date,
        max_days=data_ranges['max_age_days'],
    )

    # Calculate weighted final score
    final_score = (
        config.weight_urgency * urgency_score
        + config.weight_payment * payment_score
        + config.weight_client * client_score
        + config.weight_age * age_score
    )

    return final_score


def get_scoring_breakdown(
    order: OrderModel,
    client: Optional[ClientModel],
    historical_order_count: int,
    config: ScoringConfig,
    reference_date: Optional[date] = None,
    data_ranges: Optional[dict] = None,
) -> dict[str, Any]:
    """Get detailed breakdown of score components for an order.

    Args:
        order: The order model.
        client: The client model.
        historical_order_count: Total orders from this client.
        config: Scoring configuration.
        reference_date: Reference date for calculation (defaults to today).
        data_ranges: Dict with actual data ranges (same as calculate_priority_score).

    Returns:
        Dictionary with full score breakdown including:
        - order_id: Order identifier
        - final_score: Calculated priority score
        - is_mandatory: Whether order is mandatory
        - components: Raw and weighted scores for each factor
        - factors: Input values used for calculation
    """
    if reference_date is None:
        reference_date = date.today()

    # Get data ranges (use defaults if not provided)
    if data_ranges is None:
        data_ranges = {
            'min_days_to_deadline': -30,
            'max_days_to_deadline': 30,
            'min_amount': 100.0,
            'max_amount': 10000.0,
            'max_age_days': 30,
        }

    # Calculate component scores
    urgency_raw = calculate_urgency_score(
        order.delivery_deadline,
        reference_date=reference_date,
        min_days=data_ranges['min_days_to_deadline'],
        max_days=data_ranges['max_days_to_deadline'],
    )
    
    payment_raw = calculate_payment_score(
        order.total_amount,
        order.payment_status,
        config,
        p15_amount=data_ranges['p15_amount'],
        p85_amount=data_ranges['p85_amount'],
    )
    
    client_raw = calculate_client_score(client, historical_order_count, config)
    
    age_raw = calculate_age_score(
        order.issue_date,
        reference_date=reference_date,
        max_days=data_ranges['max_age_days'],
    )

    # Calculate weighted scores
    urgency_weighted = config.weight_urgency * urgency_raw
    payment_weighted = config.weight_payment * payment_raw
    client_weighted = config.weight_client * client_raw
    age_weighted = config.weight_age * age_raw

    payment_weighted = round(payment_weighted, 2)
    client_weighted = round(client_weighted, 2)
    age_weighted = round(age_weighted, 2)
    urgency_weighted = round(urgency_weighted, 2)

    # Calculate final score
    if order.is_mandatory:
        final_score = config.mandatory_score
    else:
        final_score = urgency_weighted + payment_weighted + client_weighted + age_weighted

    final_score = round(final_score, 2)

    # Get client type label
    client_type = get_client_type_label(client, historical_order_count, config)

    # Calculate days
    days_to_deadline = (order.delivery_deadline - reference_date).days
    days_since_issue = (reference_date - order.issue_date).days

    return {
        "order_id": order.order_id,
        "final_score": final_score,
        "is_mandatory": order.is_mandatory,
        "components": {
            "urgency": {"raw": urgency_raw, "weighted": urgency_weighted},
            "payment": {"raw": payment_raw, "weighted": payment_weighted},
            "client": {"raw": client_raw, "weighted": client_weighted},
            "age": {"raw": age_raw, "weighted": age_weighted},
        },
        "factors": {
            "days_to_deadline": days_to_deadline,
            "payment_status": order.payment_status or "pending",
            "total_amount": order.total_amount,
            "client_type": client_type,
            "days_since_issue": days_since_issue,
            "delivery_deadline": order.delivery_deadline,
            "issue_date": order.issue_date,
        },
        "weights": {
            "urgency": config.weight_urgency,
            "payment": config.weight_payment,
            "client": config.weight_client,
            "age": config.weight_age,
        },
    }


def calculate_data_ranges(
    db: DatabaseManager,
    reference_date: Optional[date] = None,
) -> dict:
    """Calculate actual min/max ranges from pending orders in database.
    
    This ensures scoring is based on real data distribution, not hardcoded values.
    
    Args:
        db: Database manager instance.
        reference_date: Reference date for calculations (defaults to today).
        
    Returns:
        Dict with keys:
        - min_days_to_deadline: Minimum days to deadline (can be negative for overdue)
        - max_days_to_deadline: Maximum days to deadline (excludes overdue for scaling)
        - p15_amount: 15th percentile order amount (robust minimum)
        - p85_amount: 85th percentile order amount (robust maximum)
        - max_age_days: Maximum order age in days
    """
    if reference_date is None:
        reference_date = date.today()
    
    with db.get_session() as session:
        # Get all pending orders
        orders = session.query(OrderModel).filter(OrderModel.status == "pending").all()
        
        if not orders:
            # Return default ranges if no data
            return {
                'min_days_to_deadline': -30,
                'max_days_to_deadline': 30,
                'p15_amount': 1000.0,
                'p85_amount': 5000.0,
                'max_age_days': 30,
            }
        
        # Calculate days to deadline for all orders
        days_to_deadline_list = [
            (order.delivery_deadline - reference_date).days
            for order in orders
        ]
        
        # Calculate age for all orders
        age_days_list = [
            (reference_date - order.issue_date).days
            for order in orders
        ]
        
        # Calculate amounts and percentiles
        amounts = [order.total_amount for order in orders]
        
        # Use percentiles for amount ranges to avoid outlier distortion
        import numpy as np
        p15_amount = float(np.percentile(amounts, 15))
        p85_amount = float(np.percentile(amounts, 85))
        
        # Filter non-overdue orders for deadline range calculation
        non_overdue_days = [d for d in days_to_deadline_list if d >= 0]
        max_non_overdue = max(non_overdue_days) if non_overdue_days else 30
        
        return {
            'min_days_to_deadline': min(days_to_deadline_list),
            'max_days_to_deadline': max_non_overdue,  # Use max non-overdue for scaling
            'p15_amount': p15_amount,
            'p85_amount': p85_amount,
            'max_age_days': max(age_days_list),
        }


def score_all_pending_orders(
    db: DatabaseManager,
    config_path: Optional[Path] = None,
    reference_date: Optional[date] = None,
) -> list[dict[str, Any]]:
    """Calculate and update priority scores for all pending orders.

    Args:
        db: Database manager instance.
        config_path: Path to scoring config file. Uses default if None.
        reference_date: Reference date for calculations (defaults to today).

    Returns:
        List of dictionaries with order_id, score, and component breakdown.
    """
    # Load config
    if config_path is not None:
        config = load_scoring_config(config_path)
    else:
        config = get_default_config()

    # Calculate actual data ranges from database
    data_ranges = calculate_data_ranges(db, reference_date)

    results = []

    with db.get_session() as session:
        # Get all pending orders with their clients
        orders = session.query(OrderModel).filter(OrderModel.status == "pending").all()

        for order in orders:
            # Get client
            client = (
                session.query(ClientModel)
                .filter(ClientModel.client_id == order.client_id)
                .first()
            )

            # Count historical orders for this client
            historical_count = (
                session.query(OrderModel)
                .filter(OrderModel.client_id == order.client_id)
                .count()
            )

            # Get full breakdown with actual data ranges
            breakdown = get_scoring_breakdown(
                order, client, historical_count, config, reference_date, data_ranges
            )

            # Store result
            results.append(breakdown)

    return results


def update_all_priority_scores(
    db: DatabaseManager,
    config_path: Optional[Path] = None,
    reference_date: Optional[date] = None,
) -> int:
    """Calculate and update priority scores for all pending orders in database.

    Args:
        db: Database manager instance.
        config_path: Path to scoring config file. Uses default if None.
        reference_date: Reference date for calculations (defaults to today).

    Returns:
        Number of orders updated.
    """
    # Get all scores
    results = score_all_pending_orders(db, config_path, reference_date)

    # Build update list
    scores = [
        {"order_id": r["order_id"], "priority_score": r["final_score"]}
        for r in results
    ]

    # Bulk update
    updated_count = db.bulk_update_priorities(scores)

    return updated_count


def explain_score_comparison(
    order1_breakdown: dict,
    order2_breakdown: dict,
) -> str:
    """Generate a human-readable explanation comparing two order scores.

    Args:
        order1_breakdown: Breakdown dict for first order.
        order2_breakdown: Breakdown dict for second order.

    Returns:
        Explanation string describing why one order has higher priority.
    """
    o1 = order1_breakdown
    o2 = order2_breakdown

    explanations = []

    # Compare final scores
    if o1["final_score"] > o2["final_score"]:
        higher, lower = o1, o2
        explanations.append(
            f"Order {higher['order_id']} has higher priority "
            f"({higher['final_score']:.1f} vs {lower['final_score']:.1f})"
        )
    else:
        higher, lower = o2, o1
        explanations.append(
            f"Order {higher['order_id']} has higher priority "
            f"({higher['final_score']:.1f} vs {lower['final_score']:.1f})"
        )

    # Check for mandatory
    if higher["is_mandatory"]:
        explanations.append(f"  - Order {higher['order_id']} is MANDATORY (maximum priority)")
        return "\n".join(explanations)

    # Compare components
    components = ["urgency", "payment", "client", "age"]
    for comp in components:
        h_raw = higher["components"][comp]["raw"]
        l_raw = lower["components"][comp]["raw"]
        h_weighted = higher["components"][comp]["weighted"]
        l_weighted = lower["components"][comp]["weighted"]

        if h_weighted > l_weighted:
            diff = h_weighted - l_weighted
            explanations.append(
                f"  - {comp.upper()}: {higher['order_id']} scores +{diff:.1f} higher "
                f"(raw: {h_raw:.0f} vs {l_raw:.0f})"
            )

    # Add factor details
    explanations.append("\nKey factors:")
    explanations.append(
        f"  - {higher['order_id']}: {higher['factors']['days_to_deadline']} days to deadline, "
        f"{higher['factors']['payment_status']} payment, {higher['factors']['client_type']} client"
    )
    explanations.append(
        f"  - {lower['order_id']}: {lower['factors']['days_to_deadline']} days to deadline, "
        f"{lower['factors']['payment_status']} payment, {lower['factors']['client_type']} client"
    )

    return "\n".join(explanations)
