"""Order selector module for dispatch candidate generation.

This module provides functions to generate dispatch candidates using various
selection strategies (greedy, zone-based, dynamic programming). It solves a
knapsack-style problem where we maximize priority while respecting truck capacity.
"""

import json
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Optional, List, Tuple

from src.database import DatabaseManager, OrderModel, ClientModel


class SelectionStrategy(Enum):
    """Available selection strategies for dispatch generation."""

    GREEDY_EFFICIENCY = "greedy_efficiency"
    GREEDY_PRIORITY = "greedy_priority"
    GREEDY_ZONE_CABA = "greedy_zone_caba"
    GREEDY_ZONE_NORTH = "greedy_zone_north"
    GREEDY_ZONE_SOUTH = "greedy_zone_south"
    GREEDY_ZONE_WEST = "greedy_zone_west"
    GREEDY_ZONE_SPILLOVER = "greedy_zone_spillover"
    GREEDY_BEST_FIT = "greedy_best_fit"
    DP_OPTIMAL = "dp_optimal"
    MANDATORY_FIRST = "mandatory_first"
    GREEDY_MANDATORY_NEAREST = "greedy_mandatory_nearest"


@dataclass
class OrderForSelection:
    """Simplified order representation for selection algorithms.

    Attributes:
        order_id: Unique identifier for the order.
        total_pallets: Number of pallets (weight for knapsack).
        priority_score: Priority value (value for knapsack).
        zone_id: Delivery zone identifier.
        is_mandatory: Whether this order must be included.
        client_name: Optional client business name for display.
        latitude: Optional delivery latitude for geographic calculations.
        longitude: Optional delivery longitude for geographic calculations.
    """

    order_id: str
    total_pallets: float
    priority_score: float
    zone_id: str
    is_mandatory: bool
    client_name: str = ""
    latitude: float = 0.0
    longitude: float = 0.0


@dataclass
class DispatchCandidate:
    """A potential dispatch configuration.

    Attributes:
        candidate_id: Unique identifier for this candidate.
        strategy: The selection strategy used to generate this candidate.
        order_ids: List of order IDs included in this dispatch.
        total_pallets: Sum of pallets across all orders.
        total_priority: Sum of priority scores across all orders.
        utilization_pct: Percentage of truck capacity used.
        zones: List of unique zones in this dispatch.
        zone_breakdown: Dictionary mapping zone_id to order count.
        zone_dispersion_penalty: Penalty factor based on zone count.
        adjusted_priority: Priority after applying zone penalty.
        includes_mandatory: Whether any mandatory orders are included.
        mandatory_count: Number of mandatory orders.
        is_single_zone: Whether all orders are in the same zone.
        orders: Optional list of full order details for export.
    """

    candidate_id: str
    strategy: SelectionStrategy
    order_ids: list[str]
    total_pallets: float
    total_priority: float
    utilization_pct: float
    zones: list[str]
    zone_breakdown: dict[str, int]
    zone_dispersion_penalty: float
    adjusted_priority: float
    includes_mandatory: bool
    mandatory_count: int
    is_single_zone: bool
    is_subset: bool = False
    orders: list[dict] = field(default_factory=list)


@dataclass
class SelectorConfig:
    """Configuration for the order selector.

    Attributes:
        nominal_capacity: Target truck capacity in pallets.
        min_acceptable: Minimum acceptable pallets for a valid dispatch.
        max_acceptable: Maximum preferred pallets.
        hard_max: Absolute maximum pallets allowed.
        min_for_zone_candidate: Minimum pallets for zone-specific candidates.
        single_zone_penalty: Penalty factor for single-zone dispatches.
        two_zones_penalty: Penalty factor for two-zone dispatches.
        three_plus_penalty: Penalty factor for 3+ zone dispatches.
        spillover_capacity_threshold: Remaining capacity to trigger spillover.
        spillover_priority_threshold: Marginal priority increase for spillover.
        multizone_exception_threshold: Priority threshold for multi-zone exception.
        dp_precision: Discretization precision for DP algorithm.
        max_candidates: Maximum number of candidates to return.
        ranking_weight_priority: Weight for priority in ranking.
        ranking_weight_utilization: Weight for utilization in ranking.
        ranking_weight_zone_coherence: Weight for zone coherence in ranking.
        zone_adjacency: Dictionary mapping zones to adjacent zones.
        strategies_enabled: List of enabled strategy names.
    """

    nominal_capacity: float = 8.0
    min_acceptable: float = 7.0
    max_acceptable: float = 8.5
    hard_max: float = 9.0
    min_for_zone_candidate: float = 4.0
    single_zone_penalty: float = 1.0
    two_zones_penalty: float = 0.95
    three_plus_penalty: float = 0.85
    spillover_capacity_threshold: float = 2.0
    spillover_priority_threshold: float = 0.15
    multizone_exception_threshold: float = 0.30
    dp_precision: float = 0.25
    max_candidates: int = 10
    ranking_weight_priority: float = 0.5
    ranking_weight_utilization: float = 0.3
    ranking_weight_zone_coherence: float = 0.2
    zone_adjacency: dict[str, list[str]] = field(default_factory=dict)
    strategies_enabled: list[str] = field(default_factory=list)


def load_selector_config(config_path: Path) -> SelectorConfig:
    """Load selector configuration from JSON file.

    Args:
        config_path: Path to the order_selector_config.json file.

    Returns:
        SelectorConfig instance with loaded values.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        json.JSONDecodeError: If the config file is invalid JSON.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return SelectorConfig(
        nominal_capacity=data["capacity"]["nominal"],
        min_acceptable=data["capacity"]["min_acceptable"],
        max_acceptable=data["capacity"]["max_acceptable"],
        hard_max=data["capacity"]["hard_max"],
        min_for_zone_candidate=data["capacity"]["min_for_zone_candidate"],
        single_zone_penalty=data["zone_penalties"]["single_zone"],
        two_zones_penalty=data["zone_penalties"]["two_zones"],
        three_plus_penalty=data["zone_penalties"]["three_plus_zones"],
        spillover_capacity_threshold=data["spillover"]["remaining_capacity_threshold"],
        spillover_priority_threshold=data["spillover"]["marginal_priority_threshold"],
        multizone_exception_threshold=data["multizone_exception_threshold"],
        dp_precision=data["dp_precision"],
        max_candidates=data["max_candidates"],
        ranking_weight_priority=data["ranking_weights"]["priority"],
        ranking_weight_utilization=data["ranking_weights"]["utilization"],
        ranking_weight_zone_coherence=data["ranking_weights"]["zone_coherence"],
        zone_adjacency=data["zone_adjacency"],
        strategies_enabled=data["strategies_enabled"],
    )


def get_default_config() -> SelectorConfig:
    """Get default selector configuration without loading from file.

    Returns:
        SelectorConfig instance with default values.
    """
    return SelectorConfig(
        zone_adjacency={
            "CABA": ["NORTH_ZONE", "WEST_ZONE", "SOUTH_ZONE"],
            "NORTH_ZONE": ["CABA", "WEST_ZONE"],
            "SOUTH_ZONE": ["CABA", "WEST_ZONE"],
            "WEST_ZONE": ["CABA", "NORTH_ZONE", "SOUTH_ZONE"],
        },
        strategies_enabled=[
            "greedy_efficiency",
            "greedy_priority",
            "greedy_zone_caba",
            "greedy_zone_north",
            "greedy_zone_south",
            "greedy_zone_west",
            "greedy_zone_spillover",
            "greedy_best_fit",
            "dp_optimal",
            "mandatory_first",
            "greedy_mandatory_nearest",
        ],
    )


# ============================================================================
# Data Loading Functions
# ============================================================================


def load_pending_orders(db: DatabaseManager) -> list[OrderForSelection]:
    """Load all pending orders formatted for selection.

    Args:
        db: DatabaseManager instance.

    Returns:
        List of OrderForSelection objects.
    """
    orders = []
    with db.get_session() as session:
        pending = (
            session.query(OrderModel, ClientModel)
            .join(ClientModel, OrderModel.client_id == ClientModel.client_id)
            .filter(OrderModel.status == "pending")
            .all()
        )

        for order, client in pending:
            orders.append(
                OrderForSelection(
                    order_id=order.order_id,
                    total_pallets=order.total_pallets,
                    priority_score=order.priority_score or 0.0,
                    zone_id=order.delivery_zone_id,
                    is_mandatory=order.is_mandatory,
                    client_name=client.business_name,
                    latitude=order.delivery_latitude,
                    longitude=order.delivery_longitude,
                )
            )

    return orders


def get_mandatory_orders(orders: list[OrderForSelection]) -> list[OrderForSelection]:
    """Filter mandatory orders that must be included.

    Args:
        orders: List of all orders.

    Returns:
        List of mandatory orders only.
    """
    return [o for o in orders if o.is_mandatory]


def get_non_mandatory_orders(orders: list[OrderForSelection]) -> list[OrderForSelection]:
    """Filter non-mandatory orders.

    Args:
        orders: List of all orders.

    Returns:
        List of non-mandatory orders only.
    """
    return [o for o in orders if not o.is_mandatory]


def calculate_mandatory_pallets(mandatory: list[OrderForSelection]) -> float:
    """Calculate pallets already consumed by mandatory orders.

    Args:
        mandatory: List of mandatory orders.

    Returns:
        Total pallets for mandatory orders.
    """
    return sum(o.total_pallets for o in mandatory)


def select_mandatory_subset(
    mandatory: list[OrderForSelection],
    max_capacity: float,
    strategy: str = "priority",
) -> tuple[list[OrderForSelection], list[OrderForSelection]]:
    """Select a subset of mandatory orders that fit within capacity.

    When mandatory orders exceed truck capacity, we must select which ones
    to include in this dispatch. The remaining mandatory orders will need
    to be handled in subsequent dispatches.

    Args:
        mandatory: All mandatory orders.
        max_capacity: Maximum pallet capacity for the dispatch.
        strategy: Selection strategy - "priority" (highest priority first),
                  "smallest" (smallest pallets first to fit more orders),
                  "zone_grouped" (group by zone, take highest priority zone).

    Returns:
        Tuple of (selected_mandatory, deferred_mandatory).
    """
    if not mandatory:
        return [], []

    total_pallets = calculate_mandatory_pallets(mandatory)
    if total_pallets <= max_capacity:
        return list(mandatory), []

    selected = []
    deferred = []
    current_pallets = 0.0

    if strategy == "smallest":
        # Sort by pallets ascending to fit more orders
        sorted_mandatory = sorted(mandatory, key=lambda o: o.total_pallets)
    elif strategy == "zone_grouped":
        # Group by zone, prioritize zone with highest total priority
        zone_priorities: dict[str, float] = {}
        for o in mandatory:
            zone_priorities[o.zone_id] = zone_priorities.get(o.zone_id, 0) + o.priority_score
        best_zone = max(zone_priorities, key=zone_priorities.get) if zone_priorities else None
        # Sort: best zone first, then by priority within zone
        sorted_mandatory = sorted(
            mandatory,
            key=lambda o: (0 if o.zone_id == best_zone else 1, -o.priority_score),
        )
    else:  # "priority" - default
        # Sort by priority descending (highest priority first)
        sorted_mandatory = sorted(mandatory, key=lambda o: -o.priority_score)

    for order in sorted_mandatory:
        if current_pallets + order.total_pallets <= max_capacity:
            selected.append(order)
            current_pallets += order.total_pallets
        else:
            deferred.append(order)

    return selected, deferred


def get_mandatory_overflow_info(
    mandatory: list[OrderForSelection],
    max_capacity: float,
) -> dict:
    """Get information about mandatory order overflow.

    Args:
        mandatory: All mandatory orders.
        max_capacity: Maximum pallet capacity.

    Returns:
        Dictionary with overflow information.
    """
    total_pallets = calculate_mandatory_pallets(mandatory)
    overflow = total_pallets - max_capacity

    return {
        "total_mandatory_orders": len(mandatory),
        "total_mandatory_pallets": round(total_pallets, 2),
        "max_capacity": max_capacity,
        "overflow_pallets": round(max(0, overflow), 2),
        "requires_multiple_dispatches": overflow > 0,
        "estimated_dispatches_needed": max(1, int(total_pallets / max_capacity) + 1),
    }


def select_mandatory_for_dispatch(
    mandatory: List[OrderForSelection],
    config: SelectorConfig,
    preferred_zone: Optional[str] = None,
    n_random_picks: int = 3,
) -> List[OrderForSelection]:
    """Select mandatory orders that fit in truck capacity using zone-based heuristics.

    Heuristic priority:
    1. Add mandatory orders from the same zone (or preferred_zone)
    2. Add mandatory orders from adjacent zones
    3. If still space, make random picks and keep best

    Never exceeds config.hard_max capacity.

    Args:
        mandatory: All mandatory orders.
        config: Selector configuration.
        preferred_zone: Optional zone to prioritize.
        n_random_picks: Number of random selections to try for best result.

    Returns:
        List of mandatory orders that fit within capacity.
    """
    if not mandatory:
        return []

    total_pallets = calculate_mandatory_pallets(mandatory)
    if total_pallets <= config.hard_max:
        return list(mandatory)

    # Determine the dominant zone among mandatory orders if not specified
    if preferred_zone is None:
        zone_pallets: dict[str, float] = {}
        for o in mandatory:
            zone_pallets[o.zone_id] = zone_pallets.get(o.zone_id, 0) + o.total_pallets
        preferred_zone = max(zone_pallets, key=zone_pallets.get) if zone_pallets else None

    # Get adjacent zones
    adjacent_zones = config.zone_adjacency.get(preferred_zone, []) if preferred_zone else []

    # Categorize mandatory orders by zone proximity
    same_zone = [o for o in mandatory if o.zone_id == preferred_zone]
    adjacent = [o for o in mandatory if o.zone_id in adjacent_zones]
    other = [o for o in mandatory if o.zone_id != preferred_zone and o.zone_id not in adjacent_zones]

    # Sort each category by priority
    same_zone.sort(key=lambda o: o.priority_score, reverse=True)
    adjacent.sort(key=lambda o: o.priority_score, reverse=True)
    other.sort(key=lambda o: o.priority_score, reverse=True)

    # Greedy fill: same zone first, then adjacent, then others
    selected = []
    current_pallets = 0.0

    for order in same_zone + adjacent + other:
        if current_pallets + order.total_pallets <= config.hard_max:
            selected.append(order)
            current_pallets += order.total_pallets

    # If we have room for random optimization, try a few random picks
    if n_random_picks > 0 and len(mandatory) > len(selected):
        best_selected = selected
        best_priority = sum(o.priority_score for o in selected)

        for _ in range(n_random_picks):
            # Random shuffle and greedy fill
            shuffled = list(mandatory)
            random.shuffle(shuffled)
            trial_selected = []
            trial_pallets = 0.0

            for order in shuffled:
                if trial_pallets + order.total_pallets <= config.hard_max:
                    trial_selected.append(order)
                    trial_pallets += order.total_pallets

            trial_priority = sum(o.priority_score for o in trial_selected)
            if trial_priority > best_priority:
                best_selected = trial_selected
                best_priority = trial_priority

        selected = best_selected

    return selected


def _run_all_strategies_on_subset(
    subset_orders: List[OrderForSelection],
    mandatory: List[OrderForSelection],
    config: SelectorConfig,
    is_subset: bool,
) -> List[DispatchCandidate]:
    """Run all selection strategies on a given subset of orders.

    Args:
        subset_orders: The subset of orders to run strategies on.
        mandatory: Mandatory orders to include (may be empty). Will be filtered
                   to fit within capacity using zone-based heuristics.
        config: Selector configuration.
        is_subset: Whether this is a subset (True) or full order set (False).

    Returns:
        List of DispatchCandidate objects.
    """
    candidates = []
    strategies = config.strategies_enabled

    # CRITICAL: Filter mandatory orders to fit within capacity
    # This prevents candidates with impossible pallet counts (e.g., 21.41 pallets)
    filtered_mandatory = select_mandatory_for_dispatch(mandatory, config)

    strategy_map = {
        "greedy_efficiency": (greedy_by_efficiency, SelectionStrategy.GREEDY_EFFICIENCY, {}),
        "greedy_priority": (greedy_by_priority, SelectionStrategy.GREEDY_PRIORITY, {}),
        "greedy_zone_caba": (greedy_by_zone, SelectionStrategy.GREEDY_ZONE_CABA, {"target_zone": "CABA"}),
        "greedy_zone_north": (greedy_by_zone, SelectionStrategy.GREEDY_ZONE_NORTH, {"target_zone": "NORTH_ZONE"}),
        "greedy_zone_south": (greedy_by_zone, SelectionStrategy.GREEDY_ZONE_SOUTH, {"target_zone": "SOUTH_ZONE"}),
        "greedy_zone_west": (greedy_by_zone, SelectionStrategy.GREEDY_ZONE_WEST, {"target_zone": "WEST_ZONE"}),
        "greedy_zone_spillover": (greedy_zone_with_spillover, SelectionStrategy.GREEDY_ZONE_SPILLOVER, {}),
        "greedy_best_fit": (greedy_best_fit, SelectionStrategy.GREEDY_BEST_FIT, {}),
        "dp_optimal": (dp_optimal_knapsack, SelectionStrategy.DP_OPTIMAL, {}),
        "mandatory_first": (mandatory_first, SelectionStrategy.MANDATORY_FIRST, {}),
        "greedy_mandatory_nearest": (greedy_mandatory_nearest, SelectionStrategy.GREEDY_MANDATORY_NEAREST, {}),
    }

    for strategy_name in strategies:
        if strategy_name not in strategy_map:
            continue

        func, strategy_enum, extra_kwargs = strategy_map[strategy_name]

        if "target_zone" in extra_kwargs:
            selected = func(subset_orders, extra_kwargs["target_zone"], filtered_mandatory, config)
        else:
            selected = func(subset_orders, filtered_mandatory, config)

        if selected:
            # Double-check: reject candidates that exceed hard_max (safety net)
            total_pallets = sum(o.total_pallets for o in selected)
            if total_pallets > config.hard_max:
                continue
            
            candidate = build_dispatch_candidate(selected, strategy_enum, config, is_subset=is_subset)
            if candidate:
                candidates.append(candidate)

    return candidates


def generate_candidates_by_subsets(
    orders: List[OrderForSelection],
    config: SelectorConfig,
    n_random_subsets: int = 10,
    top_n_for_random: int = 14,
    random_seed: Optional[int] = None,
) -> Tuple[List[DispatchCandidate], dict]:
    """Generate dispatch candidates by running all strategies on multiple subsets.

    Workflow:
    1. Run all strategies on FULL order set (is_subset=False)
    2. Run all strategies on NON-MANDATORY orders only (is_subset=True)
    3. Random subsets from ALL orders (including mandatory)
    4. Random subsets from ALL non-mandatory orders
    5. Random subsets from TOP priority orders (with mandatory)
    6. Random subsets from TOP priority non-mandatory orders

    Subset size minimum: 60% of total orders length.

    Args:
        orders: Full list of pending orders.
        config: Selector configuration.
        n_random_subsets: Number of random subsets per category.
        top_n_for_random: Consider top N orders by priority for "top priority" subsets.
        random_seed: Optional seed for reproducibility.

    Returns:
        Tuple of (candidates, generation_info) where generation_info contains subset details.
    """
    if random_seed is not None:
        random.seed(random_seed)

    all_candidates = []
    generation_info = {
        "full_set_count": 0,
        "non_mandatory_set_count": 0,
        "random_all_orders_count": 0,
        "random_non_mandatory_count": 0,
        "random_top_with_mandatory_count": 0,
        "random_top_without_mandatory_count": 0,
        "subsets_generated": [],
    }

    # Separate mandatory and non-mandatory orders
    mandatory_orders = [o for o in orders if o.is_mandatory]
    non_mandatory_orders = [o for o in orders if not o.is_mandatory]

    # Minimum subset size: 60% of total orders
    min_subset_size = max(3, int(len(orders) * 0.6))

    # 1. FULL ORDER SET (is_subset=False)
    candidates_full = _run_all_strategies_on_subset(
        orders, mandatory_orders, config, is_subset=False
    )
    all_candidates.extend(candidates_full)
    generation_info["full_set_count"] = len(candidates_full)
    generation_info["subsets_generated"].append({
        "name": "full_set",
        "order_count": len(orders),
        "includes_mandatory": True,
        "candidates_generated": len(candidates_full),
    })

    # 2. NON-MANDATORY ORDERS ONLY (is_subset=True)
    if len(non_mandatory_orders) >= min_subset_size:
        candidates_non_mand = _run_all_strategies_on_subset(
            non_mandatory_orders, [], config, is_subset=True
        )
        all_candidates.extend(candidates_non_mand)
        generation_info["non_mandatory_set_count"] = len(candidates_non_mand)
        generation_info["subsets_generated"].append({
            "name": "non_mandatory_only",
            "order_count": len(non_mandatory_orders),
            "includes_mandatory": False,
            "candidates_generated": len(candidates_non_mand),
        })

    # Prepare pools for random subsets
    top_priority_orders = sorted(orders, key=lambda o: o.priority_score, reverse=True)[:top_n_for_random]
    top_priority_non_mandatory = sorted(non_mandatory_orders, key=lambda o: o.priority_score, reverse=True)[:top_n_for_random]

    # Helper function to generate random subset and run strategies
    def _generate_random_subset_candidates(
        pool: List[OrderForSelection],
        include_mandatory: bool,
        subset_name: str,
    ) -> int:
        """Generate candidates from a random subset of the pool."""
        if len(pool) < min_subset_size:
            return 0

        # Random subset size: between min_subset_size and len(pool)
        subset_size = random.randint(min_subset_size, len(pool))
        subset = random.sample(pool, subset_size)

        # Determine which mandatory orders to include (from subset or separately)
        if include_mandatory:
            subset_mandatory = [o for o in subset if o.is_mandatory]
            # If no mandatory in subset, add some from mandatory_orders
            if not subset_mandatory and mandatory_orders:
                # Randomly pick some mandatory orders to include
                n_mandatory_to_add = min(len(mandatory_orders), random.randint(1, len(mandatory_orders)))
                subset_mandatory = random.sample(mandatory_orders, n_mandatory_to_add)
                subset = subset + [o for o in subset_mandatory if o not in subset]
        else:
            subset_mandatory = []
            # Remove any mandatory from subset
            subset = [o for o in subset if not o.is_mandatory]

        if len(subset) < 3:
            return 0

        candidates_random = _run_all_strategies_on_subset(
            subset, subset_mandatory, config, is_subset=True
        )
        all_candidates.extend(candidates_random)
        generation_info["subsets_generated"].append({
            "name": subset_name,
            "order_count": len(subset),
            "includes_mandatory": include_mandatory,
            "candidates_generated": len(candidates_random),
        })
        return len(candidates_random)

    # 3. RANDOM SUBSETS FROM ALL ORDERS (including mandatory)
    n_per_category = max(1, n_random_subsets // 4)
    for i in range(n_per_category):
        count = _generate_random_subset_candidates(
            list(orders), True, f"random_all_orders_{i+1}"
        )
        generation_info["random_all_orders_count"] += count

    # 4. RANDOM SUBSETS FROM ALL NON-MANDATORY ORDERS
    for i in range(n_per_category):
        count = _generate_random_subset_candidates(
            list(non_mandatory_orders), False, f"random_non_mandatory_{i+1}"
        )
        generation_info["random_non_mandatory_count"] += count

    # 5. RANDOM SUBSETS FROM TOP PRIORITY ORDERS (with mandatory)
    for i in range(n_per_category):
        count = _generate_random_subset_candidates(
            list(top_priority_orders), True, f"random_top_with_mandatory_{i+1}"
        )
        generation_info["random_top_with_mandatory_count"] += count

    # 6. RANDOM SUBSETS FROM TOP PRIORITY NON-MANDATORY ORDERS
    for i in range(n_per_category):
        count = _generate_random_subset_candidates(
            list(top_priority_non_mandatory), False, f"random_top_without_mandatory_{i+1}"
        )
        generation_info["random_top_without_mandatory_count"] += count

    return all_candidates, generation_info


# ============================================================================
# Zone and Penalty Functions
# ============================================================================


def get_zone_dispersion_penalty(zones: list[str], config: SelectorConfig) -> float:
    """Calculate penalty factor based on zone count.

    Args:
        zones: List of unique zone IDs.
        config: Selector configuration.

    Returns:
        Penalty factor (1.0 = no penalty, <1.0 = penalty applied).
    """
    zone_count = len(set(zones))
    if zone_count <= 1:
        return config.single_zone_penalty
    elif zone_count == 2:
        return config.two_zones_penalty
    else:
        return config.three_plus_penalty


def get_unique_zones(orders: list[OrderForSelection]) -> list[str]:
    """Get unique zones from a list of orders.

    Args:
        orders: List of orders.

    Returns:
        List of unique zone IDs.
    """
    return list(set(o.zone_id for o in orders))


def get_zone_breakdown(orders: list[OrderForSelection]) -> dict[str, int]:
    """Get order count per zone.

    Args:
        orders: List of orders.

    Returns:
        Dictionary mapping zone_id to order count.
    """
    breakdown: dict[str, int] = {}
    for order in orders:
        breakdown[order.zone_id] = breakdown.get(order.zone_id, 0) + 1
    return breakdown


def get_dominant_zone(orders: list[OrderForSelection]) -> str:
    """Get the zone with the highest total priority.

    Args:
        orders: List of orders.

    Returns:
        Zone ID with highest total priority.
    """
    zone_priorities: dict[str, float] = {}
    for order in orders:
        zone_priorities[order.zone_id] = (
            zone_priorities.get(order.zone_id, 0) + order.priority_score
        )
    if not zone_priorities:
        return "CABA"
    return max(zone_priorities, key=zone_priorities.get)


# ============================================================================
# Capacity Validation Functions
# ============================================================================


def is_within_capacity_range(
    total_pallets: float,
    min_capacity: float = 7.0,
    max_capacity: float = 8.5,
    hard_max: float = 9.0,
) -> bool:
    """Check if pallet count is within acceptable range.

    Args:
        total_pallets: Total number of pallets.
        min_capacity: Minimum acceptable capacity.
        max_capacity: Maximum preferred capacity.
        hard_max: Absolute maximum allowed.

    Returns:
        True if within acceptable range.
    """
    return min_capacity <= total_pallets <= hard_max


def is_within_preferred_range(
    total_pallets: float,
    min_capacity: float = 7.0,
    max_capacity: float = 8.5,
) -> bool:
    """Check if pallet count is within preferred range.

    Args:
        total_pallets: Total number of pallets.
        min_capacity: Minimum acceptable capacity.
        max_capacity: Maximum preferred capacity.

    Returns:
        True if within preferred range.
    """
    return min_capacity <= total_pallets <= max_capacity


# ============================================================================
# Selection Strategy Implementations
# ============================================================================


def greedy_by_efficiency(
    orders: list[OrderForSelection],
    mandatory: list[OrderForSelection],
    config: SelectorConfig,
) -> list[OrderForSelection]:
    """Select orders using efficiency ratio (priority/pallets).

    Args:
        orders: All available orders.
        mandatory: Mandatory orders (already included).
        config: Selector configuration.

    Returns:
        Selected orders including mandatory.
    """
    selected = list(mandatory)
    current_pallets = calculate_mandatory_pallets(mandatory)
    mandatory_ids = {o.order_id for o in mandatory}

    # Get non-mandatory orders and sort by efficiency
    available = [o for o in orders if o.order_id not in mandatory_ids]
    available.sort(key=lambda o: o.priority_score / max(o.total_pallets, 0.1), reverse=True)

    for order in available:
        if current_pallets + order.total_pallets <= config.max_acceptable:
            selected.append(order)
            current_pallets += order.total_pallets
        elif current_pallets >= config.min_acceptable:
            break
        elif current_pallets + order.total_pallets <= config.hard_max:
            # Consider adding if we're below minimum
            selected.append(order)
            current_pallets += order.total_pallets

    return selected


def greedy_by_priority(
    orders: list[OrderForSelection],
    mandatory: list[OrderForSelection],
    config: SelectorConfig,
) -> list[OrderForSelection]:
    """Select orders by highest priority first.

    Args:
        orders: All available orders.
        mandatory: Mandatory orders (already included).
        config: Selector configuration.

    Returns:
        Selected orders including mandatory.
    """
    selected = list(mandatory)
    current_pallets = calculate_mandatory_pallets(mandatory)
    mandatory_ids = {o.order_id for o in mandatory}

    # Get non-mandatory orders and sort by priority
    available = [o for o in orders if o.order_id not in mandatory_ids]
    available.sort(key=lambda o: o.priority_score, reverse=True)

    for order in available:
        if current_pallets + order.total_pallets <= config.max_acceptable:
            selected.append(order)
            current_pallets += order.total_pallets
        elif current_pallets >= config.min_acceptable:
            break
        elif current_pallets + order.total_pallets <= config.hard_max:
            selected.append(order)
            current_pallets += order.total_pallets

    return selected


def greedy_by_zone(
    orders: list[OrderForSelection],
    target_zone: str,
    mandatory: list[OrderForSelection],
    config: SelectorConfig,
) -> list[OrderForSelection]:
    """Select orders from a specific zone only.

    Args:
        orders: All available orders.
        target_zone: Zone ID to filter by.
        mandatory: Mandatory orders (always included).
        config: Selector configuration.

    Returns:
        Selected orders including mandatory, filtered by zone.
        Returns empty list if utilization below minimum threshold.
    """
    # Include mandatory orders that are in the target zone
    zone_mandatory = [o for o in mandatory if o.zone_id == target_zone]
    other_mandatory = [o for o in mandatory if o.zone_id != target_zone]

    selected = list(zone_mandatory)
    current_pallets = calculate_mandatory_pallets(zone_mandatory)
    mandatory_ids = {o.order_id for o in mandatory}

    # Get zone-specific non-mandatory orders
    zone_orders = [
        o for o in orders if o.zone_id == target_zone and o.order_id not in mandatory_ids
    ]
    zone_orders.sort(key=lambda o: o.priority_score, reverse=True)

    for order in zone_orders:
        if current_pallets + order.total_pallets <= config.max_acceptable:
            selected.append(order)
            current_pallets += order.total_pallets
        elif current_pallets >= config.min_acceptable:
            break
        elif current_pallets + order.total_pallets <= config.hard_max:
            selected.append(order)
            current_pallets += order.total_pallets

    # Check minimum threshold
    if current_pallets < config.min_for_zone_candidate:
        return []

    # Add other mandatory orders if they fit
    for order in other_mandatory:
        if current_pallets + order.total_pallets <= config.hard_max:
            selected.append(order)
            current_pallets += order.total_pallets

    return selected


def greedy_zone_with_spillover(
    orders: list[OrderForSelection],
    mandatory: list[OrderForSelection],
    config: SelectorConfig,
) -> list[OrderForSelection]:
    """Select from dominant zone with controlled spillover.

    Args:
        orders: All available orders.
        mandatory: Mandatory orders (always included).
        config: Selector configuration.

    Returns:
        Selected orders with zone-dominant approach.
    """
    selected = list(mandatory)
    current_pallets = calculate_mandatory_pallets(mandatory)
    mandatory_ids = {o.order_id for o in mandatory}

    # Identify dominant zone
    available = [o for o in orders if o.order_id not in mandatory_ids]
    if not available and not mandatory:
        return []

    dominant_zone = get_dominant_zone(orders)

    # Fill from dominant zone first
    dominant_orders = [o for o in available if o.zone_id == dominant_zone]
    dominant_orders.sort(key=lambda o: o.priority_score, reverse=True)

    for order in dominant_orders:
        if current_pallets + order.total_pallets <= config.max_acceptable:
            selected.append(order)
            current_pallets += order.total_pallets
        elif current_pallets >= config.min_acceptable:
            break

    # Check if spillover is warranted
    remaining_capacity = config.max_acceptable - current_pallets

    if remaining_capacity > config.spillover_capacity_threshold:
        # Calculate current total priority
        current_priority = sum(o.priority_score for o in selected)

        # Get adjacent zone orders
        adjacent_zones = config.zone_adjacency.get(dominant_zone, [])
        adjacent_orders = [
            o
            for o in available
            if o.zone_id in adjacent_zones and o.order_id not in {o.order_id for o in selected}
        ]
        adjacent_orders.sort(key=lambda o: o.priority_score, reverse=True)

        for order in adjacent_orders:
            if current_pallets + order.total_pallets <= config.max_acceptable:
                marginal_increase = order.priority_score / max(current_priority, 1)
                if marginal_increase >= config.spillover_priority_threshold:
                    selected.append(order)
                    current_pallets += order.total_pallets
                    current_priority += order.priority_score
            elif current_pallets >= config.min_acceptable:
                break

    return selected


def greedy_best_fit(
    orders: list[OrderForSelection],
    mandatory: list[OrderForSelection],
    config: SelectorConfig,
) -> list[OrderForSelection]:
    """Maximize utilization closeness to target, then priority.

    Args:
        orders: All available orders.
        mandatory: Mandatory orders (always included).
        config: Selector configuration.

    Returns:
        Selected orders optimized for utilization fit.
    """
    selected = list(mandatory)
    current_pallets = calculate_mandatory_pallets(mandatory)
    mandatory_ids = {o.order_id for o in mandatory}

    available = [o for o in orders if o.order_id not in mandatory_ids]

    # Score each order by how well it fits + priority
    target = config.nominal_capacity
    penalty_factor = 50.0  # Higher = more emphasis on fit

    def fit_score(order: OrderForSelection, current: float) -> float:
        new_total = current + order.total_pallets
        if new_total > config.hard_max:
            return -float("inf")
        fit_penalty = abs(target - new_total) * penalty_factor
        return order.priority_score - fit_penalty

    while available:
        # Find best order to add
        best_order = None
        best_score = -float("inf")

        for order in available:
            score = fit_score(order, current_pallets)
            if score > best_score and current_pallets + order.total_pallets <= config.hard_max:
                best_score = score
                best_order = order

        if best_order is None:
            break

        # Add if still beneficial
        new_total = current_pallets + best_order.total_pallets
        if new_total > config.max_acceptable and current_pallets >= config.min_acceptable:
            break

        selected.append(best_order)
        current_pallets += best_order.total_pallets
        available.remove(best_order)

        if current_pallets >= config.max_acceptable:
            break

    return selected


def dp_optimal_knapsack(
    orders: list[OrderForSelection],
    mandatory: list[OrderForSelection],
    config: SelectorConfig,
) -> list[OrderForSelection]:
    """Solve knapsack optimally using dynamic programming.

    Args:
        orders: All available orders.
        mandatory: Mandatory orders (always included).
        config: Selector configuration.

    Returns:
        Optimal selection of orders.
    """
    selected_mandatory = list(mandatory)
    mandatory_pallets = calculate_mandatory_pallets(mandatory)
    mandatory_ids = {o.order_id for o in mandatory}

    available = [o for o in orders if o.order_id not in mandatory_ids]

    if not available:
        return selected_mandatory

    # Discretize capacity
    precision = config.dp_precision
    remaining_capacity = config.max_acceptable - mandatory_pallets
    if remaining_capacity <= 0:
        return selected_mandatory

    capacity_units = int(remaining_capacity / precision) + 1

    # Build DP table
    n = len(available)
    dp = [[0.0] * (capacity_units + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        order = available[i - 1]
        order_units = int(order.total_pallets / precision)

        for w in range(capacity_units + 1):
            # Don't take this order
            dp[i][w] = dp[i - 1][w]
            # Take this order if it fits
            if order_units <= w:
                value_with = dp[i - 1][w - order_units] + order.priority_score
                if value_with > dp[i][w]:
                    dp[i][w] = value_with

    # Backtrack to find selected orders
    selected = list(selected_mandatory)
    w = capacity_units

    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            order = available[i - 1]
            selected.append(order)
            w -= int(order.total_pallets / precision)

    return selected


def mandatory_first(
    orders: list[OrderForSelection],
    mandatory: list[OrderForSelection],
    config: SelectorConfig,
) -> list[OrderForSelection]:
    """Start with mandatory orders, fill rest from same zone.

    Args:
        orders: All available orders.
        mandatory: Mandatory orders (always included).
        config: Selector configuration.

    Returns:
        Selected orders starting with mandatory.
    """
    if not mandatory:
        # Fall back to greedy by priority if no mandatory
        return greedy_by_priority(orders, [], config)

    selected = list(mandatory)
    current_pallets = calculate_mandatory_pallets(mandatory)
    mandatory_ids = {o.order_id for o in mandatory}

    # Determine majority zone of mandatory orders
    zone_counts = get_zone_breakdown(mandatory)
    majority_zone = max(zone_counts, key=zone_counts.get) if zone_counts else None

    available = [o for o in orders if o.order_id not in mandatory_ids]

    # Prioritize same zone as majority of mandatory
    if majority_zone:
        same_zone = [o for o in available if o.zone_id == majority_zone]
        other_zone = [o for o in available if o.zone_id != majority_zone]

        same_zone.sort(
            key=lambda o: o.priority_score / max(o.total_pallets, 0.1), reverse=True
        )
        other_zone.sort(
            key=lambda o: o.priority_score / max(o.total_pallets, 0.1), reverse=True
        )

        available = same_zone + other_zone

    for order in available:
        if current_pallets + order.total_pallets <= config.max_acceptable:
            selected.append(order)
            current_pallets += order.total_pallets
        elif current_pallets >= config.min_acceptable:
            break
        elif current_pallets + order.total_pallets <= config.hard_max:
            selected.append(order)
            current_pallets += order.total_pallets

    return selected


def greedy_mandatory_nearest(
    orders: list[OrderForSelection],
    mandatory: list[OrderForSelection],
    config: SelectorConfig,
) -> list[OrderForSelection]:
    """Start with mandatory, fill with geographically nearest orders.

    Args:
        orders: All available orders.
        mandatory: Mandatory orders (always included).
        config: Selector configuration.

    Returns:
        Selected orders based on geographic proximity.
    """
    import random

    if not mandatory:
        return greedy_by_priority(orders, [], config)

    # Check if mandatory orders are geographically close
    mandatory_with_coords = [
        o for o in mandatory if o.latitude != 0.0 and o.longitude != 0.0
    ]

    if len(mandatory_with_coords) < 2:
        # Not enough data for centroid, use first mandatory
        selected = list(mandatory)
    else:
        # Calculate centroid of mandatory orders
        centroid_lat = sum(o.latitude for o in mandatory_with_coords) / len(
            mandatory_with_coords
        )
        centroid_lng = sum(o.longitude for o in mandatory_with_coords) / len(
            mandatory_with_coords
        )

        # Check dispersion (simplified: max distance from centroid)
        max_distance = 0.0
        for o in mandatory_with_coords:
            dist = ((o.latitude - centroid_lat) ** 2 + (o.longitude - centroid_lng) ** 2) ** 0.5
            max_distance = max(max_distance, dist)

        # If too dispersed, start with one random mandatory
        if max_distance > 0.1:  # ~10km rough threshold
            selected = [random.choice(mandatory)]
            centroid_lat = selected[0].latitude
            centroid_lng = selected[0].longitude
        else:
            selected = list(mandatory)

    current_pallets = sum(o.total_pallets for o in selected)
    selected_ids = {o.order_id for o in selected}
    mandatory_ids = {o.order_id for o in mandatory}

    # Add remaining mandatory orders
    for o in mandatory:
        if o.order_id not in selected_ids:
            selected.append(o)
            current_pallets += o.total_pallets
            selected_ids.add(o.order_id)

    # Calculate centroid based on selected
    if selected:
        selected_with_coords = [
            o for o in selected if o.latitude != 0.0 and o.longitude != 0.0
        ]
        if selected_with_coords:
            centroid_lat = sum(o.latitude for o in selected_with_coords) / len(
                selected_with_coords
            )
            centroid_lng = sum(o.longitude for o in selected_with_coords) / len(
                selected_with_coords
            )
        else:
            centroid_lat, centroid_lng = -34.6037, -58.3816  # Buenos Aires default

    # Get available orders with distance to centroid
    available = [o for o in orders if o.order_id not in selected_ids]

    def distance_to_centroid(order: OrderForSelection) -> float:
        if order.latitude == 0.0 and order.longitude == 0.0:
            return float("inf")
        return (
            (order.latitude - centroid_lat) ** 2 + (order.longitude - centroid_lng) ** 2
        ) ** 0.5

    available.sort(key=distance_to_centroid)

    for order in available:
        if current_pallets + order.total_pallets <= config.max_acceptable:
            selected.append(order)
            current_pallets += order.total_pallets
        elif current_pallets >= config.min_acceptable:
            break
        elif current_pallets + order.total_pallets <= config.hard_max:
            selected.append(order)
            current_pallets += order.total_pallets

    return selected


# ============================================================================
# Candidate Building Functions
# ============================================================================


def build_dispatch_candidate(
    selected_orders: list[OrderForSelection],
    strategy: SelectionStrategy,
    config: SelectorConfig,
    is_subset: bool = False,
) -> Optional[DispatchCandidate]:
    """Build a DispatchCandidate from selected orders.

    Args:
        selected_orders: List of selected orders.
        strategy: The strategy used for selection.
        config: Selector configuration.
        is_subset: Whether this candidate was generated from a subset of orders.

    Returns:
        DispatchCandidate object or None if invalid.
    """
    if not selected_orders:
        return None

    total_pallets = sum(o.total_pallets for o in selected_orders)
    total_priority = sum(o.priority_score for o in selected_orders)
    zones = get_unique_zones(selected_orders)
    zone_breakdown = get_zone_breakdown(selected_orders)
    zone_penalty = get_zone_dispersion_penalty(zones, config)
    mandatory_orders = [o for o in selected_orders if o.is_mandatory]

    # Build order details for export
    order_details = [
        {
            "order_id": o.order_id,
            "client_name": o.client_name,
            "pallets": o.total_pallets,
            "priority_score": o.priority_score,
            "zone_id": o.zone_id,
            "is_mandatory": o.is_mandatory,
        }
        for o in selected_orders
    ]

    # Include subset indicator in candidate ID if applicable
    subset_tag = "-SUB" if is_subset else ""
    candidate_id = f"DISP-{datetime.now().strftime('%Y%m%d')}-{strategy.value[:6].upper()}{subset_tag}-{uuid.uuid4().hex[:4].upper()}"

    return DispatchCandidate(
        candidate_id=candidate_id,
        strategy=strategy,
        order_ids=[o.order_id for o in selected_orders],
        total_pallets=round(total_pallets, 2),
        total_priority=round(total_priority, 2),
        utilization_pct=round((total_pallets / config.nominal_capacity) * 100, 1),
        zones=zones,
        zone_breakdown=zone_breakdown,
        zone_dispersion_penalty=zone_penalty,
        adjusted_priority=round(total_priority * zone_penalty, 2),
        includes_mandatory=len(mandatory_orders) > 0,
        mandatory_count=len(mandatory_orders),
        is_single_zone=len(zones) == 1,
        is_subset=is_subset,
        orders=order_details,
    )


# ============================================================================
# Main Generation Functions
# ============================================================================


def generate_all_candidates(
    db: DatabaseManager,
    config_path: Optional[Path] = None,
) -> list[DispatchCandidate]:
    """Generate dispatch candidates using all enabled strategies.

    Args:
        db: DatabaseManager instance.
        config_path: Optional path to config file.

    Returns:
        List of dispatch candidates.
    """
    if config_path:
        config = load_selector_config(config_path)
    else:
        config = get_default_config()

    orders = load_pending_orders(db)
    all_mandatory = get_mandatory_orders(orders)

    # Handle mandatory overflow - select subset that fits
    mandatory_pallets = calculate_mandatory_pallets(all_mandatory)
    deferred_mandatory: list[OrderForSelection] = []

    if mandatory_pallets > config.hard_max:
        # Select subset of mandatory orders that fit
        mandatory, deferred_mandatory = select_mandatory_subset(
            all_mandatory, config.hard_max, strategy="priority"
        )
        print(
            f"⚠️ Mandatory overflow: {len(all_mandatory)} orders ({mandatory_pallets:.2f}p) "
            f"exceed capacity. Selected {len(mandatory)} for this dispatch, "
            f"{len(deferred_mandatory)} deferred."
        )
        # Remove deferred mandatory orders from orders list to avoid double-selection
        deferred_ids = {o.order_id for o in deferred_mandatory}
        orders = [o for o in orders if o.order_id not in deferred_ids]
    else:
        mandatory = all_mandatory

    candidates = []
    strategies = config.strategies_enabled

    # Map strategy names to functions and parameters
    strategy_map = {
        "greedy_efficiency": (
            greedy_by_efficiency,
            SelectionStrategy.GREEDY_EFFICIENCY,
            {},
        ),
        "greedy_priority": (
            greedy_by_priority,
            SelectionStrategy.GREEDY_PRIORITY,
            {},
        ),
        "greedy_zone_caba": (
            greedy_by_zone,
            SelectionStrategy.GREEDY_ZONE_CABA,
            {"target_zone": "CABA"},
        ),
        "greedy_zone_north": (
            greedy_by_zone,
            SelectionStrategy.GREEDY_ZONE_NORTH,
            {"target_zone": "NORTH_ZONE"},
        ),
        "greedy_zone_south": (
            greedy_by_zone,
            SelectionStrategy.GREEDY_ZONE_SOUTH,
            {"target_zone": "SOUTH_ZONE"},
        ),
        "greedy_zone_west": (
            greedy_by_zone,
            SelectionStrategy.GREEDY_ZONE_WEST,
            {"target_zone": "WEST_ZONE"},
        ),
        "greedy_zone_spillover": (
            greedy_zone_with_spillover,
            SelectionStrategy.GREEDY_ZONE_SPILLOVER,
            {},
        ),
        "greedy_best_fit": (
            greedy_best_fit,
            SelectionStrategy.GREEDY_BEST_FIT,
            {},
        ),
        "dp_optimal": (
            dp_optimal_knapsack,
            SelectionStrategy.DP_OPTIMAL,
            {},
        ),
        "mandatory_first": (
            mandatory_first,
            SelectionStrategy.MANDATORY_FIRST,
            {},
        ),
        "greedy_mandatory_nearest": (
            greedy_mandatory_nearest,
            SelectionStrategy.GREEDY_MANDATORY_NEAREST,
            {},
        ),
    }

    for strategy_name in strategies:
        if strategy_name not in strategy_map:
            continue

        func, strategy_enum, extra_kwargs = strategy_map[strategy_name]

        if "target_zone" in extra_kwargs:
            selected = func(orders, extra_kwargs["target_zone"], mandatory, config)
        else:
            selected = func(orders, mandatory, config)

        if selected:
            candidate = build_dispatch_candidate(selected, strategy_enum, config)
            if candidate:
                candidates.append(candidate)

    return candidates


def generate_all_candidates_with_info(
    db: DatabaseManager,
    config_path: Optional[Path] = None,
) -> tuple[list[DispatchCandidate], dict]:
    """Generate dispatch candidates with additional info about mandatory overflow.

    Args:
        db: DatabaseManager instance.
        config_path: Optional path to config file.

    Returns:
        Tuple of (candidates, info_dict) where info_dict contains:
        - deferred_mandatory: List of mandatory orders not included in this dispatch
        - overflow_info: Dictionary with overflow statistics
    """
    if config_path:
        config = load_selector_config(config_path)
    else:
        config = get_default_config()

    orders = load_pending_orders(db)
    all_mandatory = get_mandatory_orders(orders)
    mandatory_pallets = calculate_mandatory_pallets(all_mandatory)

    overflow_info = get_mandatory_overflow_info(all_mandatory, config.hard_max)
    deferred_mandatory: list[OrderForSelection] = []

    if mandatory_pallets > config.hard_max:
        mandatory, deferred_mandatory = select_mandatory_subset(
            all_mandatory, config.hard_max, strategy="priority"
        )
        # Remove deferred mandatory orders from orders list to avoid double-selection
        deferred_ids = {o.order_id for o in deferred_mandatory}
        orders = [o for o in orders if o.order_id not in deferred_ids]
    else:
        mandatory = all_mandatory

    candidates = []
    strategies = config.strategies_enabled

    strategy_map = {
        "greedy_efficiency": (greedy_by_efficiency, SelectionStrategy.GREEDY_EFFICIENCY, {}),
        "greedy_priority": (greedy_by_priority, SelectionStrategy.GREEDY_PRIORITY, {}),
        "greedy_zone_caba": (greedy_by_zone, SelectionStrategy.GREEDY_ZONE_CABA, {"target_zone": "CABA"}),
        "greedy_zone_north": (greedy_by_zone, SelectionStrategy.GREEDY_ZONE_NORTH, {"target_zone": "NORTH_ZONE"}),
        "greedy_zone_south": (greedy_by_zone, SelectionStrategy.GREEDY_ZONE_SOUTH, {"target_zone": "SOUTH_ZONE"}),
        "greedy_zone_west": (greedy_by_zone, SelectionStrategy.GREEDY_ZONE_WEST, {"target_zone": "WEST_ZONE"}),
        "greedy_zone_spillover": (greedy_zone_with_spillover, SelectionStrategy.GREEDY_ZONE_SPILLOVER, {}),
        "greedy_best_fit": (greedy_best_fit, SelectionStrategy.GREEDY_BEST_FIT, {}),
        "dp_optimal": (dp_optimal_knapsack, SelectionStrategy.DP_OPTIMAL, {}),
        "mandatory_first": (mandatory_first, SelectionStrategy.MANDATORY_FIRST, {}),
        "greedy_mandatory_nearest": (greedy_mandatory_nearest, SelectionStrategy.GREEDY_MANDATORY_NEAREST, {}),
    }

    for strategy_name in strategies:
        if strategy_name not in strategy_map:
            continue

        func, strategy_enum, extra_kwargs = strategy_map[strategy_name]

        if "target_zone" in extra_kwargs:
            selected = func(orders, extra_kwargs["target_zone"], mandatory, config)
        else:
            selected = func(orders, mandatory, config)

        if selected:
            candidate = build_dispatch_candidate(selected, strategy_enum, config)
            if candidate:
                candidates.append(candidate)

    info = {
        "deferred_mandatory": deferred_mandatory,
        "overflow_info": overflow_info,
        "included_mandatory_count": len(mandatory),
        "deferred_mandatory_count": len(deferred_mandatory),
    }

    return candidates, info


def deduplicate_candidates(
    candidates: list[DispatchCandidate],
) -> list[DispatchCandidate]:
    """Remove duplicate candidates (same order set).

    Args:
        candidates: List of candidates to deduplicate.

    Returns:
        List of unique candidates.
    """
    seen: set[frozenset[str]] = set()
    unique = []

    for candidate in candidates:
        key = frozenset(candidate.order_ids)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)

    return unique


def rank_candidates(
    candidates: list[DispatchCandidate],
    config: SelectorConfig,
) -> list[DispatchCandidate]:
    """Rank candidates by combined score.

    Args:
        candidates: List of candidates to rank.
        config: Selector configuration.

    Returns:
        Sorted list of candidates (best first).
    """
    if not candidates:
        return []

    # Normalize values for scoring
    max_priority = max(c.adjusted_priority for c in candidates)
    max_utilization = max(c.utilization_pct for c in candidates)

    def combined_score(candidate: DispatchCandidate) -> float:
        norm_priority = candidate.adjusted_priority / max_priority if max_priority > 0 else 0
        norm_utilization = candidate.utilization_pct / max_utilization if max_utilization > 0 else 0
        zone_coherence = 1.0 if candidate.is_single_zone else 0.5

        return (
            config.ranking_weight_priority * norm_priority
            + config.ranking_weight_utilization * norm_utilization
            + config.ranking_weight_zone_coherence * zone_coherence
        )

    return sorted(candidates, key=combined_score, reverse=True)


def get_best_single_zone_candidate(
    candidates: list[DispatchCandidate],
) -> Optional[DispatchCandidate]:
    """Return highest priority candidate that is single-zone.

    Args:
        candidates: List of candidates.

    Returns:
        Best single-zone candidate or None.
    """
    single_zone = [c for c in candidates if c.is_single_zone]
    if not single_zone:
        return None
    return max(single_zone, key=lambda c: c.adjusted_priority)


def get_exceptional_multizone_candidates(
    candidates: list[DispatchCandidate],
    best_single_zone_priority: float,
    threshold: float = 0.30,
) -> list[DispatchCandidate]:
    """Return multi-zone candidates that exceed single-zone by threshold.

    Args:
        candidates: List of candidates.
        best_single_zone_priority: Priority of best single-zone candidate.
        threshold: Percentage threshold for exception.

    Returns:
        List of exceptional multi-zone candidates.
    """
    min_priority = best_single_zone_priority * (1 + threshold)
    multizone = [c for c in candidates if not c.is_single_zone]
    return [c for c in multizone if c.total_priority >= min_priority]


def get_top_n_candidates(
    candidates: list[DispatchCandidate],
    n: int = 5,
) -> list[DispatchCandidate]:
    """Return top N candidates after ranking.

    Args:
        candidates: List of candidates (should be ranked).
        n: Number of candidates to return.

    Returns:
        Top N candidates.
    """
    return candidates[:n]


# ============================================================================
# Export Functions
# ============================================================================


def export_candidate_to_dict(candidate: DispatchCandidate) -> dict:
    """Convert a candidate to a dictionary for JSON export.

    Args:
        candidate: DispatchCandidate to export.

    Returns:
        Dictionary representation.
    """
    return {
        "candidate_id": candidate.candidate_id,
        "strategy": candidate.strategy.value,
        "generated_at": datetime.now().isoformat(),
        "orders": candidate.orders,
        "summary": {
            "total_pallets": candidate.total_pallets,
            "total_priority": candidate.total_priority,
            "utilization_pct": candidate.utilization_pct,
            "order_count": len(candidate.order_ids),
            "zones": candidate.zones,
            "zone_breakdown": candidate.zone_breakdown,
            "is_single_zone": candidate.is_single_zone,
            "zone_dispersion_penalty": candidate.zone_dispersion_penalty,
            "adjusted_priority": candidate.adjusted_priority,
            "mandatory_included": candidate.includes_mandatory,
            "mandatory_count": candidate.mandatory_count,
        },
    }


def export_candidates_to_json(
    candidates: list[DispatchCandidate],
    output_path: Path,
) -> None:
    """Export candidates to a JSON file.

    Args:
        candidates: List of candidates to export.
        output_path: Path to output JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_data = {
        "generated_at": datetime.now().isoformat(),
        "candidate_count": len(candidates),
        "candidates": [export_candidate_to_dict(c) for c in candidates],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)


def get_candidates_summary_df(candidates: list[DispatchCandidate], include_order_ids: bool = False):
    """Create a pandas DataFrame summary of candidates.

    Args:
        candidates: List of candidates.
        include_order_ids: Whether to include a column with order IDs list.

    Returns:
        pandas DataFrame with candidate summaries.
    """
    import pandas as pd

    data = []
    for c in candidates:
        # Show strategy with (subset) suffix if applicable
        strategy_display = c.strategy.value
        if c.is_subset:
            strategy_display = f"{strategy_display} (subset)"
        
        row = {
            "candidate_id": c.candidate_id,
            "strategy": strategy_display,
            "total_pallets": c.total_pallets,
            "adjusted_priority": c.adjusted_priority,
            "total_priority": c.total_priority,
            "utilization_pct": c.utilization_pct,
            "zones": ", ".join(c.zones),
            "zone_count": len(c.zones),
            "is_single_zone": c.is_single_zone,
            "order_count": len(c.order_ids),
            "mandatory_count": c.mandatory_count,
            "is_subset": c.is_subset,
        }
        if include_order_ids:
            row["order_ids"] = ", ".join(c.order_ids)
        data.append(row)

    return pd.DataFrame(data).sort_values(by="adjusted_priority", ascending=False)




def generate_candidates_no_mandatory(
    db: DatabaseManager,
    config_path: Optional[Path] = None,
) -> list[DispatchCandidate]:
    """Generate dispatch candidates without including mandatory orders.

    This is useful for comparison to see dispatch quality without the
    999999 priority boost from mandatory orders.

    Args:
        db: DatabaseManager instance.
        config_path: Optional path to config file.

    Returns:
        List of dispatch candidates (without mandatory orders).
    """
    if config_path:
        config = load_selector_config(config_path)
    else:
        config = get_default_config()

    orders = load_pending_orders(db)
    non_mandatory_orders = get_non_mandatory_orders(orders)

    # Empty mandatory list - no mandatory orders to include
    empty_mandatory: list[OrderForSelection] = []

    candidates = []

    # Only use strategies that make sense without mandatory orders
    strategies_for_non_mandatory = [
        ("greedy_efficiency", greedy_by_efficiency, SelectionStrategy.GREEDY_EFFICIENCY, {}),
        ("greedy_priority", greedy_by_priority, SelectionStrategy.GREEDY_PRIORITY, {}),
        ("greedy_zone_caba", greedy_by_zone, SelectionStrategy.GREEDY_ZONE_CABA, {"target_zone": "CABA"}),
        ("greedy_zone_north", greedy_by_zone, SelectionStrategy.GREEDY_ZONE_NORTH, {"target_zone": "NORTH_ZONE"}),
        ("greedy_zone_south", greedy_by_zone, SelectionStrategy.GREEDY_ZONE_SOUTH, {"target_zone": "SOUTH_ZONE"}),
        ("greedy_zone_west", greedy_by_zone, SelectionStrategy.GREEDY_ZONE_WEST, {"target_zone": "WEST_ZONE"}),
        ("greedy_zone_spillover", greedy_zone_with_spillover, SelectionStrategy.GREEDY_ZONE_SPILLOVER, {}),
        ("greedy_best_fit", greedy_best_fit, SelectionStrategy.GREEDY_BEST_FIT, {}),
        ("dp_optimal", dp_optimal_knapsack, SelectionStrategy.DP_OPTIMAL, {}),
    ]

    for strategy_name, func, strategy_enum, extra_kwargs in strategies_for_non_mandatory:
        if strategy_name not in config.strategies_enabled:
            continue

        if "target_zone" in extra_kwargs:
            selected = func(non_mandatory_orders, extra_kwargs["target_zone"], empty_mandatory, config)
        else:
            selected = func(non_mandatory_orders, empty_mandatory, config)

        if selected:
            candidate = build_dispatch_candidate(selected, strategy_enum, config)
            if candidate:
                candidates.append(candidate)

    return candidates
