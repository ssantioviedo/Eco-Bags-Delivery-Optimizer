"""Route optimization module for the Delivery Optimization System.

This module provides functions for solving TSP/VRP problems to optimize
delivery routes. It supports both simple nearest-neighbor heuristics and
OR-Tools based optimization with time window constraints.

The route cache is persistent using SQLite to avoid recalculating routes
for the same set of orders across kernel restarts.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import time, datetime, timedelta
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import folium
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

if TYPE_CHECKING:
    from src.database import DatabaseManager


# ============================================================================
# Route Optimization Cache
# ============================================================================

# Reference to database manager for persistent cache
_cache_db: Optional["DatabaseManager"] = None


def _generate_cache_key(order_ids: list[str], solver: str) -> str:
    """Generate a unique cache key for a dispatch based on order IDs and solver.

    Args:
        order_ids: List of order IDs in the dispatch.
        solver: The solver being used.

    Returns:
        Hash string as cache key.
    """
    # Sort order IDs to ensure consistent key regardless of order
    sorted_ids = sorted(order_ids)
    key_string = f"{solver}:{'|'.join(sorted_ids)}"
    return hashlib.md5(key_string.encode()).hexdigest()


def init_route_cache(db: "DatabaseManager") -> None:
    """Initialize the route cache with a database manager.
    
    This must be called before using cache features. The cache is
    persistent in SQLite and survives kernel restarts.
    
    Args:
        db: DatabaseManager instance for persistent storage.
    """
    global _cache_db
    _cache_db = db
    # Ensure the route_cache table exists
    db.create_tables()

def minmax_scale(arr: np.ndarray) -> np.ndarray:
    """Escala valores a rango [0, 1]."""
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

def clear_route_cache() -> int:
    """Clear the route optimization cache.
    
    Returns:
        Number of entries cleared, or 0 if cache not initialized.
    """
    global _cache_db
    if _cache_db is not None:
        return _cache_db.clear_route_cache()
    return 0


def get_cache_stats() -> dict:
    """Get statistics about the route cache.

    Returns:
        Dictionary with cache size and solver breakdown.
    """
    global _cache_db
    if _cache_db is not None:
        return _cache_db.get_route_cache_stats()
    return {"size": 0, "by_solver": {}}


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class Location:
    """A geographic location for routing."""

    id: str
    name: str
    latitude: float
    longitude: float
    time_window_start: Optional[time] = None
    time_window_end: Optional[time] = None


@dataclass
class RouteStop:
    """A single stop in the optimized route."""

    sequence: int
    location: Location
    arrival_time: Optional[time]
    departure_time: Optional[time]
    distance_from_previous_km: float
    cumulative_distance_km: float
    wait_time_minutes: int = 0


@dataclass
class RouteResult:
    """Complete route solution."""

    dispatch_id: str
    stops: list[RouteStop]
    total_distance_km: float
    total_duration_minutes: int
    route_start_time: time
    route_end_time: time
    feasible: bool
    solver_used: str


@dataclass
class DispatchWithRoute:
    """Dispatch candidate enriched with route information."""

    candidate_id: str
    strategy: str
    order_ids: list[str]
    order_count: int
    total_pallets: float
    total_priority: float
    zones: list[str]
    is_single_zone: bool
    has_mandatory: bool
    mandatory_count: int
    route: RouteResult
    total_distance_km: float
    total_duration_minutes: int
    orders: list[dict] = field(default_factory=list)


@dataclass
class RoutingConfig:
    """Configuration for route optimization."""

    depot: Location
    average_speed_kmh: float
    service_time_minutes: int
    default_start_time: time
    max_route_duration_minutes: int
    preferred_solver: str
    fallback_solver: str
    ortools_time_limit_seconds: int
    visualization: dict


# ============================================================================
# Route Serialization for Cache
# ============================================================================


def _serialize_route_result(route: RouteResult) -> dict:
    """Serialize a RouteResult to a dictionary for caching.
    
    Args:
        route: RouteResult to serialize.
        
    Returns:
        Dictionary representation suitable for JSON storage.
    """
    def time_to_str(t: Optional[time]) -> Optional[str]:
        return t.isoformat() if t else None
    
    return {
        "dispatch_id": route.dispatch_id,
        "total_distance_km": route.total_distance_km,
        "total_duration_minutes": route.total_duration_minutes,
        "route_start_time": time_to_str(route.route_start_time),
        "route_end_time": time_to_str(route.route_end_time),
        "feasible": route.feasible,
        "solver_used": route.solver_used,
        "stops": [
            {
                "sequence": stop.sequence,
                "location": {
                    "id": stop.location.id,
                    "name": stop.location.name,
                    "latitude": stop.location.latitude,
                    "longitude": stop.location.longitude,
                    "time_window_start": time_to_str(stop.location.time_window_start),
                    "time_window_end": time_to_str(stop.location.time_window_end),
                },
                "arrival_time": time_to_str(stop.arrival_time),
                "departure_time": time_to_str(stop.departure_time),
                "distance_from_previous_km": stop.distance_from_previous_km,
                "cumulative_distance_km": stop.cumulative_distance_km,
                "wait_time_minutes": stop.wait_time_minutes,
            }
            for stop in route.stops
        ],
    }


def _deserialize_route_result(data: dict) -> RouteResult:
    """Deserialize a dictionary to a RouteResult.
    
    Args:
        data: Dictionary from cache storage.
        
    Returns:
        RouteResult object.
    """
    def str_to_time(s: Optional[str]) -> Optional[time]:
        if s is None:
            return None
        return time.fromisoformat(s)
    
    stops = []
    for stop_data in data["stops"]:
        loc_data = stop_data["location"]
        location = Location(
            id=loc_data["id"],
            name=loc_data["name"],
            latitude=loc_data["latitude"],
            longitude=loc_data["longitude"],
            time_window_start=str_to_time(loc_data.get("time_window_start")),
            time_window_end=str_to_time(loc_data.get("time_window_end")),
        )
        stop = RouteStop(
            sequence=stop_data["sequence"],
            location=location,
            arrival_time=str_to_time(stop_data["arrival_time"]),
            departure_time=str_to_time(stop_data["departure_time"]),
            distance_from_previous_km=stop_data["distance_from_previous_km"],
            cumulative_distance_km=stop_data["cumulative_distance_km"],
            wait_time_minutes=stop_data.get("wait_time_minutes", 0),
        )
        stops.append(stop)
    
    return RouteResult(
        dispatch_id=data["dispatch_id"],
        stops=stops,
        total_distance_km=data["total_distance_km"],
        total_duration_minutes=data["total_duration_minutes"],
        route_start_time=str_to_time(data["route_start_time"]),
        route_end_time=str_to_time(data["route_end_time"]),
        feasible=data["feasible"],
        solver_used=data["solver_used"],
    )


# ============================================================================
# Configuration Loading
# ============================================================================


def load_routing_config(config_path: Path) -> RoutingConfig:
    """Load routing configuration from JSON file.

    Args:
        config_path: Path to the routing_config.json file.

    Returns:
        RoutingConfig object with all settings.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    depot_data = data["depot"]
    depot = Location(
        id=depot_data["id"],
        name=depot_data["name"],
        latitude=depot_data["latitude"],
        longitude=depot_data["longitude"],
    )

    assumptions = data["assumptions"]
    start_time_parts = assumptions["default_start_time"].split(":")
    default_start = time(int(start_time_parts[0]), int(start_time_parts[1]))

    solver = data["solver"]

    return RoutingConfig(
        depot=depot,
        average_speed_kmh=assumptions["average_speed_kmh"],
        service_time_minutes=assumptions["service_time_minutes"],
        default_start_time=default_start,
        max_route_duration_minutes=assumptions["max_route_duration_minutes"],
        preferred_solver=solver["preferred"],
        fallback_solver=solver["fallback"],
        ortools_time_limit_seconds=solver["ortools_time_limit_seconds"],
        visualization=data.get("visualization", {}),
    )


# ============================================================================
# Distance Calculations
# ============================================================================


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculate the great-circle distance between two points in kilometers.

    Uses the Haversine formula to calculate straight-line distance.

    Args:
        lat1: Latitude of first point in degrees.
        lon1: Longitude of first point in degrees.
        lat2: Latitude of second point in degrees.
        lon2: Longitude of second point in degrees.

    Returns:
        Distance in kilometers.
    """
    R = 6371  # Earth's radius in kilometers

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def build_distance_matrix(locations: list[Location]) -> list[list[float]]:
    """Build NxN distance matrix for all locations.

    Args:
        locations: List of locations where index 0 is the depot.

    Returns:
        2D list of distances in kilometers.
    """
    n = len(locations)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = haversine_distance(
                    locations[i].latitude,
                    locations[i].longitude,
                    locations[j].latitude,
                    locations[j].longitude,
                )

    return matrix


def estimate_travel_time(distance_km: float, avg_speed_kmh: float = 30) -> int:
    """Estimate travel time in minutes given distance and average speed.

    Args:
        distance_km: Distance in kilometers.
        avg_speed_kmh: Average speed in km/h (default 30 for urban delivery).

    Returns:
        Estimated time in minutes.
    """
    if distance_km <= 0:
        return 0
    hours = distance_km / avg_speed_kmh
    return int(math.ceil(hours * 60))


# ============================================================================
# Nearest Neighbor Heuristic
# ============================================================================


def solve_tsp_nearest_neighbor(
    locations: list[Location], depot_index: int = 0
) -> list[int]:
    """Solve TSP using nearest neighbor heuristic.

    Starting from the depot, always visit the nearest unvisited location.

    Args:
        locations: List of all locations (depot + delivery points).
        depot_index: Index of the depot in the locations list.

    Returns:
        List of indices representing visit order (excludes return to depot).
    """
    n = len(locations)
    if n <= 1:
        return [depot_index]

    distance_matrix = build_distance_matrix(locations)
    visited = [False] * n
    route = [depot_index]
    visited[depot_index] = True

    current = depot_index
    for _ in range(n - 1):
        nearest = -1
        nearest_dist = float("inf")

        for j in range(n):
            if not visited[j] and distance_matrix[current][j] < nearest_dist:
                nearest = j
                nearest_dist = distance_matrix[current][j]

        if nearest != -1:
            route.append(nearest)
            visited[nearest] = True
            current = nearest

    return route


# ============================================================================
# OR-Tools TSP Solver
# ============================================================================


def solve_tsp_ortools(
    locations: list[Location],
    time_limit_seconds: int = 30,
) -> list[int]:
    """Solve TSP using Google OR-Tools.

    Args:
        locations: List of all locations (depot at index 0).
        time_limit_seconds: Maximum solver time.

    Returns:
        List of indices representing optimal visit order.
    """
    n = len(locations)
    if n <= 2:
        return list(range(n))

    # Build distance matrix (scaled to integers for OR-Tools)
    distance_matrix = build_distance_matrix(locations)
    # Scale to meters for integer precision
    int_matrix = [[int(d * 1000) for d in row] for row in distance_matrix]

    # Create routing model
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # n nodes, 1 vehicle, depot=0
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = time_limit_seconds

    # Solve
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return route
    else:
        # Fallback to nearest neighbor if OR-Tools fails
        return solve_tsp_nearest_neighbor(locations)


# ============================================================================
# OR-Tools VRPTW Solver (with Time Windows)
# ============================================================================


def time_to_minutes(t: time) -> int:
    """Convert time object to minutes since midnight."""
    return t.hour * 60 + t.minute


def minutes_to_time(minutes: int) -> time:
    """Convert minutes since midnight to time object."""
    hours = minutes // 60
    mins = minutes % 60
    return time(hour=min(hours, 23), minute=mins)


def solve_vrptw_ortools(
    locations: list[Location],
    start_time: time = time(8, 0),
    service_time_minutes: int = 15,
    avg_speed_kmh: float = 30,
    time_limit_seconds: int = 30,
) -> tuple[list[int], bool]:
    """Solve VRP with Time Windows using OR-Tools.

    Args:
        locations: List of locations (depot at index 0).
        start_time: Route start time.
        service_time_minutes: Time spent at each stop.
        avg_speed_kmh: Average travel speed.
        time_limit_seconds: Maximum solver time.

    Returns:
        Tuple of (visit order as list of indices, feasibility boolean).
    """
    n = len(locations)
    if n <= 2:
        return list(range(n)), True

    # Build matrices
    distance_matrix = build_distance_matrix(locations)
    int_distance = [[int(d * 1000) for d in row] for row in distance_matrix]

    # Time matrix (in minutes)
    time_matrix = [
        [estimate_travel_time(d, avg_speed_kmh) for d in row]
        for row in distance_matrix
    ]

    # Create routing model
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int_distance[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Time callback (travel time + service time)
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel = time_matrix[from_node][to_node]
        # Add service time at destination (except depot)
        service = service_time_minutes if to_node != 0 else 0
        return travel + service

    time_callback_index = routing.RegisterTransitCallback(time_callback)

    # Add time dimension
    start_minutes = time_to_minutes(start_time)
    max_time = 24 * 60  # Max 24 hours

    routing.AddDimension(
        time_callback_index,
        60,  # Allow waiting time up to 60 minutes
        max_time,
        False,  # Don't force start cumul to zero
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    # Set time windows
    for i in range(n):
        index = manager.NodeToIndex(i)
        loc = locations[i]

        if i == 0:
            # Depot: route starts at start_time
            time_dimension.CumulVar(index).SetRange(start_minutes, max_time)
        elif loc.time_window_start and loc.time_window_end:
            # Customer with time window
            tw_start = time_to_minutes(loc.time_window_start)
            tw_end = time_to_minutes(loc.time_window_end)
            time_dimension.CumulVar(index).SetRange(tw_start, tw_end)
        else:
            # No time window - can visit anytime
            time_dimension.CumulVar(index).SetRange(start_minutes, max_time)

    # Minimize time
    for i in range(manager.GetNumberOfVehicles()):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i))
        )
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i))
        )

    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = time_limit_seconds

    # Solve
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return route, True
    else:
        # Infeasible - fall back to TSP without time windows
        return solve_tsp_ortools(locations, time_limit_seconds), False


# ============================================================================
# Route Building
# ============================================================================


def build_route_result(
    dispatch_id: str,
    locations: list[Location],
    visit_order: list[int],
    start_time: time,
    service_time_minutes: int,
    avg_speed_kmh: float,
    solver_used: str,
    feasible: bool = True,
) -> RouteResult:
    """Build complete RouteResult from visit order.

    Args:
        dispatch_id: Identifier for the dispatch.
        locations: All locations (depot at index 0).
        visit_order: Order of visits as location indices.
        start_time: Route start time.
        service_time_minutes: Time at each stop.
        avg_speed_kmh: Average travel speed.
        solver_used: Name of solver used.
        feasible: Whether time windows are satisfied.

    Returns:
        Complete RouteResult with all stop details.
    """
    stops = []
    cumulative_distance = 0.0
    current_time_minutes = time_to_minutes(start_time)

    for seq, loc_idx in enumerate(visit_order):
        loc = locations[loc_idx]

        if seq == 0:
            # Depot - starting point
            stop = RouteStop(
                sequence=seq,
                location=loc,
                arrival_time=start_time,
                departure_time=start_time,
                distance_from_previous_km=0.0,
                cumulative_distance_km=0.0,
                wait_time_minutes=0,
            )
        else:
            # Calculate distance from previous
            prev_loc = locations[visit_order[seq - 1]]
            dist = haversine_distance(
                prev_loc.latitude, prev_loc.longitude,
                loc.latitude, loc.longitude
            )
            cumulative_distance += dist

            # Calculate arrival time
            travel_minutes = estimate_travel_time(dist, avg_speed_kmh)
            arrival_minutes = current_time_minutes + travel_minutes

            # Check for wait time if time window exists
            wait_time = 0
            if loc.time_window_start:
                tw_start = time_to_minutes(loc.time_window_start)
                if arrival_minutes < tw_start:
                    wait_time = tw_start - arrival_minutes
                    arrival_minutes = tw_start

            departure_minutes = arrival_minutes + service_time_minutes
            current_time_minutes = departure_minutes

            stop = RouteStop(
                sequence=seq,
                location=loc,
                arrival_time=minutes_to_time(arrival_minutes),
                departure_time=minutes_to_time(departure_minutes),
                distance_from_previous_km=round(dist, 2),
                cumulative_distance_km=round(cumulative_distance, 2),
                wait_time_minutes=wait_time,
            )

        stops.append(stop)

    # Calculate return to depot
    if len(visit_order) > 1:
        last_loc = locations[visit_order[-1]]
        depot = locations[0]
        return_dist = haversine_distance(
            last_loc.latitude, last_loc.longitude,
            depot.latitude, depot.longitude
        )
        cumulative_distance += return_dist
        return_travel = estimate_travel_time(return_dist, avg_speed_kmh)
        end_minutes = current_time_minutes + return_travel
    else:
        end_minutes = current_time_minutes

    total_duration = end_minutes - time_to_minutes(start_time)

    return RouteResult(
        dispatch_id=dispatch_id,
        stops=stops,
        total_distance_km=round(cumulative_distance, 2),
        total_duration_minutes=total_duration,
        route_start_time=start_time,
        route_end_time=minutes_to_time(end_minutes),
        feasible=feasible,
        solver_used=solver_used
    )


# ============================================================================
# Main Optimization Functions
# ============================================================================


def optimize_dispatch_route(
    dispatch_candidate: dict,
    config: RoutingConfig,
    db,
    use_cache: bool = True,
) -> DispatchWithRoute:
    """Optimize route for a single dispatch candidate.

    Args:
        dispatch_candidate: Dispatch data dictionary.
        config: Routing configuration.
        db: Database manager for fetching order details.
        use_cache: Whether to use cached results if available.

    Returns:
        DispatchWithRoute with optimized route.
    """
    global _cache_db
    
    # Initialize cache with this db if not already set
    if _cache_db is None:
        init_route_cache(db)

    # Extract order IDs
    order_ids = dispatch_candidate.get("order_ids", [])
    if not order_ids and "orders" in dispatch_candidate:
        order_ids = [o["order_id"] for o in dispatch_candidate["orders"]]

    # Check persistent cache
    cache_key = _generate_cache_key(order_ids, config.preferred_solver)
    if use_cache and _cache_db is not None:
        cached_data = _cache_db.get_cached_route(cache_key)
        if cached_data:
            # Deserialize route from cache
            cached_route = _deserialize_route_result(cached_data["route_data"])
            # Return cached result with candidate-specific fields
            return DispatchWithRoute(
                candidate_id=dispatch_candidate.get("candidate_id", ""),
                strategy=dispatch_candidate.get("strategy", ""),
                order_ids=order_ids,
                order_count=len(order_ids),
                total_pallets=dispatch_candidate.get("summary", {}).get("total_pallets", 0),
                total_priority=dispatch_candidate.get("summary", {}).get("total_priority", 0),
                zones=dispatch_candidate.get("summary", {}).get("zones", []),
                is_single_zone=dispatch_candidate.get("summary", {}).get("is_single_zone", False),
                has_mandatory=dispatch_candidate.get("summary", {}).get("mandatory_count", 0) > 0,
                mandatory_count=dispatch_candidate.get("summary", {}).get("mandatory_count", 0),
                route=cached_route,
                total_distance_km=cached_data["total_distance_km"],
                total_duration_minutes=cached_data["total_duration_minutes"],
                orders=dispatch_candidate.get("orders", []),
            )

    # Get order coordinates from database
    coords = db.get_order_coordinates(order_ids)

    # Build locations list (depot first)
    locations = [config.depot]

    orders_data = dispatch_candidate.get("orders", [])
    order_map = {o["order_id"]: o for o in orders_data}

    for order_id in order_ids:
        if order_id in coords:
            lat, lon = coords[order_id]
            order_info = order_map.get(order_id, {})

            # Parse time windows if present
            tw_start = None
            tw_end = None
            if "time_window_start" in order_info and order_info["time_window_start"]:
                try:
                    tw_start = time.fromisoformat(order_info["time_window_start"])
                except (ValueError, TypeError):
                    pass
            if "time_window_end" in order_info and order_info["time_window_end"]:
                try:
                    tw_end = time.fromisoformat(order_info["time_window_end"])
                except (ValueError, TypeError):
                    pass

            loc = Location(
                id=order_id,
                name=order_info.get("client_name", order_id),
                latitude=lat,
                longitude=lon,
                time_window_start=tw_start,
                time_window_end=tw_end,
            )
            locations.append(loc)

    # Check if any location has time windows
    has_time_windows = any(
        loc.time_window_start is not None for loc in locations[1:]
    )

    # Solve route
    if config.preferred_solver == "ortools":
        if has_time_windows:
            visit_order, feasible = solve_vrptw_ortools(
                locations,
                start_time=config.default_start_time,
                service_time_minutes=config.service_time_minutes,
                avg_speed_kmh=config.average_speed_kmh,
                time_limit_seconds=config.ortools_time_limit_seconds,
            )
            solver_used = "ortools_vrptw"
        else:
            visit_order = solve_tsp_ortools(
                locations,
                time_limit_seconds=config.ortools_time_limit_seconds,
            )
            feasible = True
            solver_used = "ortools_tsp"
    else:
        visit_order = solve_tsp_nearest_neighbor(locations)
        feasible = True
        solver_used = "nearest_neighbor"

    # Build route result
    route = build_route_result(
        dispatch_id=dispatch_candidate.get("candidate_id", "unknown"),
        locations=locations,
        visit_order=visit_order,
        start_time=config.default_start_time,
        service_time_minutes=config.service_time_minutes,
        avg_speed_kmh=config.average_speed_kmh,
        solver_used=solver_used,
        feasible=feasible,
    )

    # Build enriched dispatch
    result = DispatchWithRoute(
        candidate_id=dispatch_candidate.get("candidate_id", ""),
        strategy=dispatch_candidate.get("strategy", ""),
        order_ids=order_ids,
        order_count=len(order_ids),
        total_pallets=dispatch_candidate.get("summary", {}).get("total_pallets", 0),
        total_priority=dispatch_candidate.get("summary", {}).get("total_priority", 0),
        zones=dispatch_candidate.get("summary", {}).get("zones", []),
        is_single_zone=dispatch_candidate.get("summary", {}).get("is_single_zone", False),
        has_mandatory=dispatch_candidate.get("summary", {}).get("mandatory_count", 0) > 0,
        mandatory_count=dispatch_candidate.get("summary", {}).get("mandatory_count", 0),
        route=route,
        total_distance_km=route.total_distance_km,
        total_duration_minutes=route.total_duration_minutes,
        orders=orders_data,
    )

    # Store in persistent cache
    if use_cache and _cache_db is not None:
        _cache_db.save_route_to_cache(
            cache_key=cache_key,
            solver=solver_used,
            order_ids=order_ids,
            total_distance_km=route.total_distance_km,
            total_duration_minutes=route.total_duration_minutes,
            route_data=_serialize_route_result(route),
            feasible=route.feasible,
        )

    return result


def optimize_all_dispatches(
    candidates_json_path: Path,
    db,
    config: RoutingConfig,
) -> list[DispatchWithRoute]:
    """Optimize routes for all dispatch candidates.

    Args:
        candidates_json_path: Path to dispatch_candidates.json.
        db: Database manager.
        config: Routing configuration.

    Returns:
        List of DispatchWithRoute objects.
    """
    with open(candidates_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = data.get("candidates", [])
    results = []

    for candidate in candidates:
        try:
            result = optimize_dispatch_route(candidate, config, db)
            results.append(result)
        except Exception as e:
            print(f"Error optimizing {candidate.get('candidate_id')}: {e}")

    return results


def rank_dispatches_with_routes(
    dispatches: list[DispatchWithRoute],
    priority_weight: float = 0.5,
    distance_weight: float = 0.4,
    utilization_weight: float = 0.1,
) -> list[DispatchWithRoute]:
    """Rank dispatches by combined score."""
    if not dispatches:
        return []

    if len(dispatches) == 1:
        return dispatches

    # Extract raw values
    priorities = np.array([d.total_priority for d in dispatches])
    distances = np.array([d.total_distance_km for d in dispatches])
    utilizations = np.array([min(d.total_pallets / 8.0, 1.0) for d in dispatches])

    # Log transform para manejar outliers en prioridades
    priorities_log = np.log1p(priorities)

    # MinMax scaling manual
    priority_scores = minmax_scale(priorities_log)
    distance_scores = 1 - minmax_scale(distances)
    utilization_scores = minmax_scale(utilizations)

    # Combined scores
    scored = []
    for i, d in enumerate(dispatches):
        combined = (
            priority_weight * priority_scores[i]
            + distance_weight * distance_scores[i]
            + utilization_weight * utilization_scores[i]
        )
        scored.append((combined, d))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [d for _, d in scored]

# ============================================================================
# Folium Visualization
# ============================================================================


def create_route_map(
    dispatch: DispatchWithRoute,
    depot: Location,
    config: RoutingConfig,
) -> folium.Map:
    """Create interactive Folium map showing the optimized route.

    Args:
        route_result: The optimized route.
        depot: Depot location.
        config: Routing configuration for styling.

    Returns:
        Folium Map object.
    """
    viz = config.visualization

    route_result = dispatch.route

    # Calculate center point
    all_lats = [s.location.latitude for s in route_result.stops]
    all_lons = [s.location.longitude for s in route_result.stops]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)

    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=viz.get("zoom_start", 11),
        tiles="cartodbpositron"
    )

    # Add depot marker
    folium.Marker(
        location=[depot.latitude, depot.longitude],
        popup="<b>Eco-Bags Factory</b><br>Depot Location",
        tooltip="Factory Depot",
        icon=folium.Icon(color="black", icon="industry", prefix="fa")
    ).add_to(m)

    # Add stop markers (skip depot at index 0)
    for stop in route_result.stops[1:]:
        popup_html = f"""
        <b>Stop #{stop.sequence}: {stop.location.name}</b><br>
        Order: {stop.location.id}<br>
        Arrival: {stop.arrival_time.strftime('%H:%M') if stop.arrival_time else 'N/A'}<br>
        Distance: {stop.cumulative_distance_km} km
        """
        if stop.wait_time_minutes > 0:
            popup_html += f"<br>Wait: {stop.wait_time_minutes} min"

        folium.Marker(
            location=[stop.location.latitude, stop.location.longitude],
            popup=popup_html,
            icon=folium.DivIcon(
                html=f'<div style="font-size: 12pt; color: white; background-color: {viz.get("stop_color", "blue")}; border-radius: 50%; width: 24px; height: 24px; text-align: center; line-height: 24px;">{stop.sequence}</div>'
            ),
        ).add_to(m)

    # Draw route lines
    route_coords = [[s.location.latitude, s.location.longitude] for s in route_result.stops]
    # Add return to depot
    route_coords.append([depot.latitude, depot.longitude])

    folium.PolyLine(
        locations=route_coords,
        weight=viz.get("route_weight", 3),
        color=viz.get("route_color", "blue"),
        opacity=0.8,
    ).add_to(m)

    # Count actual delivery stops (excluding depot)
    delivery_stops = [s for s in route_result.stops if s.sequence > 0]
    actual_stop_count = len(delivery_stops)

    # Add route info box
    info_html = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.3); z-index: 1000;">
    <b>Route Summary</b><br>
    Delivery Stops: {actual_stop_count}<br>
    Distance: {route_result.total_distance_km} km<br>
    Duration: {route_result.total_duration_minutes} min<br>
    Pallets: {dispatch.total_pallets}<br>
    Priority: {dispatch.total_priority}<br>
    Solver: {route_result.solver_used}
    </div>
    """
    m.get_root().html.add_child(folium.Element(info_html))

    return m


def create_multi_dispatch_map(
    dispatches: list[DispatchWithRoute],
    depot: Location,
    config: RoutingConfig,
    show_top_n: int = 3,
) -> folium.Map:
    """Create map showing multiple dispatch routes overlaid.

    Args:
        dispatches: List of dispatches with routes.
        depot: Depot location.
        config: Routing configuration.
        show_top_n: Number of routes to show.

    Returns:
        Folium Map object.
    """
    viz = config.visualization
    colors = viz.get("route_colors", ["#3388ff", "#ff7800", "#28a745", "#dc3545", "#6f42c1"])
    color_names = ["Blue", "Orange", "Green", "Red", "Purple"]

    # Calculate center from all routes
    all_lats = []
    all_lons = []
    for d in dispatches[:show_top_n]:
        for stop in d.route.stops:
            all_lats.append(stop.location.latitude)
            all_lons.append(stop.location.longitude)

    if not all_lats:
        center_lat, center_lon = depot.latitude, depot.longitude
    else:
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=viz.get("zoom_start", 11),
        tiles="cartodbpositron"
    )

    # Add depot
    folium.Marker(
        location=[depot.latitude, depot.longitude],
        popup="<b>Eco-Bags Factory</b><br>Depot Location",
        tooltip="Factory Depot",
        icon=folium.Icon(color="black", icon="industry", prefix="fa")
    ).add_to(m)

    # Draw routes in reverse order so higher-ranked routes appear on top
    for i in range(min(show_top_n, len(dispatches)) - 1, -1, -1):
        dispatch = dispatches[i]
        color = colors[i % len(colors)]
        route = dispatch.route

        # Draw route line with varying weight (thicker for higher rank)
        route_coords = [[s.location.latitude, s.location.longitude] for s in route.stops]
        route_coords.append([depot.latitude, depot.longitude])

        # Higher ranked routes get thicker lines
        line_weight = 5 - i  # Rank 1 = 5, Rank 2 = 4, Rank 3 = 3

        folium.PolyLine(
            locations=route_coords,
            weight=line_weight,
            color=color,
            opacity=0.9,
            popup=f"<b>Route #{i+1}</b><br>{dispatch.strategy}<br>Distance: {dispatch.total_distance_km:.1f} km",
            tooltip=f"Route #{i+1}: {dispatch.strategy}",
        ).add_to(m)

        # Add numbered markers for each stop in this route
        for stop in route.stops[1:]:  # Skip depot
            # Create numbered marker with DivIcon
            number_html = f'''
                <div style="
                    background-color: {color};
                    color: white;
                    border: 2px solid white;
                    border-radius: 50%;
                    width: {22 - i*2}px;
                    height: {22 - i*2}px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: {12 - i}px;
                    font-weight: bold;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                ">{stop.sequence}</div>
            '''
            folium.Marker(
                location=[stop.location.latitude, stop.location.longitude],
                icon=folium.DivIcon(
                    html=number_html,
                    icon_size=(22 - i*2, 22 - i*2),
                    icon_anchor=((22 - i*2)//2, (22 - i*2)//2),
                ),
                popup=f"<b>Route #{i+1} - Stop {stop.sequence}</b><br>{stop.location.name}",
                tooltip=f"Route #{i+1}: Stop {stop.sequence} - {stop.location.name}",
            ).add_to(m)

    # Add legend
    legend_items = ""
    for i in range(min(show_top_n, len(dispatches))):
        color = colors[i % len(colors)]
        color_name = color_names[i % len(color_names)]
        strategy = dispatches[i].strategy[:25] + "..." if len(dispatches[i].strategy) > 25 else dispatches[i].strategy
        legend_items += f'<i style="background:{color}; width:18px; height:18px; display:inline-block; margin-right:5px; border-radius:3px;"></i> #{i+1} {strategy}<br>'

    legend_html = f"""
    <div style="position: fixed; top: 20px; right: 20px; background: white; padding: 12px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.3); z-index: 1000; font-size: 12px;">
        <b>Route Comparison</b><br>
        <hr style="margin: 5px 0;">
        {legend_items}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def save_route_map(folium_map: folium.Map, output_path: Path) -> None:
    """Save Folium map to HTML file.

    Args:
        folium_map: The Folium map to save.
        output_path: Output file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    folium_map.save(str(output_path))


# ============================================================================
# Export Functions
# ============================================================================


def dispatch_with_route_to_dict(d: DispatchWithRoute) -> dict:
    """Convert DispatchWithRoute to dictionary for JSON export."""
    return {
        "candidate_id": d.candidate_id,
        "strategy": d.strategy,
        "order_ids": d.order_ids,
        "order_count": d.order_count,
        "total_pallets": d.total_pallets,
        "total_priority": d.total_priority,
        "zones": d.zones,
        "is_single_zone": d.is_single_zone,
        "has_mandatory": d.has_mandatory,
        "mandatory_count": d.mandatory_count,
        "route": {
            "total_distance_km": d.route.total_distance_km,
            "total_duration_minutes": d.route.total_duration_minutes,
            "route_start_time": d.route.route_start_time.isoformat(),
            "route_end_time": d.route.route_end_time.isoformat(),
            "feasible": d.route.feasible,
            "solver_used": d.route.solver_used,
            "stops": [
                {
                    "sequence": s.sequence,
                    "location_id": s.location.id,
                    "location_name": s.location.name,
                    "latitude": s.location.latitude,
                    "longitude": s.location.longitude,
                    "arrival_time": s.arrival_time.isoformat() if s.arrival_time else None,
                    "departure_time": s.departure_time.isoformat() if s.departure_time else None,
                    "distance_from_previous_km": s.distance_from_previous_km,
                    "cumulative_distance_km": s.cumulative_distance_km,
                    "wait_time_minutes": s.wait_time_minutes,
                }
                for s in d.route.stops
            ],
        },
        "total_distance_km": d.total_distance_km,
        "total_duration_minutes": d.total_duration_minutes,
        "priority_per_km": round(d.total_priority / max(d.total_distance_km, 0.1), 2),
    }


def export_dispatches_with_routes(
    dispatches: list[DispatchWithRoute],
    output_path: Path,
) -> None:
    """Export dispatches with routes to JSON file.

    Args:
        dispatches: List of dispatches with routes.
        output_path: Output JSON file path.
    """
    data = {
        "generated_at": datetime.now().isoformat(),
        "dispatch_count": len(dispatches),
        "dispatches": [dispatch_with_route_to_dict(d) for d in dispatches],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
