"""Helper functions for the Streamlit showcase app.

This module contains utility functions for data loading, transformations,
and visualization support for the delivery optimization showcase.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import func

from src.database import (
    ClientModel,
    DatabaseManager,
    LocalityModel,
    OrderModel,
    ProductModel,
    ZoneModel,
)

# ============================================================================
# Constants
# ============================================================================

# Purple/violet monochromatic color palette - darker, more professional
PURPLE_COLORS = [
    "#5B21B6",  # Primary purple (darker)
    "#6D28D9",  # Secondary purple
    "#7C3AED",  # Medium purple
    "#8B5CF6",  # Lighter purple
    "#A78BFA",  # Even lighter
    "#C4B5FD",  # Soft purple
]

# Zone colors for maps (consistent across all visualizations)
ZONE_COLORS = {
    "CABA": "#DC2626",          # Red - matches 🔴
    "NORTH_ZONE": "#2563EB",    # Blue - matches 🔵
    "SOUTH_ZONE": "#16A34A",    # Green - matches 🟢
    "WEST_ZONE": "#F59E0B",     # Amber/Yellow - matches 🟡
}

# Zone display names
ZONE_DISPLAY_NAMES = {
    "CABA": "Buenos Aires City",
    "NORTH_ZONE": "North Zone",
    "SOUTH_ZONE": "South Zone",
    "WEST_ZONE": "West Zone",
}

# Depot location
DEPOT_LAT = -34.73231090267173
DEPOT_LON = -58.295889556357935
DEPOT_NAME = "ECO-BAGS Factory"

# Plotly template configuration
PLOTLY_TEMPLATE = {
    "layout": {
        "colorway": PURPLE_COLORS,
        "font": {"color": "#374151", "family": "sans-serif"},
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "title": {"font": {"color": "#1F2937"}},
    }
}


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """Apply consistent purple theme to Plotly figures.

    Args:
        fig: Plotly figure to style.

    Returns:
        Styled Plotly figure.
    """
    fig.update_layout(
        font_family="sans-serif",
        font_color="#374151",
        paper_bgcolor="white",
        plot_bgcolor="white",
        title_font_color="#1F2937",
        colorway=PURPLE_COLORS,
    )
    return fig


# ============================================================================
# Score Normalization (MinMax Scaling)
# ============================================================================

# Define the actual min/max ranges for each score component
# Based on scoring.py configuration:
SCORE_RANGES = {
    "urgency": {"min": 0, "max": 150},      # Can go up to 150 for overdue orders
    "payment": {"min": 0, "max": 100},     # Capped at 100
    "client": {"min": 20, "max": 100},    # Ranges from occasional (20) to star (100)
    "age": {"min": 0, "max": 100},        # Capped at 100
}


def minmax_scale(value: float, min_val: float, max_val: float) -> float:
    """Scale a value to 0-1 range using MinMax normalization.

    Args:
        value: The value to scale.
        min_val: The minimum value in the range.
        max_val: The maximum value in the range.

    Returns:
        Normalized value between 0 and 1.
    """
    if max_val == min_val:
        return 0.5  # If no range, return middle value
    return (value - min_val) / (max_val - min_val)


def normalize_score(
    score: float,
    score_type: str,
) -> float:
    """Normalize a score component to 0-1 range.

    Args:
        score: The raw score value.
        score_type: Type of score ("urgency", "payment", "client", or "age").

    Returns:
        Normalized score between 0 and 1.
    """
    score_range = SCORE_RANGES.get(score_type, {"min": 0, "max": 100})
    return minmax_scale(score, score_range["min"], score_range["max"])


def normalize_scores_dict(raw_scores: dict[str, float]) -> dict[str, float]:
    """Normalize all score components in a dictionary.

    Args:
        raw_scores: Dictionary with keys "urgency_score", "payment_score",
                   "client_score", "age_score".

    Returns:
        Dictionary with normalized scores (0-1 range).
    """
    return {
        "urgency": normalize_score(
            raw_scores.get("urgency_score", 0), "urgency"
        ),
        "payment": normalize_score(
            raw_scores.get("payment_score", 0), "payment"
        ),
        "client": normalize_score(
            raw_scores.get("client_score", 0), "client"
        ),
        "age": normalize_score(
            raw_scores.get("age_score", 0), "age"
        ),
    }


# ============================================================================
# Data Loading Functions
# ============================================================================


def get_db_manager() -> DatabaseManager:
    """Get database manager instance.

    Returns:
        DatabaseManager connected to the processed database.
    """
    db_path = Path("data/processed/delivery.db")
    return DatabaseManager(db_path)


def load_zones_df(db: DatabaseManager) -> pd.DataFrame:
    """Load zones from database as DataFrame.

    Args:
        db: Database manager instance.

    Returns:
        DataFrame with zone data.
    """
    with db.get_session() as session:
        zones = session.query(ZoneModel).all()
        data = [
            {
                "zone_id": z.zone_id,
                "name": z.name,
                "color": z.color,
            }
            for z in zones
        ]
    return pd.DataFrame(data)


def load_localities_df(db: DatabaseManager) -> pd.DataFrame:
    """Load localities from database as DataFrame.

    Args:
        db: Database manager instance.

    Returns:
        DataFrame with locality data.
    """
    with db.get_session() as session:
        localities = session.query(LocalityModel).all()
        data = [
            {
                "locality_id": loc.locality_id,
                "name": loc.name,
                "zone_id": loc.zone_id,
                "latitude": loc.latitude,
                "longitude": loc.longitude,
            }
            for loc in localities
        ]
    return pd.DataFrame(data)


def load_clients_df(db: DatabaseManager) -> pd.DataFrame:
    """Load clients from database as DataFrame.

    Args:
        db: Database manager instance.

    Returns:
        DataFrame with client data.
    """
    with db.get_session() as session:
        clients = session.query(ClientModel).all()
        data = [
            {
                "client_id": c.client_id,
                "business_name": c.business_name,
                "zone_id": c.zone_id,
                "latitude": c.latitude,
                "longitude": c.longitude,
                "is_star_client": c.is_star_client,
                "is_new_client": c.is_new_client,
            }
            for c in clients
        ]
    return pd.DataFrame(data)


def load_products_df(db: DatabaseManager) -> pd.DataFrame:
    """Load products from database as DataFrame.

    Args:
        db: Database manager instance.

    Returns:
        DataFrame with product data.
    """
    with db.get_session() as session:
        products = session.query(ProductModel).all()
        data = [
            {
                "product_id": p.product_id,
                "name": p.name,
                "bag_type": p.bag_type,
                "packs_per_pallet": p.packs_per_pallet,
            }
            for p in products
        ]
    return pd.DataFrame(data)


def load_orders_df(db: DatabaseManager, status: str | None = None) -> pd.DataFrame:
    """Load orders from database as DataFrame.

    Args:
        db: Database manager instance.
        status: Optional filter by order status.

    Returns:
        DataFrame with order data.
    """
    with db.get_session() as session:
        query = session.query(OrderModel, ClientModel).join(
            ClientModel, OrderModel.client_id == ClientModel.client_id
        )
        if status:
            query = query.filter(OrderModel.status == status)
        
        results = query.all()
        data = []
        for order, client in results:
            data.append({
                "order_id": order.order_id,
                "client_id": order.client_id,
                "client_name": client.business_name,
                "issue_date": order.issue_date,
                "delivery_deadline": order.delivery_deadline,
                "zone_id": order.delivery_zone_id,
                "total_amount": order.total_amount,
                "payment_status": order.payment_status,
                "total_pallets": order.total_pallets,
                "priority_score": order.priority_score,
                "is_mandatory": order.is_mandatory,
                "status": order.status,
                "latitude": order.delivery_latitude,
                "longitude": order.delivery_longitude,
            })
    return pd.DataFrame(data)


def load_dispatch_candidates() -> list[dict[str, Any]]:
    """Load dispatch candidates from JSON file.

    Returns:
        List of dispatch candidate dictionaries.
    """
    path = Path("output/dispatches/dispatch_candidates.json")
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("candidates", [])


def load_ranked_dispatches() -> list[dict[str, Any]]:
    """Load ranked dispatches with routes from JSON file.

    Returns:
        List of ranked dispatch dictionaries with route info.
    """
    path = Path("output/dispatches/ranked_dispatches_with_routes.json")
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("dispatches", [])


def load_scoring_weights() -> dict[str, float]:
    """Load scoring weights from config file.

    Returns:
        Dictionary with weight configuration.
    """
    path = Path("data/config/scoring_weights.json")
    if not path.exists():
        return {
            "urgency": 0.40,
            "payment": 0.25,
            "client": 0.20,
            "age": 0.15,
        }
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("weights", {})


# ============================================================================
# Scoring Functions (In-Memory Recalculation)
# ============================================================================


def calculate_urgency_score(
    delivery_deadline: date,
    reference_date: date | None = None,
    max_days: int = 30,
) -> float:
    """Calculate urgency score based on days until deadline.

    Args:
        delivery_deadline: The order's delivery deadline.
        reference_date: Reference date for calculation.
        max_days: Maximum days for normalization.

    Returns:
        Urgency score between 0 and 150.
    """
    if reference_date is None:
        reference_date = date.today()

    days_remaining = (delivery_deadline - reference_date).days

    if days_remaining < 0:
        overdue_penalty = abs(days_remaining) * 10
        return min(150.0, 100.0 + overdue_penalty)

    days_remaining = max(0, min(days_remaining, max_days))
    if max_days == 0:
        return 100.0

    score = 100.0 * (1 - (days_remaining / max_days))
    return max(0.0, min(100.0, score))


def calculate_payment_score(
    total_amount: float,
    payment_status: str | None,
    p15_amount: float = 1000.0,
    p85_amount: float = 5000.0,
) -> float:
    """Calculate payment score based on amount and status.

    Args:
        total_amount: Order total amount.
        payment_status: Payment status string.
        p15_amount: 15th percentile amount.
        p85_amount: 85th percentile amount.

    Returns:
        Payment score between 0 and 100.
    """
    # Payment multipliers
    multipliers = {"paid": 1.0, "partial": 0.6, "pending": 0.3}

    range_size = p85_amount - p15_amount
    if range_size == 0:
        base_score = 100.0
    elif total_amount <= p15_amount:
        base_score = 20.0
    elif total_amount >= p85_amount:
        base_score = 100.0
    else:
        base_score = 20.0 + ((total_amount - p15_amount) / range_size) * 80.0

    status = (payment_status or "pending").lower().strip()
    multiplier = multipliers.get(status, 0.3)

    return max(0.0, min(100.0, base_score * multiplier))


def calculate_client_score(is_star: bool, is_new: bool) -> float:
    """Calculate client score based on client type.

    Args:
        is_star: Whether client is a star client.
        is_new: Whether client is a new client.

    Returns:
        Client score between 0 and 100.
    """
    if is_star:
        return 100.0
    elif is_new:
        return 80.0
    else:
        return 40.0  # Regular client default


def calculate_age_score(
    issue_date: date,
    reference_date: date | None = None,
    max_days: int = 30,
) -> float:
    """Calculate age score based on days since order issue.

    Args:
        issue_date: Order issue date.
        reference_date: Reference date for calculation.
        max_days: Maximum age for normalization.

    Returns:
        Age score between 0 and 100.
    """
    if reference_date is None:
        reference_date = date.today()

    days_old = (reference_date - issue_date).days
    days_old = max(0, min(days_old, max_days))

    if max_days == 0:
        return 0.0

    return (days_old / max_days) * 100.0


def recalculate_priorities(
    orders_df: pd.DataFrame,
    weights: dict[str, float],
    clients_df: pd.DataFrame,
) -> pd.DataFrame:
    """Recalculate priority scores with custom weights.

    Args:
        orders_df: DataFrame with order data.
        weights: Dictionary with component weights.
        clients_df: DataFrame with client data.

    Returns:
        DataFrame with recalculated scores and components.
    """
    result_df = orders_df.copy()
    reference_date = date.today()

    # Merge client info
    client_lookup = clients_df.set_index("client_id")[
        ["is_star_client", "is_new_client"]
    ].to_dict("index")

    # Calculate component scores
    urgency_scores = []
    payment_scores = []
    client_scores = []
    age_scores = []

    for _, row in result_df.iterrows():
        # Urgency
        deadline = row["delivery_deadline"]
        if isinstance(deadline, str):
            deadline = date.fromisoformat(deadline)
        urgency = calculate_urgency_score(deadline, reference_date)
        urgency_scores.append(urgency)

        # Payment
        payment = calculate_payment_score(row["total_amount"], row["payment_status"])
        payment_scores.append(payment)

        # Client
        client_info = client_lookup.get(row["client_id"], {})
        client_score = calculate_client_score(
            client_info.get("is_star_client", False),
            client_info.get("is_new_client", False),
        )
        client_scores.append(client_score)

        # Age
        issue = row["issue_date"]
        if isinstance(issue, str):
            issue = date.fromisoformat(issue)
        age = calculate_age_score(issue, reference_date)
        age_scores.append(age)

    result_df["urgency_score"] = urgency_scores
    result_df["payment_score"] = payment_scores
    result_df["client_score"] = client_scores
    result_df["age_score"] = age_scores

    # Calculate weighted priority
    w_urgency = weights.get("urgency", 0.4)
    w_payment = weights.get("payment", 0.25)
    w_client = weights.get("client", 0.2)
    w_age = weights.get("age", 0.15)

    result_df["calculated_priority"] = (
        result_df["urgency_score"] * w_urgency
        + result_df["payment_score"] * w_payment
        + result_df["client_score"] * w_client
        + result_df["age_score"] * w_age
    )

    # Mandatory orders get infinite priority
    result_df.loc[result_df["is_mandatory"] == True, "calculated_priority"] = 999999.0

    return result_df


def get_client_type_label(is_star: bool, is_new: bool) -> str:
    """Get human-readable client type label.

    Args:
        is_star: Whether client is a star client.
        is_new: Whether client is a new client.

    Returns:
        Client type label string.
    """
    if is_star:
        return "Star Client"
    elif is_new:
        return "New Client"
    else:
        return "Regular"


# ============================================================================
# Map Functions
# ============================================================================


def create_overview_map(
    localities_df: pd.DataFrame,
    zones_df: pd.DataFrame,
) -> folium.Map:
    """Create overview map showing zones and localities.

    Args:
        localities_df: DataFrame with locality data.
        zones_df: DataFrame with zone data.

    Returns:
        Folium map object.
    """
    # Center on Buenos Aires
    center_lat = -34.6037
    center_lon = -58.3816

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="cartodbpositron",
    )

    # Add depot marker
    folium.Marker(
        location=[DEPOT_LAT, DEPOT_LON],
        popup=f"<b>{DEPOT_NAME}</b><br>Depot Location",
        tooltip=DEPOT_NAME,
        icon=folium.Icon(color="black", icon="industry", prefix="fa"),
    ).add_to(m)

    # Add locality markers by zone - use ZONE_COLORS for consistency
    for zone_id in localities_df["zone_id"].unique():
        zone_locs = localities_df[localities_df["zone_id"] == zone_id]
        color = ZONE_COLORS.get(zone_id, "#5B21B6")

        for _, loc in zone_locs.iterrows():
            folium.CircleMarker(
                location=[loc["latitude"], loc["longitude"]],
                radius=6,
                popup=f"<b>{loc['name']}</b><br>Zone: {zone_id}",
                tooltip=loc["name"],
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
            ).add_to(m)

    # Add legend with consistent colors
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index:1000; 
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid grey; font-size: 12px;">
        <b>Zones</b><br>
    """
    for zone_id, color in ZONE_COLORS.items():
        name = ZONE_DISPLAY_NAMES.get(zone_id, zone_id)
        legend_html += f'<i style="background:{color};width:12px;height:12px;'
        legend_html += f'display:inline-block;margin-right:5px;"></i>{name}<br>'
    legend_html += '<i style="background:black;width:12px;height:12px;display:inline-block;margin-right:5px;"></i>Depot</div>'

    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def create_route_map(dispatch: dict[str, Any]) -> folium.Map:
    """Create map showing delivery route for a dispatch.

    Args:
        dispatch: Dispatch dictionary with route information.

    Returns:
        Folium map object with route visualization.
    """
    route = dispatch.get("route", {})
    stops = route.get("stops", [])

    if not stops:
        # Return empty map centered on depot
        return folium.Map(
            location=[DEPOT_LAT, DEPOT_LON],
            zoom_start=11,
            tiles="cartodbpositron",
        )

    # Calculate map center
    lats = [s["latitude"] for s in stops]
    lons = [s["longitude"] for s in stops]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="cartodbpositron",
    )

    # Add route lines
    route_coords = [[s["latitude"], s["longitude"]] for s in stops]
    # Add return to depot
    route_coords.append([DEPOT_LAT, DEPOT_LON])

    folium.PolyLine(
        locations=route_coords,
        color="#5B21B6",
        weight=3,
        opacity=0.8,
        dash_array="10",
    ).add_to(m)

    # Add stop markers
    for stop in stops:
        seq = stop["sequence"]
        name = stop["location_name"]
        arrival = stop.get("arrival_time", "N/A")
        dist = stop.get("distance_from_previous_km", 0)

        if seq == 0:
            # Depot marker
            folium.Marker(
                location=[stop["latitude"], stop["longitude"]],
                popup=f"<b>{name}</b><br>Start: {arrival}",
                tooltip=f"Depot: {name}",
                icon=folium.Icon(color="black", icon="industry", prefix="fa"),
            ).add_to(m)
        else:
            # Delivery stop marker with sequence number
            popup_html = f"""
            <b>Stop {seq}: {name}</b><br>
            Arrival: {arrival}<br>
            Distance from prev: {dist:.1f} km
            """
            folium.Marker(
                location=[stop["latitude"], stop["longitude"]],
                popup=popup_html,
                tooltip=f"Stop {seq}: {name}",
                icon=folium.DivIcon(
                    html=f"""
                    <div style="background-color: #5B21B6; color: white; 
                                border-radius: 50%; width: 24px; height: 24px;
                                display: flex; align-items: center; justify-content: center;
                                font-weight: bold; font-size: 12px; border: 2px solid white;">
                        {seq}
                    </div>
                    """,
                    icon_size=(24, 24),
                    icon_anchor=(12, 12),
                ),
            ).add_to(m)

    return m


# ============================================================================
# Chart Functions
# ============================================================================


def create_priority_radar_chart(
    order_id: str,
    orders_df: pd.DataFrame,
    weights: dict[str, float],
) -> go.Figure:
    """Create radar chart showing score breakdown for an order.

    Uses MinMax-scaled normalized scores (0-1 range) for consistent radar visualization.
    Each component is scaled based on its actual data range:
    - Urgency: 0-150 (can exceed 100 for overdue orders)
    - Payment: 0-100
    - Client: 20-100
    - Age: 0-100

    Args:
        order_id: Order ID to visualize.
        orders_df: DataFrame with order scores.
        weights: Current weight configuration.

    Returns:
        Plotly radar chart figure.
    """
    order = orders_df[orders_df["order_id"] == order_id].iloc[0]

    categories = ["Urgency", "Payment", "Client", "Age"]
    raw_scores = {
        "urgency_score": order.get("urgency_score", 0),
        "payment_score": order.get("payment_score", 0),
        "client_score": order.get("client_score", 0),
        "age_score": order.get("age_score", 0),
    }

    # Normalize all scores to 0-1 range
    normalized = normalize_scores_dict(raw_scores)
    normalized_scores = [
        normalized["urgency"],
        normalized["payment"],
        normalized["client"],
        normalized["age"],
    ]

    # Create hover text with raw and normalized values
    hover_text = [
        f"Urgency<br>Raw: {raw_scores['urgency_score']:.1f}/150<br>Normalized: {normalized['urgency']:.3f}",
        f"Payment<br>Raw: {raw_scores['payment_score']:.1f}/100<br>Normalized: {normalized['payment']:.3f}",
        f"Client<br>Raw: {raw_scores['client_score']:.1f}/100<br>Normalized: {normalized['client']:.3f}",
        f"Age<br>Raw: {raw_scores['age_score']:.1f}/100<br>Normalized: {normalized['age']:.3f}",
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=normalized_scores + [normalized_scores[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill="toself",
            name="Normalized Score",
            fillcolor="rgba(91, 33, 182, 0.3)",
            line=dict(color="#5B21B6", width=2),
            hovertext=hover_text + [hover_text[0]],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10),
            ),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        showlegend=False,
        title=f"Score Breakdown (Normalized): {order_id}<br><sub>All scores scaled to 0-1 range</sub>",
        height=350,
    )

    return apply_plotly_theme(fig)


def create_priority_comparison_chart(
    orders_df: pd.DataFrame,
    order_ids: list[str],
    priority_labels: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> go.Figure:
    """Create stacked bar chart comparing priority components across orders.

    Shows raw scores and weighted contributions for High, Medium, Low priority orders.
    Uses purple/violet color degradation with additional component details in hover text.

    Args:
        orders_df: DataFrame with order scores.
        order_ids: List of order IDs to compare.
        priority_labels: Optional labels for orders (e.g., ["High", "Medium", "Low"]).
        weights: Scoring weights dict with urgency, payment, client, age keys.

    Returns:
        Plotly stacked bar chart figure.
    """
    components = ["urgency_score", "payment_score", "client_score", "age_score"]
    component_names = ["Urgency", "Payment", "Client", "Age"]
    # Purple/violet degradation: dark purple -> medium purple -> light purple -> lavender
    component_colors = ["#3E1F47", "#553399", "#7851A9", "#9370DB"]
    
    if priority_labels is None:
        priority_labels = ["🔴 High Priority", "🟡 Medium Priority", "🟢 Low Priority"]
    
    if weights is None:
        weights = {"urgency": 0.40, "payment": 0.25, "client": 0.20, "age": 0.15}
    
    weight_values = [
        weights.get("urgency", 0.4),
        weights.get("payment", 0.25),
        weights.get("client", 0.2),
        weights.get("age", 0.15),
    ]

    fig = go.Figure()
    
    # Build data for each order with additional component details
    order_data = []
    for i, order_id in enumerate(order_ids):
        order = orders_df[orders_df["order_id"] == order_id]
        if order.empty:
            continue
        order = order.iloc[0]
        
        label = priority_labels[i] if i < len(priority_labels) else f"Order {i+1}"
        raw_scores = [order.get(comp, 0) for comp in components]
        weighted_scores = [r * w for r, w in zip(raw_scores, weight_values)]
        final_score = sum(weighted_scores)
        
        # Extract component-specific details
        delivery_deadline = order.get("delivery_deadline")
        if isinstance(delivery_deadline, str):
            delivery_deadline = pd.to_datetime(delivery_deadline).date()
        days_to_deadline = (delivery_deadline - date.today()).days if delivery_deadline else 0
        
        payment_status = order.get("payment_status", "pending")
        total_amount = order.get("total_amount", 0)
        
        issue_date = order.get("issue_date")
        if isinstance(issue_date, str):
            issue_date = pd.to_datetime(issue_date).date()
        days_since_issue = (date.today() - issue_date).days if issue_date else 0
        
        # Determine client type
        is_star = order.get("is_star_client", False)
        is_new = order.get("is_new_client", False)
        if is_star:
            client_type = "Star Client"
        elif is_new:
            client_type = "New Client"
        else:
            client_type = "Regular Client"
        
        # Determine payment multiplier
        payment_status_lower = payment_status.lower() if isinstance(payment_status, str) else "pending"
        if payment_status_lower == "paid":
            payment_multiplier = 1.0
        elif payment_status_lower == "partial":
            payment_multiplier = 0.6
        else:
            payment_multiplier = 0.3
        
        order_data.append({
            "label": label,
            "order_id": order_id,
            "raw_scores": raw_scores,
            "weighted_scores": weighted_scores,
            "final_score": final_score,
            "days_to_deadline": days_to_deadline,
            "payment_status": payment_status,
            "payment_multiplier": payment_multiplier,
            "total_amount": total_amount,
            "days_since_issue": days_since_issue,
            "client_type": client_type,
        })
    
    # Create stacked bar chart with weighted contributions and detailed hover info
    for j, (comp_name, color) in enumerate(zip(component_names, component_colors)):
        # Build detailed hover text based on component type
        hover_texts = []
        for d in order_data:
            if j == 0:  # Urgency
                hover_detail = f"Days to deadline: {d['days_to_deadline']}"
            elif j == 1:  # Payment
                hover_detail = f"Status: {d['payment_status']}<br>Amount: ${d['total_amount']:.2f}<br>Multiplier: {d['payment_multiplier']:.1f}x"
            elif j == 2:  # Client
                hover_detail = f"Type: {d['client_type']}"
            else:  # Age
                hover_detail = f"Days since issue: {d['days_since_issue']}"
            
            hover_texts.append(hover_detail)
        
        fig.add_trace(
            go.Bar(
                name=f"{comp_name} ({weight_values[j]:.0%})",
                x=[d["label"] for d in order_data],
                y=[d["weighted_scores"][j] for d in order_data],
                marker_color=color,
                text=[f"{d['raw_scores'][j]:.0f}×{weight_values[j]:.0%}={d['weighted_scores'][j]:.1f}" 
                      for d in order_data],
                textposition="inside",
                textfont=dict(size=9, color="white"),
                hovertext=hover_texts,
                customdata=[[d["raw_scores"][j]] for d in order_data],
                hovertemplate=f"<b>{comp_name}</b><br>Raw: %{{customdata[0]:.1f}}<br>Weighted: %{{y:.1f}}<br>%{{hovertext}}<extra></extra>",
            )
        )
    
    # Add total score annotations
    for d in order_data:
        fig.add_annotation(
            x=d["label"],
            y=d["final_score"] + 3,
            text=f"Total: {d['final_score']:.1f}",
            showarrow=False,
            font=dict(size=11, color="#1F2937", weight="bold"),
        )

    fig.update_layout(
        barmode="stack",
        title="Priority Score Comparison<br><sup>Stacked bars show weighted component contributions (Raw × Weight = Weighted)</sup>",
        xaxis_title="",
        yaxis_title="Weighted Score Contribution",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        yaxis=dict(range=[0, max(d["final_score"] for d in order_data) * 1.15 if order_data else 120]),
    )

    return apply_plotly_theme(fig)


def create_strategy_comparison_chart(candidates: list[dict]) -> go.Figure:
    """Create bar chart comparing dispatch strategies.

    Args:
        candidates: List of dispatch candidate dictionaries.

    Returns:
        Plotly bar chart figure.
    """
    # Group by strategy
    strategy_data = {}
    for c in candidates:
        strategy = c.get("strategy", "unknown")
        summary = c.get("summary", {})
        priority = summary.get("total_priority", 0)
        utilization = summary.get("utilization_pct", 0)

        if strategy not in strategy_data:
            strategy_data[strategy] = {"priorities": [], "utilizations": []}
        strategy_data[strategy]["priorities"].append(priority)
        strategy_data[strategy]["utilizations"].append(utilization)

    # Calculate averages - only for strategies that generated candidates
    strategies = sorted(strategy_data.keys())
    avg_priorities = [
        sum(strategy_data[s]["priorities"]) / len(strategy_data[s]["priorities"])
        for s in strategies
    ]
    avg_utilizations = [
        sum(strategy_data[s]["utilizations"]) / len(strategy_data[s]["utilizations"])
        for s in strategies
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Avg Priority Score",
            x=strategies,
            y=avg_priorities,
            marker_color=PURPLE_COLORS[0],
            yaxis="y",
            text=[f"{p:.0f}" for p in avg_priorities],
            textposition="outside",
            textfont=dict(size=9),
        )
    )

    fig.add_trace(
        go.Scatter(
            name="Avg Utilization %",
            x=strategies,
            y=avg_utilizations,
            mode="lines+markers",
            marker=dict(color=PURPLE_COLORS[2], size=10),
            line=dict(color=PURPLE_COLORS[2], width=2),
            yaxis="y2",
            text=[f"{u:.0f}%" for u in avg_utilizations],
            textposition="top center",
            textfont=dict(size=8),
        )
    )

    fig.update_layout(
        title="Strategy Performance Comparison",
        xaxis_title="Strategy",
        xaxis=dict(tickangle=-45),
        yaxis=dict(title="Average Priority", side="left"),
        yaxis2=dict(title="Utilization %", side="right", overlaying="y", range=[0, 120]),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(b=100),
    )

    return apply_plotly_theme(fig)


def create_zone_composition_chart(candidates: list[dict]) -> go.Figure:
    """Create stacked bar chart showing zone breakdown per candidate.

    Args:
        candidates: List of dispatch candidate dictionaries.

    Returns:
        Plotly stacked bar chart figure.
    """
    # Limit to first 10 candidates for readability
    candidates = candidates[:10]

    candidate_ids = []
    zone_data = {zone: [] for zone in ZONE_COLORS.keys()}

    for c in candidates:
        cid = c.get("candidate_id", "")[-8:]  # Short ID
        candidate_ids.append(cid)
        breakdown = c.get("summary", {}).get("zone_breakdown", {})

        for zone in ZONE_COLORS.keys():
            zone_data[zone].append(breakdown.get(zone, 0))

    fig = go.Figure()

    for zone, color in ZONE_COLORS.items():
        fig.add_trace(
            go.Bar(
                name=zone,
                x=candidate_ids,
                y=zone_data[zone],
                marker_color=color,
            )
        )

    fig.update_layout(
        barmode="stack",
        title="Zone Composition by Candidate",
        xaxis_title="Candidate",
        yaxis_title="Order Count",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return apply_plotly_theme(fig)


def create_candidate_comparison_chart(
    candidates: list[dict],
    color_by: str = "strategy",
) -> go.Figure:
    """Create scatter plot comparing candidates by priority vs utilization.

    Args:
        candidates: List of dispatch candidate dictionaries.
        color_by: Field to color by ('strategy' or 'is_single_zone').

    Returns:
        Plotly scatter plot figure.
    """
    if not candidates:
        fig = go.Figure()
        fig.add_annotation(text="No candidates available", x=0.5, y=0.5, showarrow=False)
        return apply_plotly_theme(fig)

    # Limit candidates for readability
    candidates = candidates[:20]

    # Extract data
    data = []
    for c in candidates:
        summary = c.get("summary", {})
        candidate_id = c.get("candidate_id", "")
        strategy = c.get("strategy", "unknown")
        
        # Detect subset strategies
        is_subset = "-SUB-" in candidate_id
        display_strategy = strategy
        if is_subset:
            display_strategy = f"{strategy} (subset)"
        
        data.append({
            "candidate_id": candidate_id[-8:],
            "strategy": display_strategy,
            "raw_strategy": strategy,
            "utilization_pct": summary.get("utilization_pct", 0),
            "total_priority": summary.get("total_priority", 0),
            "order_count": summary.get("order_count", 0),
            "is_single_zone": summary.get("is_single_zone", False),
            "is_subset": is_subset,
        })

    df = pd.DataFrame(data)

    fig = px.scatter(
        df,
        x="utilization_pct",
        y="total_priority",
        size="order_count",
        hover_data=["candidate_id", "order_count", "is_single_zone", "strategy"],
        labels={
            "utilization_pct": "Truck Utilization (%)",
            "total_priority": "Total Priority Score",
            "order_count": "Orders",
        },
        title="Candidate Comparison: Priority vs Utilization",
    )

    fig.update_traces(
        marker=dict(line=dict(width=1, color="white"), color="#8B5CF6"),
        hovertemplate="<b>%{customdata[0]}</b><br>Utilization: %{x:.1f}%<br>Priority: %{y:,.0f}<br>Orders: %{customdata[1]}<extra></extra>",
    )

    # Add reference lines for capacity thresholds
    # 7 pallets = 87.5% of 8-pallet truck capacity
    # 8.5 pallets = 106.25% of 8-pallet truck capacity (maximum)
    
    fig.add_vline(
        x=87.5,
        line_dash="dash",
        line_color="#10B981",
        annotation_text="Min (7 pallets)",
        annotation_position="top left",
        annotation_font_size=11,
        annotation_font_color="#10B981",
    )
    
    fig.add_vline(
        x=106.25,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text="Max (8.5 pallets)",
        annotation_position="top right",
        annotation_font_size=11,
        annotation_font_color="#EF4444",
    )

    fig.update_layout(
        height=400,
        xaxis=dict(range=[50, 120]),
        showlegend=False,
    )

    return apply_plotly_theme(fig)


def create_priority_vs_distance_chart(
    dispatches: list[dict],
    include_mandatory: bool = False,
) -> go.Figure:
    """Create scatter plot of priority vs distance with bubble size for order count.

    Args:
        dispatches: List of ranked dispatch dictionaries.
        include_mandatory: Whether to include dispatches with mandatory orders.

    Returns:
        Plotly scatter plot figure with ideal zone annotation.
    """
    # Filter based on has_mandatory field (more reliable than priority threshold)
    if include_mandatory:
        filtered = dispatches
    else:
        filtered = [d for d in dispatches if not d.get("has_mandatory", False)]

    if not filtered:
        filtered = dispatches[:10]

    priorities = [d.get("total_priority", 0) for d in filtered]
    distances = [d.get("total_distance_km", 0) for d in filtered]
    order_counts = [d.get("order_count", 1) for d in filtered]
    is_single_zone = [d.get("is_single_zone", False) for d in filtered]
    candidate_ids = [d.get("candidate_id", "")[-8:] for d in filtered]
    
    # Normalize bubble sizes (min 10, max 35)
    min_orders = min(order_counts) if order_counts else 1
    max_orders = max(order_counts) if order_counts else 1
    order_range = max(max_orders - min_orders, 1)
    bubble_sizes = [10 + 25 * ((c - min_orders) / order_range) for c in order_counts]

    fig = go.Figure()

    # Single zone points
    single_idx = [i for i, sz in enumerate(is_single_zone) if sz]
    multi_idx = [i for i, sz in enumerate(is_single_zone) if not sz]

    if single_idx:
        fig.add_trace(
            go.Scatter(
                x=[distances[i] for i in single_idx],
                y=[priorities[i] for i in single_idx],
                mode="markers",
                name="Single Zone",
                marker=dict(
                    color=PURPLE_COLORS[0],
                    size=[bubble_sizes[i] for i in single_idx],
                    line=dict(width=1, color="white"),
                ),
                text=[f"{candidate_ids[i]}<br>Orders: {order_counts[i]}" for i in single_idx],
                hovertemplate="<b>%{text}</b><br>Distance: %{x:.1f} km<br>Priority: %{y:.0f}<extra></extra>",
            )
        )

    if multi_idx:
        fig.add_trace(
            go.Scatter(
                x=[distances[i] for i in multi_idx],
                y=[priorities[i] for i in multi_idx],
                mode="markers",
                name="Multi Zone",
                marker=dict(
                    color=PURPLE_COLORS[3],
                    size=[bubble_sizes[i] for i in multi_idx],
                    line=dict(width=1, color="white"),
                ),
                text=[f"{candidate_ids[i]}<br>Orders: {order_counts[i]}" for i in multi_idx],
                hovertemplate="<b>%{text}</b><br>Distance: %{x:.1f} km<br>Priority: %{y:.0f}<extra></extra>",
            )
        )
    
    # Mark the top ranked candidate (first in list after filtering)
    if filtered:
        top = filtered[0]
        fig.add_trace(
            go.Scatter(
                x=[top.get("total_distance_km", 0)],
                y=[top.get("total_priority", 0)],
                mode="markers+text",
                name="Top Ranked",
                marker=dict(
                    color="#DC2626",
                    size=18,
                    symbol="star",
                    line=dict(width=2, color="white"),
                ),
                text=["★ TOP"],
                textposition="top center",
                textfont=dict(size=10, color="#DC2626"),
                hovertemplate="<b>TOP RANKED</b><br>%{text}<br>Distance: %{x:.1f} km<br>Priority: %{y:.0f}<extra></extra>",
            )
        )
    
    # Add ideal zone annotation (upper-left quadrant = high priority, low distance)
    max_priority = max(priorities) if priorities else 100
    min_distance = min(distances) if distances else 0
    max_distance = max(distances) if distances else 100
    
    fig.add_shape(
        type="rect",
        x0=min_distance,
        y0=max_priority * 0.7,
        x1=min_distance + (max_distance - min_distance) * 0.35,
        y1=max_priority * 1.05,
        fillcolor="rgba(22, 163, 74, 0.1)",
        line=dict(color="rgba(22, 163, 74, 0.5)", width=2, dash="dash"),
    )
    
    fig.add_annotation(
        x=min_distance + (max_distance - min_distance) * 0.17,
        y=max_priority * 1.02,
        text="IDEAL ZONE",
        showarrow=False,
        font=dict(size=10, color="#16A34A"),
    )

    fig.update_layout(
        title="Priority vs Distance Trade-off (bubble size = order count)",
        xaxis_title="Total Distance (km)",
        yaxis_title="Total Priority Score",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return apply_plotly_theme(fig)


# ============================================================================
# Dispatch Helper Functions
# ============================================================================


def get_dispatch_summary_df(dispatches: list[dict]) -> pd.DataFrame:
    """Convert dispatch list to summary DataFrame.

    Args:
        dispatches: List of dispatch dictionaries.

    Returns:
        DataFrame with dispatch summary data.
    """
    data = []
    for i, d in enumerate(dispatches):
        route = d.get("route", {})
        data.append({
            "rank": i + 1,
            "candidate_id": d.get("candidate_id", ""),
            "strategy": d.get("strategy", ""),
            "order_count": d.get("order_count", 0),
            "total_pallets": d.get("total_pallets", 0),
            "total_priority": d.get("total_priority", 0),
            "zones": ", ".join(d.get("zones", [])),
            "is_single_zone": d.get("is_single_zone", False),
            "has_mandatory": d.get("has_mandatory", False),
            "distance_km": d.get("total_distance_km", route.get("total_distance_km", 0)),
            "duration_min": d.get("total_duration_minutes", route.get("total_duration_minutes", 0)),
            "priority_per_km": d.get("priority_per_km", 0),
        })
    return pd.DataFrame(data)


def get_candidates_summary_df(
    candidates: list[dict],
    include_mandatory: bool = True,
) -> pd.DataFrame:
    """Convert candidates list to summary DataFrame with ranking.

    Args:
        candidates: List of dispatch candidate dictionaries.
        include_mandatory: Whether to include candidates with mandatory orders.

    Returns:
        DataFrame with candidate summary data including rank and subset indicator.
    """
    data = []
    for c in candidates:
        summary = c.get("summary", {})
        has_mandatory = summary.get("mandatory_included", False)
        candidate_id = c.get("candidate_id", "")
        strategy = c.get("strategy", "")
        
        # Skip if filtering mandatory and this has mandatory
        if not include_mandatory and has_mandatory:
            continue
        
        # Detect subset strategies from candidate ID
        is_subset = "-SUB-" in candidate_id
        display_strategy = strategy
        if is_subset:
            display_strategy = f"{strategy} (subset)"
            
        data.append({
            "candidate_id": candidate_id,
            "strategy": display_strategy,
            "raw_strategy": strategy,
            "is_subset": is_subset,
            "order_count": summary.get("order_count", 0),
            "total_pallets": summary.get("total_pallets", 0),
            "utilization_pct": summary.get("utilization_pct", 0),
            "total_priority": summary.get("total_priority", 0),
            "zones": ", ".join(summary.get("zones", [])),
            "is_single_zone": summary.get("is_single_zone", False),
            "has_mandatory": has_mandatory,
        })
    
    df = pd.DataFrame(data)
    
    # Add ranking by priority (higher is better)
    if not df.empty:
        df = df.sort_values("total_priority", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
    
    return df


def filter_candidates_by_mandatory(
    candidates: list[dict],
    include_mandatory: bool = True,
) -> list[dict]:
    """Filter dispatch candidates by mandatory order status.

    Args:
        candidates: List of dispatch candidate dictionaries.
        include_mandatory: Whether to include candidates with mandatory orders.

    Returns:
        Filtered list of candidates.
    """
    if include_mandatory:
        return candidates
    
    return [
        c for c in candidates
        if not c.get("summary", {}).get("mandatory_included", False)
    ]


def get_strategy_display_name(strategy: str) -> str:
    """Get user-friendly strategy name with subset indicator.

    Args:
        strategy: Raw strategy name.

    Returns:
        Display name with [SUBSET] indicator if applicable.
    """
    subset_strategies = {
        "greedy_priority": "Greedy Priority",
        "greedy_efficiency": "Greedy Efficiency",
        "greedy_mandatory_nearest": "Greedy Mandatory + Nearest",
        "greedy_zone_caba": "Zone: CABA [SUBSET]",
        "greedy_zone_north": "Zone: North [SUBSET]",
        "greedy_zone_south": "Zone: South [SUBSET]", 
        "greedy_zone_west": "Zone: West [SUBSET]",
        "dp_optimal": "DP Optimal",
        "dp_optimal_subset": "DP Optimal [SUBSET]",
    }
    return subset_strategies.get(strategy, strategy)


def get_route_stops_df(dispatch: dict, candidates: list[dict] | None = None) -> pd.DataFrame:
    """Extract route stops as DataFrame with pallet information.

    Args:
        dispatch: Dispatch dictionary with route information.
        candidates: Optional list of dispatch candidates to look up pallet info.

    Returns:
        DataFrame with stop details including pallets and remaining capacity.
    """
    route = dispatch.get("route", {})
    stops = route.get("stops", [])
    
    # Build order lookup for pallets
    # First try to get from dispatch itself
    orders = dispatch.get("orders", [])
    order_pallets = {}
    for order in orders:
        order_id = order.get("order_id", "")
        order_pallets[order_id] = order.get("pallets", 0)
    
    # If no orders in dispatch, try looking up from candidates
    if not order_pallets and candidates:
        candidate_id = dispatch.get("candidate_id", "")
        for c in candidates:
            if c.get("candidate_id") == candidate_id:
                for order in c.get("orders", []):
                    order_id = order.get("order_id", "")
                    order_pallets[order_id] = order.get("pallets", 0)
                break

    # Get total pallets to calculate remaining
    total_pallets = dispatch.get("total_pallets", 0)
    if not total_pallets:
        total_pallets = sum(order_pallets.values())
    
    cumulative_dropped = 0.0
    data = []
    for stop in stops:
        location_id = stop.get("location_id", "")
        # Get pallets for this stop (0 for depot)
        pallets = order_pallets.get(location_id, 0)

        # Calculate remaining pallets AFTER this delivery
        cumulative_dropped += pallets
        pallets_remaining = total_pallets - cumulative_dropped
        
        data.append({
            "sequence": stop.get("sequence", 0),
            "location": stop.get("location_name", ""),
            "arrival_time": stop.get("arrival_time", ""),
            "distance_from_prev_km": stop.get("distance_from_previous_km", 0),
            "cumulative_km": stop.get("cumulative_distance_km", 0),
            "pallets_to_drop": round(pallets, 2) if pallets else 0,
            "pallets_remaining": round(pallets_remaining, 2),
        })
    return pd.DataFrame(data)


# ============================================================================
# Sample Receipt Data for Demo
# ============================================================================

SAMPLE_RECEIPTS = {
    "Invoice_Coffee_Shop.pdf": {
        "client_name": "Café Buenos Aires",
        "tax_id": "30-71234567-8",
        "issue_date": "2026-01-15",
        "delivery_address": "Av. Corrientes 1234, CABA",
        "items": [
            {"product": "ECO-MED-001", "quantity": 500, "pallets": 2.5},
            {"product": "ECO-LRG-002", "quantity": 200, "pallets": 1.2},
        ],
        "total_amount": 45000.00,
        "payment_status": "paid",
        "extraction_confidence": 0.95,
    },
    "Invoice_FMartínez.pdf": {
        "client_name": "Ferretería Martínez",
        "tax_id": "20-25896314-7",
        "issue_date": "2026-01-12",
        "delivery_address": "Av. San Martín 567, San Isidro",
        "items": [
            {"product": "ECO-SML-003", "quantity": 1000, "pallets": 3.0},
        ],
        "total_amount": 32000.00,
        "payment_status": "partial",
        "extraction_confidence": 0.92,
    },
    "Invoice_Toy_Store.pdf": {
        "client_name": "Juguetería El Mundo",
        "tax_id": "33-70159842-5",
        "issue_date": "2026-01-18",
        "delivery_address": "Calle Florida 890, CABA",
        "items": [
            {"product": "ECO-MED-001", "quantity": 300, "pallets": 1.5},
            {"product": "ECO-XLG-004", "quantity": 100, "pallets": 0.8},
        ],
        "total_amount": 28500.00,
        "payment_status": "pending",
        "extraction_confidence": 0.88,
    },
    "Receipt_El_Gaucho.pdf": {
        "client_name": "Restaurante El Gaucho",
        "tax_id": "30-65478912-3",
        "issue_date": "2026-01-10",
        "delivery_address": "Av. de Mayo 456, CABA",
        "items": [
            {"product": "ECO-LRG-002", "quantity": 400, "pallets": 2.0},
        ],
        "total_amount": 38000.00,
        "payment_status": "paid",
        "extraction_confidence": 0.97,
    },
}


def load_extracted_receipts() -> dict:
    """Load extracted receipt data from JSON file.
    
    Returns:
        Dictionary mapping filename to extracted data, or empty dict if file not found.
    """
    import json
    
    json_path = Path(__file__).parent.parent / "data" / "processed" / "extracted_receipts.json"
    
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load extracted_receipts.json: {e}")
            return {}
    return {}


def get_sample_receipt_data(filename: str) -> dict:
    """Get extracted data for a receipt file from real extraction.

    Args:
        filename: Name of the receipt file.

    Returns:
        Dictionary with extracted receipt data from real extraction.
    """
    # Load from real extracted data
    extracted_receipts = load_extracted_receipts()
    
    if extracted_receipts and filename in extracted_receipts:
        return extracted_receipts[filename]
    
    # Fallback to sample data if real extraction not available
    return SAMPLE_RECEIPTS.get(filename, SAMPLE_RECEIPTS["Invoice_Coffee_Shop.pdf"])


def get_receipt_image_path(filename: str) -> Path:
    """Get path to receipt image file.
    
    Args:
        filename: Receipt filename (e.g., "Invoice_Coffee_Shop.pdf").
        
    Returns:
        Path object to the image file (.jpg), or None if not found.
    """
    receipts_dir = Path(__file__).parent.parent / "data" / "raw" / "receipts"
    
    # Try to find corresponding JPG image
    base_name = filename.replace(".pdf", "").replace(".jpg", "")
    jpg_path = receipts_dir / f"{base_name}.jpg"
    
    if jpg_path.exists():
        return jpg_path
    
    return None


def get_receipt_image_base64(filename: str) -> str:
    """Get base64 encoded receipt image for display.
    
    Args:
        filename: Receipt filename.
        
    Returns:
        Base64 string of image data, or empty string if not found.
    """
    import base64
    
    image_path = get_receipt_image_path(filename)
    
    if image_path and image_path.exists():
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
    
    return ""


def get_available_receipt_files() -> list:
    """Get list of all available receipt files.
    
    Returns:
        List of PDF filenames in receipts directory.
    """
    receipts_dir = Path(__file__).parent.parent / "data" / "raw" / "receipts"
    
    if not receipts_dir.exists():
        return []
    
    pdf_files = sorted([f.name for f in receipts_dir.glob("*.pdf")])
    return pdf_files

