"""Streamlit Showcase App for Delivery Optimization System.

A multi-page application demonstrating the end-to-end delivery optimization
pipeline for an eco-friendly bag factory.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from src.app_utils import (
    DEPOT_LAT,
    DEPOT_LON,
    DEPOT_NAME,
    PURPLE_COLORS,
    ZONE_COLORS,
    ZONE_DISPLAY_NAMES,
    apply_plotly_theme,
    create_candidate_comparison_chart,
    create_overview_map,
    create_priority_comparison_chart,
    create_priority_radar_chart,
    create_priority_vs_distance_chart,
    create_route_map,
    create_strategy_comparison_chart,
    create_zone_composition_chart,
    filter_candidates_by_mandatory,
    get_available_receipt_files,
    get_candidates_summary_df,
    get_client_type_label,
    get_db_manager,
    get_dispatch_summary_df,
    get_receipt_image_base64,
    get_route_stops_df,
    get_sample_receipt_data,
    get_strategy_display_name,
    load_clients_df,
    load_dispatch_candidates,
    load_localities_df,
    load_orders_df,
    load_products_df,
    load_ranked_dispatches,
    load_scoring_weights,
    load_zones_df,
    recalculate_priorities,
)

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Delivery Optimization System",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================================
# Data Loading with Caching
# ============================================================================


@st.cache_data
def load_all_data():
    """Load all data from database and files."""
    db = get_db_manager()
    
    zones_df = load_zones_df(db)
    localities_df = load_localities_df(db)
    clients_df = load_clients_df(db)
    products_df = load_products_df(db)
    orders_df = load_orders_df(db, status="pending")
    
    dispatch_candidates = load_dispatch_candidates()
    ranked_dispatches = load_ranked_dispatches()
    default_weights = load_scoring_weights()
    
    return {
        "zones": zones_df,
        "localities": localities_df,
        "clients": clients_df,
        "products": products_df,
        "orders": orders_df,
        "dispatch_candidates": dispatch_candidates,
        "ranked_dispatches": ranked_dispatches,
        "default_weights": default_weights,
    }


# Load data
data = load_all_data()

zones_df = data["zones"]
localities_df = data["localities"]
clients_df = data["clients"]
products_df = data["products"]
orders_df = data["orders"]
dispatch_candidates = data["dispatch_candidates"]
ranked_dispatches = data["ranked_dispatches"]
default_weights = data["default_weights"]


# ============================================================================
# Header Section
# ============================================================================

st.title("🚚 Delivery Optimization System")
st.markdown("### *End-to-end pipeline for eco-friendly bag factory logistics*")

st.markdown("""
This application showcases a complete delivery optimization solution built for an eco-friendly 
bag factory. The system processes sales receipts, extracts data using AI, calculates delivery 
priorities, and generates optimized dispatch routes.
""")

# ============================================================================
# Tab Navigation
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Data Overview",
    "📄 Receipt Extraction", 
    "🎯 Priority Scoring",
    "📦 Order Selection",
    "🚚 Route Optimization",
    "ℹ️ About"
])


# ============================================================================
# Tab 1: Data Overview
# ============================================================================

with tab1:
    st.header("🗺️ Data Overview")
    st.markdown("*Geographic and business context of the delivery operation*")
    
    # KPI Cards with individual borders
    st.markdown("""
    <style>
    [data-testid="stMetric"] {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Zones", len(zones_df))
    with col2:
        st.metric("Localities", len(localities_df))
    with col3:
        st.metric("Clients", len(clients_df))
    with col4:
        st.metric("Products", len(products_df))
    with col5:
        st.metric("Pending Orders", len(orders_df))
    with col6:
        total_pallets = orders_df["total_pallets"].sum() if not orders_df.empty else 0
        st.metric("Total Pallets", f"{total_pallets:.1f}")
    
    st.markdown("")
    
    # Map and explanation - larger map, centered
    col_map, col_text = st.columns([3, 2])
    
    with col_map:
        overview_map = create_overview_map(localities_df, zones_df)
        st_folium(overview_map, width=None, height=520, returned_objects=[], use_container_width=True)
    
    with col_text:
        st.markdown("""
        #### Geographic Coverage
        
        The factory operates in the **Buenos Aires metropolitan area**, divided into 
        4 delivery zones:
        
        - 🔴 **CABA** - Buenos Aires City
        - 🔵 **North Zone** - Northern suburbs
        - 🟢 **South Zone** - Southern suburbs
        - 🟡 **West Zone** - Western suburbs
        
        The **depot** (marked with 🏭) is located in a industrial zone, 
        providing optimal access to major highways for deliveries.
        
        Each zone contains multiple localities where clients are located.
        The system optimizes routes within and across zones to minimize 
        travel time and fuel consumption.
        """)


# ============================================================================
# Tab 2: Receipt Extraction
# ============================================================================

with tab2:
    st.header("📄 Receipt Extraction")
    st.markdown("*AI-powered document processing for variable-format receipts*")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### Sample Receipt")
        
        # Get available receipts
        receipt_files = get_available_receipt_files()
        
        if receipt_files:
            # Create selectable dropdown
            selected_receipt = st.selectbox(
                "Select a receipt to view extracted data",
                options=receipt_files,
                index=0,
                key="receipt_selector",
            )
            
            st.success(f"Selected: **{selected_receipt}**")
            
            # Show receipt image
            image_base64 = get_receipt_image_base64(selected_receipt)
            if image_base64:
                st.markdown(f"""
                <div style="background-color: #EDE9FE; border-radius: 10px; padding: 20px; text-align: center;">
                    <img src="data:image/jpeg;base64,{image_base64}" style="max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #EDE9FE; border-radius: 10px; padding: 40px; 
                            text-align: center; border: 2px dashed #5B21B6;">
                    <h4 style="color: #5B21B6;">📋 Receipt Image Preview</h4>
                    <p style="color: #6B7280;">Image file not found</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            selected_receipt = "Invoice_Coffee_Shop.pdf"
            st.warning("No receipt files found in data/raw/receipts/")
    
    with col_right:
        st.markdown("#### Extracted Data")
        
        # Get real extracted data for selected receipt
        extracted_data = get_sample_receipt_data(selected_receipt if 'selected_receipt' in dir() else "Invoice_Coffee_Shop.pdf")
        
        # Create a formatted display
        st.markdown("**Client Information:**")
        st.markdown(f"""
        - **Name**: {extracted_data.get("client_name", "N/A")}
        - **Tax ID**: {extracted_data.get("tax_id", "N/A")}
        - **Issue Date**: {extracted_data.get("issue_date", "N/A")}
        - **Delivery Address**: {extracted_data.get("delivery_address", "N/A")}
        """)
        
        st.markdown("**Items:**")
        items = extracted_data.get("items", [])
        if items:
            for i, item in enumerate(items, 1):
                st.markdown(f"""
                - **Item {i}**: {item.get('bag_type_raw', 'Unknown')}
                  - Bag Type: {item.get('bag_type', 'Unknown')}
                  - Quantity: {item.get('quantity', 0)}
                """)
        else:
            st.markdown("- No items found")
        
        st.markdown("**Totals:**")
        st.markdown(f"""
        - **Total Amount**: ${extracted_data.get("total_amount", 0):.2f}
        - **Total Packs**: {extracted_data.get("total_packs", 0)}
        """)
        
        # Show confidence indicator
        confidence = extracted_data.get("extraction_confidence", 0.9)
        confidence_color = "#16A34A" if confidence >= 0.9 else "#F59E0B" if confidence >= 0.8 else "#DC2626"
        st.markdown(f"""
        <div style="background-color: #F3F4F6; padding: 10px; border-radius: 5px; margin-top: 10px;">
            <strong>Extraction Confidence:</strong> 
            <span style="color: {confidence_color}; font-weight: bold;">{confidence:.0%}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Show extraction notes if any
        if extracted_data.get("requires_review"):
            st.warning(f"⚠️ **Review Needed**: {extracted_data.get('extraction_notes', 'Check data manually')}")
    
    st.markdown("")
    
    # Metrics row - calculate from all available data
    all_extractions = {}
    for receipt_file in receipt_files:
        all_extractions[receipt_file] = get_sample_receipt_data(receipt_file)
    
    avg_confidence = sum(d.get("extraction_confidence", 0) for d in all_extractions.values()) / len(all_extractions) if all_extractions else 0.94
    total_items = sum(len(d.get("items", [])) for d in all_extractions.values())
    
    st.markdown("#### Extraction Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg. Confidence", f"{avg_confidence:.1%}", delta="High")
    with col2:
        st.metric("Total Items Extracted", str(total_items), delta=None)
    with col3:
        st.metric("Receipts Processed", str(len(all_extractions)), delta=None)
    
    st.markdown("""
    > **Technology**: Using **Google Gemini 2.5 Flash**, we extract structured data from 
    > variable-format receipts (PDF, images, DOCX). The AI handles different layouts, fonts, 
    > and formats from various vendors, normalizing everything into a consistent schema for processing.
    """)



# ============================================================================
# Tab 3: Priority Scoring
# ============================================================================

with tab3:
    st.header("🎯 Priority Scoring")
    st.markdown("*Dynamic scoring system to prioritize deliveries*")
    
    # Explanation
    with st.expander("📖 How Priority Scoring Works", expanded=True):
        st.markdown("""
        Each order receives a **priority score** based on four weighted components:
        
        | Component | Description | Weight |
        |-----------|-------------|--------|
        | **Urgency** | Days until delivery deadline (overdue orders get bonus) | 40% |
        | **Payment** | Order value × payment status multiplier | 25% |
        | **Client** | Client type (Star > New > Regular > Occasional) | 20% |
        | **Age** | Days since order was placed | 15% |
        
        **Formula**: `Priority = Σ (Component Score × Weight)`
        
        **Mandatory orders** receive priority score of **999,999** to ensure inclusion.
        """)
    
    # Interactive weight adjustment
    st.markdown("#### Adjust Scoring Weights")
    st.markdown("*Move the sliders to see how different weights affect priorities*")
    
    # Reset button
    reset_col, _ = st.columns([1, 3])
    with reset_col:
        if st.button("Reset to Default", use_container_width=True):
            st.session_state.w_urgency = default_weights.get("urgency", 0.4)
            st.session_state.w_payment = default_weights.get("payment", 0.25)
            st.session_state.w_client = default_weights.get("client", 0.2)
            st.session_state.w_age = default_weights.get("age", 0.15)
            st.rerun()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        w_urgency = st.slider(
            "Urgency Weight",
            min_value=0.0,
            max_value=1.0,
            value=default_weights.get("urgency", 0.4),
            step=0.05,
            key="w_urgency",
        )
    
    with col2:
        w_payment = st.slider(
            "Payment Weight",
            min_value=0.0,
            max_value=1.0,
            value=default_weights.get("payment", 0.25),
            step=0.05,
            key="w_payment",
        )
    
    with col3:
        w_client = st.slider(
            "Client Weight",
            min_value=0.0,
            max_value=1.0,
            value=default_weights.get("client", 0.2),
            step=0.05,
            key="w_client",
        )
    
    with col4:
        w_age = st.slider(
            "Age Weight",
            min_value=0.0,
            max_value=1.0,
            value=default_weights.get("age", 0.15),
            step=0.05,
            key="w_age",
        )
    
    # Weight sum check
    total_weight = w_urgency + w_payment + w_client + w_age
    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"⚠️ Weights sum to {total_weight:.2f}. Ideally should sum to 1.0")
    else:
        st.success("✓ Weights sum to 1.0")
    
    # Recalculate with custom weights
    custom_weights = {
        "urgency": w_urgency,
        "payment": w_payment,
        "client": w_client,
        "age": w_age,
    }
    
    # Check if weights changed from default
    weights_changed = any(
        abs(custom_weights[k] - default_weights.get(k, 0)) > 0.01
        for k in custom_weights
    )
    
    if weights_changed:
        st.info("📊 Scores recalculated with custom weights (in memory only)")
    
    # Recalculate priorities
    scored_orders = recalculate_priorities(orders_df, custom_weights, clients_df)
    
    st.markdown("")
    
    # Score breakdown and comparison
    col_radar, col_compare = st.columns(2)
    
    with col_radar:
        st.markdown("#### Score Breakdown")
        
        if not scored_orders.empty:
            order_options = scored_orders["order_id"].tolist()
            selected_order = st.selectbox(
                "Select an order to analyze",
                options=order_options,
                index=0,
                key="radar_order",
            )
            
            radar_fig = create_priority_radar_chart(
                selected_order, scored_orders, custom_weights
            )
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.info("No orders available for analysis")
    
    with col_compare:
        st.markdown("#### Priority Comparison")
        
        if not scored_orders.empty and len(scored_orders) >= 3:
            # Get high, medium, low priority orders (excluding mandatory)
            non_mandatory = scored_orders[scored_orders["is_mandatory"] == False].copy()
            
            if len(non_mandatory) >= 3:
                sorted_orders = non_mandatory.sort_values("calculated_priority", ascending=False)
                high_order = sorted_orders.iloc[0]["order_id"]
                mid_idx = len(sorted_orders) // 2
                mid_order = sorted_orders.iloc[mid_idx]["order_id"]
                low_order = sorted_orders.iloc[-1]["order_id"]
                
                # Create comparison chart with proper labels
                compare_fig = create_priority_comparison_chart(
                    scored_orders, 
                    [high_order, mid_order, low_order],
                    priority_labels=["High Priority", "Medium Priority", "Low Priority"],
                    weights=custom_weights,
                )
                st.plotly_chart(compare_fig, use_container_width=True)
            else:
                st.info("Not enough non-mandatory orders for comparison")
        else:
            st.info("Not enough orders for comparison")
    
    # Top priority orders table
    st.markdown("#### Top Priority Orders")
    
    if not scored_orders.empty:
        # Prepare display dataframe
        display_df = scored_orders.copy()
        display_df["client_type"] = display_df.apply(
            lambda row: get_client_type_label(
                clients_df[clients_df["client_id"] == row["client_id"]]["is_star_client"].values[0]
                if not clients_df[clients_df["client_id"] == row["client_id"]].empty
                else False,
                clients_df[clients_df["client_id"] == row["client_id"]]["is_new_client"].values[0]
                if not clients_df[clients_df["client_id"] == row["client_id"]].empty
                else False,
            ),
            axis=1,
        )
        
        # Sort and select columns
        top_orders = display_df.sort_values("calculated_priority", ascending=False).head(10)
        
        show_columns = [
            "order_id",
            "client_name",
            "calculated_priority",
            "urgency_score",
            "payment_score",
            "client_type",
            "age_score",
            "is_mandatory",
        ]
        
        st.dataframe(
            top_orders[show_columns].rename(
                columns={
                    "order_id": "Order ID",
                    "client_name": "Client",
                    "calculated_priority": "Priority Score",
                    "urgency_score": "Urgency",
                    "payment_score": "Payment",
                    "client_type": "Client Type",
                    "age_score": "Age",
                    "is_mandatory": "Mandatory",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    
    # Mandatory orders callout
    mandatory_orders = scored_orders[scored_orders["is_mandatory"] == True]
    
    if not mandatory_orders.empty:
        st.markdown("#### Mandatory Orders")
        st.warning(f"""
        **{len(mandatory_orders)} order(s) marked as mandatory** - These MUST be included in the next dispatch.
        
        Mandatory orders receive a priority score of **999,999** to ensure they are always selected 
        first by the optimization algorithm.
        """)
        
        st.dataframe(
            mandatory_orders[["order_id", "client_name", "total_pallets", "zone_id"]].rename(
                columns={
                    "order_id": "Order ID",
                    "client_name": "Client",
                    "total_pallets": "Pallets",
                    "zone_id": "Zone",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================================
# Tab 4: Order Selection (Dispatch Generator)
# ============================================================================

with tab4:
    st.header("📦 Order Selection")
    st.markdown("*Knapsack optimization to build dispatch candidates*")
    
    # Explanation
    with st.expander("📖 How Order Selection Works", expanded=True):
        st.markdown("""
        The system uses **multiple strategies** to generate dispatch candidates, treating this as a 
        **knapsack optimization problem**:
        
        - **Constraint**: Truck capacity of **8 pallets** (flexible 7-9 range)
        - **Objective**: Maximize total priority while respecting capacity
        
        **Strategies employed**:
        1. **Greedy Priority** - Select highest priority orders first until capacity is reached
        2. **Greedy Efficiency** - Maximize priority-per-pallet ratio for better value utilization
        3. **Greedy Best Fit** - Fill capacity by finding best-fitting combinations of orders
        4. **Dynamic Programming (Optimal)** - Globally optimal knapsack solution (computationally expensive)
        5. **Zone-Based** - Prioritize single-zone dispatches to minimize travel distance
        6. **Zone Spillover** - Primary zone prioritization with overflow to secondary zones
        7. **Mandatory First** - Ensure mandatory orders are included, then optimize remaining capacity
        8. **Greedy Nearest Neighbor** - Select orders by proximity to mandatory orders
        
        Each strategy may produce different candidates, offering trade-offs between 
        priority maximization and route efficiency.
        
        **Subset-Based Approach**:
        Beyond applying strategies to the full order set, the system generates **random subsets** 
        of orders (each containing a mix of mandatory and non-mandatory orders) and applies **all 8 strategies** 
        to each subset independently. This creates diverse alternative solutions by exploring different 
        combinations:
        
        - Random subsets from **all pending orders** (including mandatory)
        - Random subsets from **non-mandatory orders only**
        - Random subsets focused on **top-priority orders** (with and without mandatory)
        
        Multiple subsets × multiple strategies = exponentially more dispatch candidates to evaluate, 
        increasing the likelihood of finding optimal or near-optimal solutions.
        """)
    
    if dispatch_candidates:
        # Toggle for mandatory orders
        include_mandatory = st.checkbox(
            "📌 Include Mandatory Orders",
            value=False,
            help="Mandatory orders have priority scores of ~999,999. Uncheck to see cleaner comparisons.",
            key="include_mandatory_orders",
        )
        
        # Filter candidates based on toggle
        filtered_candidates = filter_candidates_by_mandatory(dispatch_candidates, include_mandatory)
        candidates_df_all = get_candidates_summary_df(dispatch_candidates, include_mandatory=True)
        candidates_df_filtered = get_candidates_summary_df(dispatch_candidates, include_mandatory=include_mandatory)
        
        st.markdown("")
        
        # KPI cards - split by mandatory status
        with st.container(border=False):
            st.markdown('<div style="background-color: #EDE9FE; padding: 20px; border-radius: 10px;">', unsafe_allow_html=True)
            
            if include_mandatory:
                st.markdown("#### 📊 KPIs (All Candidates)")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Candidates", len(candidates_df_all))
                with col2:
                    best_util = candidates_df_all["utilization_pct"].max()
                    st.metric("Best Utilization", f"{best_util:.1f}%")
                with col3:
                    st.metric("Max Priority", f"{candidates_df_all['total_priority'].max():,.0f}")
                with col4:
                    mandatory_count = len(candidates_df_all[candidates_df_all["has_mandatory"] == True])
                    st.metric("With Mandatory", mandatory_count)
            else:
                st.markdown("#### 📊 KPIs (Excluding Mandatory Orders)")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Candidates", len(candidates_df_filtered))
                with col2:
                    if not candidates_df_filtered.empty:
                        best_util = candidates_df_filtered["utilization_pct"].max()
                        st.metric("Best Utilization", f"{best_util:.1f}%")
                    else:
                        st.metric("Best Utilization", "N/A")
                with col3:
                    if not candidates_df_filtered.empty:
                        best_priority = candidates_df_filtered["total_priority"].max()
                        st.metric("Best Priority", f"{best_priority:,.0f}")
                    else:
                        st.metric("Best Priority", "N/A")
                with col4:
                    st.metric("Excluding", f"{len(candidates_df_all) - len(candidates_df_filtered)} mandatory")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("")
        
        # Strategy comparison (use filtered candidates)
        col_strat, col_compare = st.columns(2)
        
        with col_strat:
            st.markdown("#### Strategy Performance")
            strategy_fig = create_strategy_comparison_chart(filtered_candidates)
            st.plotly_chart(strategy_fig, use_container_width=True)
        
        with col_compare:
            st.markdown("#### Priority vs Utilization")
            comparison_fig = create_candidate_comparison_chart(filtered_candidates)
            st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Candidates table - split into two sections if we have both types
        st.markdown("#### Dispatch Candidates")
        
        # Define display columns for the table
        display_columns = {
            "rank": "#",
            "candidate_id": "Candidate ID",
            "strategy": "Strategy",
            "order_count": "Orders",
            "total_pallets": "Pallets",
            "utilization_pct": "Utilization %",
            "total_priority": "Total Priority",
            "zones": "Zones",
            "is_single_zone": "Single Zone",
        }
        
        if include_mandatory:
            # Split into two tabs: with and without mandatory
            tab_non_mand, tab_mand = st.tabs(["🔹 Standard Orders", "⚠️ With Mandatory"])
            
            with tab_non_mand:
                non_mand_df = candidates_df_all[candidates_df_all["has_mandatory"] == False]
                if not non_mand_df.empty:
                    st.markdown(f"*{len(non_mand_df)} candidates ranked by priority (range: {non_mand_df['total_priority'].min():.0f} - {non_mand_df['total_priority'].max():.0f})*")
                    st.dataframe(
                        non_mand_df.rename(columns=display_columns)[list(display_columns.values())],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No candidates without mandatory orders")
            
            with tab_mand:
                mand_df = candidates_df_all[candidates_df_all["has_mandatory"] == True]
                if not mand_df.empty:
                    st.warning(f"⚠️ {len(mand_df)} candidates include mandatory orders with priority ~999,999 each")
                    st.dataframe(
                        mand_df.rename(columns=display_columns)[list(display_columns.values())],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No candidates with mandatory orders")
        else:
            # Show only non-mandatory
            st.dataframe(
                candidates_df_filtered.rename(columns=display_columns)[list(display_columns.values())],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No dispatch candidates found. Run the order selector notebook first.")


# ============================================================================
# Tab 5: Route Optimization
# ============================================================================

with tab5:
    st.header("🚚 Route Optimization")
    st.markdown("*Vehicle Routing Problem with Time Windows (VRPTW)*")
    
    # Explanation
    with st.expander("📖 How Route Optimization Works", expanded=True):
        st.markdown(f"""
        Once dispatch candidates are generated, the system optimizes the **delivery route** for each:
        
        - **Algorithm**: Google OR-Tools (CP-SAT solver)
        - **Problem Type**: Vehicle Routing Problem with Time Windows (VRPTW)
        - **Depot Location**: {DEPOT_NAME} ({DEPOT_LAT:.4f}, {DEPOT_LON:.4f})
        - **Objective**: Minimize total travel distance while respecting time windows
        
        The optimizer finds the best sequence of stops, considering:
        - Distance between locations
        - Client time windows (when they accept deliveries)
        - Service time at each stop (15 minutes per delivery)
        
        Routes are compared against a **nearest-neighbor baseline** to measure improvement.
        """)
    
    if ranked_dispatches:
        # Filter for demo (exclude extreme mandatory-only candidates)
        demo_dispatches = [
            d for d in ranked_dispatches
            if d.get("total_priority", 0) < 1500000 or d.get("candidate_id") in [
                "DISP-20260120-GREEDY-B27D",
                "DISP-20260120-GREEDY-B86C",
            ]
        ]
        
        if not demo_dispatches:
            demo_dispatches = ranked_dispatches[:10]
        
        # Ranked dispatch table
        st.markdown("#### Ranked Dispatch Candidates")
        st.markdown("""
        *Candidates ranked by combined score considering priority, distance, and utilization. 
        Higher priority-per-km indicates better efficiency.*
        """)
        
        dispatch_summary_df = get_dispatch_summary_df(demo_dispatches[:15])
        
        st.dataframe(
            dispatch_summary_df.rename(
                columns={
                    "rank": "Rank",
                    "candidate_id": "Candidate ID",
                    "strategy": "Strategy",
                    "order_count": "Orders",
                    "total_priority": "Priority",
                    "zones": "Zones",
                    "has_mandatory": "Mandatory",
                    "distance_km": "Distance (km)",
                    "duration_min": "Duration (min)",
                    "priority_per_km": "Priority/km",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        
        st.markdown("")
        
        # Priority vs Distance scatter
        st.markdown("#### Priority vs Distance Trade-off")
        
        # Toggle for including mandatory in scatter
        include_mandatory_scatter = st.checkbox(
            "📌 Include Mandatory Orders in Chart",
            value=False,
            help="Include dispatches with mandatory orders (priority ~999,999+)",
            key="include_mandatory_scatter",
        )
        
        st.markdown("*Identify Pareto-efficient dispatches - high priority with low distance. Bubble size = order count.*")
        
        scatter_fig = create_priority_vs_distance_chart(demo_dispatches, include_mandatory=include_mandatory_scatter)
        st.plotly_chart(scatter_fig, use_container_width=True)
        
        st.markdown("")
        
        # Top candidates detail
        st.markdown("#### Top 3 Candidates Detail")
        
        top_3 = demo_dispatches[:3]
        
        cols = st.columns(3)
        for i, dispatch in enumerate(top_3):
            with cols[i]:
                cid = dispatch.get("candidate_id", "")
                with st.expander(f"#{i+1}: {cid[-8:]}", expanded=True):
                    st.markdown(f"""
                    **Strategy**: {dispatch.get('strategy', 'N/A')}  
                    **Orders**: {dispatch.get('order_count', 0)}  
                    **Pallets**: {dispatch.get('total_pallets', 0):.1f}  
                    **Priority**: {dispatch.get('total_priority', 0):,.0f}  
                    **Distance**: {dispatch.get('total_distance_km', 0):.1f} km  
                    **Duration**: {dispatch.get('total_duration_minutes', 0)} min
                    """)
                    
                    zones = dispatch.get("zones", [])
                    st.markdown(f"**Zones**: {', '.join(zones)}")
        
        st.markdown("")
        
        # Interactive route map
        st.markdown("#### Interactive Route Map")
        
        # Dispatch selector
        dispatch_options = {
            d.get("candidate_id", ""): d for d in demo_dispatches
        }
        
        # Find default selection
        default_id = "DISP-20260120-DP_OPT-SUB-6138"
        if default_id not in dispatch_options:
            default_id = list(dispatch_options.keys())[0] if dispatch_options else None
        
        selected_dispatch_id = st.selectbox(
            "Select dispatch to visualize",
            options=list(dispatch_options.keys()),
            index=list(dispatch_options.keys()).index(default_id) if default_id in dispatch_options else 0,
            key="route_select",
        )
        
        if selected_dispatch_id:
            selected_dispatch = dispatch_options[selected_dispatch_id]
            
            # Larger map layout - give more space to map
            col_map, col_info = st.columns([5, 3])
            
            with col_map:
                route_map = create_route_map(selected_dispatch)
                st_folium(route_map, width=None, height=600, returned_objects=[], use_container_width=True)
            
            with col_info:
                route = selected_dispatch.get("route", {})
                
                st.markdown("#### Route Summary")
                
                # Add priority and pallets to summary
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Total Distance", f"{route.get('total_distance_km', 0):.1f} km")
                    st.metric("Stops", len(route.get("stops", [])) - 1)  # Exclude depot
                with col_m2:
                    st.metric("Total Duration", f"{route.get('total_duration_minutes', 0)} min")
                    st.metric("Priority", f"{selected_dispatch.get('total_priority', 0):,.0f}")
                
                # Add pallets info
                st.metric("Total Pallets", f"{selected_dispatch.get('total_pallets', 0):.1f}")
                
                st.markdown("")
                st.markdown("#### Visit Sequence")
                
                stops_df = get_route_stops_df(selected_dispatch, dispatch_candidates)
                if not stops_df.empty:
                    # Include pallets and remaining columns
                    display_cols = ["#", "Location", "Arrival", "Drop", "Remaining", "Km"]
                    st.dataframe(
                        stops_df.rename(
                            columns={
                                "sequence": "#",
                                "location": "Location",
                                "arrival_time": "Arrival",
                                "distance_from_prev_km": "Km",
                                "pallets_to_drop": "Drop",
                                "pallets_remaining": "Remaining",
                            }
                        )[display_cols],
                        use_container_width=True,
                        hide_index=True,
                        height=360,
                    )
        
        st.markdown("")
        
        # Algorithm comparison
        st.markdown("#### Algorithm Comparison")
        
        with st.expander("🔬 OR-Tools vs Nearest Neighbor Baseline"):
            st.markdown("""
            The OR-Tools solver typically achieves **15-25% distance reduction** compared to 
            a simple nearest-neighbor heuristic.
            
            | Metric | OR-Tools | Nearest Neighbor | Improvement |
            |--------|----------|------------------|-------------|
            | Distance | 50.9 km | 62.4 km | **-18.4%** |
            | Duration | 149 min | 178 min | **-16.3%** |
            
            *Values shown for the top dispatch candidate*
            
            The optimization considers:
            - Global route optimization (not just local greedy choices)
            - Time window constraints
            - Service time at each location
            - Return to depot
            """)
    
    else:
        st.info("No ranked dispatches found. Run the route optimizer notebook first.")


# ============================================================================
# Tab 6: About
# ============================================================================

with tab6:
    st.header("ℹ️ About This Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Project Overview
        
        **Delivery Optimization System** is a portfolio project demonstrating 
        end-to-end data science and operations research capabilities.
        
        Built with real-world constraints in mind:
        - Variable-format document processing
        - Multi-objective optimization
        - Geographic routing with time windows
        - Interactive visualization
        
        ### Pipeline Phases
        
        1. **Data Setup** - Zones, localities, clients, products
        2. **Receipt Extraction** - AI-powered document parsing
        3. **Priority Scoring** - Weighted multi-factor ranking
        4. **Order Selection** - Knapsack optimization
        5. **Route Optimization** - VRPTW with OR-Tools
        """)
    
    with col2:
        st.markdown("""
        ### Technologies Used
        
        | Category | Technology |
        |----------|------------|
        | Language | Python 3.13 |
        | Web App | Streamlit |
        | Database | SQLite + SQLAlchemy |
        | AI/LLM | Google Gemini 2.5 Flash |
        | Optimization | OR-Tools (CP-SAT) |
        | Visualization | Plotly, Folium |
        | Data Validation | Pydantic |
        | Package Manager | Poetry |
        
        ### Links
        """)
        
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/ssantioviedo/Eco-Bags-Delivery-Optimizer)")
        st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://linkedin.com/in/ssantioviedo)")
    
    st.markdown("")
    st.markdown("---")
    st.caption("© 2026 Delivery Optimization System | Built with Streamlit")
