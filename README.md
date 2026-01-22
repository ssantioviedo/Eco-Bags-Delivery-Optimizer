# Eco-Bags Delivery Optimization System

### Document AI → Priority Scoring → Dispatch Selection → Route Optimization

An end-to-end applied optimization project that transforms messy sales receipts into actionable daily dispatch plans for an eco-friendly reusable bag factory in Buenos Aires. The system maximizes service level and truck utilization while respecting real operational constraints.

**Tech Stack**: Python · Poetry · SQLite (SQLAlchemy) · Pydantic v2 · Google Gemini (AI Extraction) · OR-Tools (Optimization) · Geopy (Geocoding) · Plotly · Folium · Streamlit

---

## Motivation

Delivery planning is where small operational frictions compound into real business cost:

- Sales orders arrive in **heterogeneous formats** (PDF/DOCX/Excel, per-salesperson styles)
- Decisions must balance **urgency, client priority, payment status, and fairness**
- A single truck with **limited capacity** forces trade-offs every day
- **Geography matters**: zone mixing wastes time, increases late deliveries, and creates operational chaos

---

## Problem Statement

Given incoming sales receipts (PDF/DOCX/Excel) and operational constraints:

1. **Extract** structured order data from inconsistent receipt templates (LLM-based)
2. **Validate & Persist** data using strict schemas + database
3. **Enrich** orders with geocoding and zone assignment
4. **Prioritize** orders using weighted scoring (urgency, payment, client type, age)
5. **Generate** candidate dispatches under truck capacity constraints
6. **Optimize** routes and publish dispatch plans with interactive maps

---

## Operational Constraints (Real-World)

| Constraint | Description        |
|------------|-------|
| Depot | Factory in Buenos Aires industrial zone |
| Fleet | Single truck |
| Capacity | 8 pallets per dispatch |
| Typical load | ~10 orders per trip |

**Delivery Rules:**
- All orders must be delivered within **7 days**
- Paid orders prioritized for delivery within **3 days**
- **Star clients** receive higher priority
- **New clients** receive higher priority (relationship building)
- **Mandatory orders** must go out today (hard constraint)
- Prefer **single-zone dispatches** to minimize route dispersion
- Avoid incompatible zone mixing (e.g., NORTH_ZONE + SOUTH_ZONE)

---

## System Architecture

```
┌──────────────┐    ┌──────────────┐    ┌────────────┐    ┌──────────────┐
│ PDF Receipts │──▶│ AI Extraction │──▶│ Validation │──▶│ Priority     │
│ (Variable)   │    │ (Gemini 2.5) │    │ (Pydantic) │    │ Score        │
└──────────────┘    └──────────────┘    └────────────┘    └──────┬───────┘
                                                                 │
┌──────────────┐   ┌──────────────┐   ┌──────────────────────────▼────────┐
│ Route Maps   │◀──│ TSP Optimizer│◀─│ Dispatch Candidates Generator     │
│ (Folium)     │   │ (OR-Tools)   │   │ (Multi-strategy)                  │
└──────────────┘   └──────────────┘   └───────────────────────────────────┘

                     ┌──────────────────────────┐
                     │     SQLite Database      │
                     │ (Shared Persistence)     │
                     └──────────────────────────┘

```

---

## Methodology (5-Phase Pipeline)

### Phase 1 — Base Data Setup 
**Goal**: Create the operational backbone (database + reference geography + synthetic portfolio data).

- SQLite database initialized via SQLAlchemy with full relational schema
- Reference datasets:
  - `zones.json` — 4 zones (CABA, NORTH_ZONE, SOUTH_ZONE, WEST_ZONE) with visualization colors
  - `localities.json` — ~80 localities with coordinates + zone mapping
- Seed data for demonstration:
  - 4 product types (3 main + 1 special)
  - Synthetic clients across all zones
  - Synthetic historical orders
- **Notebook**: [01_base_data_setup.ipynb](notebooks/01_base_data_setup.ipynb)
  - Database validation queries
  - Plotly EDA charts (orders by zone, client distribution)
  - Folium map of client locations

### Phase 2 — Receipt Extraction (Document AI)
**Goal**: Convert unstructured receipts into normalized order records.

- PDF ingestion with image conversion for Gemini processing
- LLM-based extraction using **Google Gemini 2.5 Flash**
- Pydantic v2 schema validation (type safety + required fields)
- Handles variability across vendors and document layouts
- Extracts: delivery address, billing data, issue date, items, totals
- Geocoding via Geopy + Nominatim with persistent SQLite cache
- Extraction artifacts stored in `processed_receipts` table for traceability
- **Notebook**: [02_receipt_extraction.ipynb](notebooks/02_receipt_extraction.ipynb)

### Phase 3 — Priority Scoring
**Goal**: Calculate priority scores to rank orders for dispatch selection.

- Configurable scoring weights via `scoring_weights.json`
- Four scoring components:
  - **Urgency (40%)**: Days until deadline, overdue bonus
  - **Payment (25%)**: Order value × payment status multiplier
  - **Client (20%)**: Star > New > Frequent > Regular > Occasional
  - **Age (15%)**: Days since order placement
- Mandatory orders receive priority score of **999,999** (hard constraint)
- Scores persisted to database for reuse
- **Notebook**: [03_priority_score.ipynb](notebooks/03_priority_score.ipynb)
  - Score distribution analysis
  - Component breakdown visualization
  - Radar charts for order comparison

### Phase 4 — Order Selection (Dispatch Candidates)
**Goal**: Generate optimal dispatch candidates under capacity constraints.

- Multiple selection strategies:
  - **Greedy Efficiency**: Maximize priority/pallet ratio
  - **Greedy Priority**: Maximize total priority
  - **Zone-Based**: Single-zone dispatches (CABA, North, South, West)
  - **DP Optimal**: Dynamic programming for true optimal solution
  - **Mandatory First**: Ensure mandatory orders are included
  - **Best Fit**: Balance utilization and priority
- Zone dispersion penalties discourage mixed-zone dispatches
- Configurable via `order_selector_config.json`
- Outputs: `dispatch_candidates.json`, `dispatch_summary.csv`
- **Notebook**: [04_order_selector.ipynb](notebooks/04_order_selector.ipynb)
  - Strategy comparison charts
  - Utilization vs priority trade-off analysis

### Phase 5 — Route Optimization
**Goal**: Optimize delivery sequences and generate route maps.

- **OR-Tools** constraint solver for TSP/VRP problems
- Multiple solver strategies:
  - Nearest Neighbor (fast heuristic)
  - OR-Tools Guided Local Search (optimal)
  - Christofides-inspired approaches
- Haversine distance calculations for accurate routing
- Persistent route cache in SQLite to avoid recalculation
- Interactive **Folium maps** with:
  - Numbered stop markers
  - Route polylines with distance annotations
  - Zone color coding
  - Depot marker
- Configurable via `routing_config.json`
- Outputs: `ranked_dispatches_with_routes.json`, `top_dispatch.json`, route HTML maps
- **Notebook**: [05_route_optimizer.ipynb](notebooks/05_route_optimizer.ipynb)
  - Solver comparison analysis
  - Route visualization
  - Distance/time estimation

---

## Interactive Showcase (Streamlit)

A multi-page Streamlit application demonstrates the complete pipeline:

[📍 Streamlit App](https://ssantioviedo-eco-bags-delivery-optimizer.streamlit.app/)


**Features**:
- **Data Overview**: Geographic coverage map, KPI dashboard
- **Receipt Extraction**: Live demo of AI extraction with confidence scores
- **Priority Scoring**: Interactive weight adjustment, score recalculation
- **Order Selection**: Strategy comparison, dispatch candidate explorer
- **Route Optimization**: Interactive route maps, solver comparison

---

## Project Structure

```text
Eco-Bags-Delivery-Optimizer/
├── README.md                    # Project documentation
├── AGENTS.md                    # AI agent instructions
├── pyproject.toml               # Poetry dependencies
├── poetry.lock                  # Locked dependencies
├── streamlit_app.py             # Interactive showcase application
│
├── src/                         # Core Python modules
│   ├── __init__.py
│   ├── database.py              # SQLAlchemy models & DatabaseManager
│   ├── schemas.py               # Pydantic validation schemas
│   ├── extraction.py            # LLM-based document parsing (Gemini)
│   ├── geo.py                   # Geocoding and distance utilities
│   ├── scoring.py               # Priority calculation engine
│   ├── order_selector.py        # Dispatch candidate generation
│   ├── routing.py               # TSP/VRP optimization (OR-Tools)
│   ├── app_utils.py             # Streamlit helper functions
│   └── extract_receipts_for_app.py  # Batch extraction utility
│
├── notebooks/                   # Development & demo notebooks
│   ├── 01_base_data_setup.ipynb       # Database + reference data
│   ├── 02_receipt_extraction.ipynb    # AI extraction pipeline
│   ├── 03_priority_score.ipynb        # Scoring analysis
│   ├── 04_order_selector.ipynb        # Dispatch generation
│   ├── 05_route_optimizer.ipynb       # Route optimization
│   └── doc_extractor_demo.ipynb       # Extraction showcase
│
├── data/
│   ├── config/                  # Configuration files
│   │   ├── scoring_weights.json       # Priority weight settings
│   │   ├── order_selector_config.json # Dispatch selection settings
│   │   └── routing_config.json        # Route optimization settings
│   ├── geo/                     # Geographic reference data
│   │   ├── zones.json                 # Zone definitions + colors
│   │   └── localities.json            # Locality coordinates
│   ├── raw/
│   │   └── receipts/            # Input PDF/image receipts
│   └── processed/
│       └── extracted_receipts.json    # Extraction results cache
│
├── output/
│   ├── priority_scores.csv      # Calculated priority scores
│   ├── dispatches/              # Dispatch candidate outputs
│   │   ├── dispatch_candidates.json
│   │   ├── dispatch_summary.csv
│   │   ├── ranked_dispatches_with_routes.json
│   │   └── top_dispatch.json
│   ├── maps/                    # Generated Folium HTML maps
│   │   ├── client_distribution.html
│   │   ├── delivery_locations.html
│   │   ├── top_dispatch_route.html
│   │   ├── top3_routes_comparison.html
│   │   └── best_single_zone_route.html
│   ├── plots/                   # Generated chart images
│   └── streamlit_apps/          # Legacy Streamlit components
│
└── prompts/                     # Development prompts archive
```

---

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/ssantioviedo/Eco-Bags-Delivery-Optimizer.git
cd Eco-Bags-Delivery-Optimizer

# Install dependencies
poetry install

# Run Jupyter notebooks
poetry run jupyter lab

# Launch Streamlit showcase
poetry run streamlit run streamlit_app.py
```

**Environment Setup:**
- Copy `.env.example` → `.env`
- Set `GEMINI_API_KEY` in `.env` (required for receipt extraction)
- The `.env` file is ignored by git for security

---

## What This Project Demonstrates

- **Real-world input handling**: Processing messy documents, not idealized datasets
- **Schema-first engineering**: Validation and auditability with Pydantic + SQLite
- **Geospatial enrichment**: Geocoding + zone assignment for operational decisions
- **Multi-strategy optimization**: Comparing different algorithms for dispatch selection
- **Production patterns**: Caching, configuration files, modular architecture
- **End-to-end pipeline**: From raw receipts → dispatch optimization → route planning

---

## Author

**Santiago Oviedo** | *Data Scientist*

🔗 **LinkedIn**: [linkedin.com/in/ssantioviedo](https://linkedin.com/in/ssantioviedo)
