# Eco-Bags Delivery Optimization System
### Document AI → Priority Scoring → Dispatch Selection → Route optimization

End-to-end applied optimization project focused on turning messy sales receipts into actionable daily dispatch plans for an eco-friendly reusable bag factory in Buenos Aires, maximizing service level and truck utilization while respecting real operational constraints.

Tech stack (current): Python · Poetry · SQLite (SQLAlchemy) · Pydantic v2 · Google Gemini (google-genai)
· Geopy (Nominatim) · Plotly · Folium · Streamlit (showcase viewer)

Status: In progress · Foundations + PDF receipt extraction implemented · Optimization/routing planned


---


## Motivation
Delivery planning is where small operational frictions compound into real business cost:

- Sales orders arrive in heterogeneous formats (DOCX/Excel, per-salesperson styles)
- Decisions must balance urgency, client priority, payment status, and fairness
- A single truck with limited capacity forces trade-offs every day
- Geography matters: zone mixing can waste time, increase late deliveries, and create operational chaos

This project builds a reusable system that standardizes incoming orders, validates and enriches them (geocoding + zoning), and lays the groundwork for a complete optimization pipeline.


---


## Problem Statement
Given incoming sales receipts (PDF/DOCX/Excel) and operational constraints:

1. Extract structured order data from inconsistent receipt templates (LLM-based)
2. Validate & persist it (schemas + database)
3. Enrich orders with geocoding and zone assignment
4. (Next) Prioritize and generate candidate dispatches under truck capacity constraints
5. (Next) Optimize routing and publish dispatch plans


---


## Operational Constraints (Real-World)
- Single depot (factory in Buenos Aires)
- One truck, capacity 8 pallets
- Typical dispatch: ~10 orders
- Prefer dispatches within the same zone
- Avoid incompatible mixing (e.g., NORTH_ZONE + SOUTH_ZONE)
- Delivery rules:
  - All orders ≤ 7 days
  - Paid orders ≤ 3 days
  - Star clients higher priority
  - New clients higher priority
  - Mandatory orders must go out today
  - Optional time windows (e.g., 10:00–14:00)


---


## Data & System Design
Instead of assuming clean tables, the system is designed around a realistic pipeline:

- Input: DOCX/Excel receipts with inconsistent structures
- Core entities: zones, localities, clients, products, orders, order items
- Enrichment: geocoding → coordinates → zone assignment
- Persistence: SQLite database for traceability and auditability
- Outputs (current): normalized orders stored in DB + extraction artifacts


---


## Methodology (Phase-Based, Production-Oriented)

### Phase 1 — Base Data Setup (Implemented)
Goal: Create the operational backbone (database + reference geography + synthetic portfolio data).

- SQLite database initialized via SQLAlchemy
- Reference datasets:
  - zones.json (CABA, NORTH_ZONE, SOUTH_ZONE, WEST_ZONE + colors)
  - localities.json (~80 localities with coordinates + zone mapping)
- Seed data for portfolio demonstration:
  - products (3 main types + 1 special)
  - synthetic clients
  - synthetic historical orders
- Notebook: notebooks/01_base_data_setup.ipynb
  - basic validation queries
  - Plotly charts (orders by zone, etc.)
  - Folium map of client distribution


### Phase 2 — Receipt Extraction (Document AI) (Implemented)
Goal: Convert unstructured receipts into a normalized order record.

- PDF ingestion (uploaded to Gemini)
- LLM-based extraction with Google Gemini (via `google-genai`)
- Output enforced with Pydantic v2 schemas (type safety + required fields)
- Handles variability across vendors and document layouts
- Extracts:
  - delivery address (may differ from billing address)
  - billing data
  - issue date
  - items (product type + quantities)
  - total amount (when present)
- Geocoding via Geopy + Nominatim
- Cached geocoding results in the SQLite table `geocoding_cache` to reduce repeated calls
- Extraction traceability:
  - raw extraction JSON stored in the SQLite table `processed_receipts`
  - confidence fields available for monitoring / QA

Notebook: notebooks/02_receipt_extraction.ipynb

Showcase:
- Notebook: notebooks/demo_showcase.ipynb (generates Folium maps)
- Streamlit viewer: output/streamlit_apps/showcase_map.py (renders the generated HTML map)


---


## Implementation Checklist

### Foundations
- [x] Define database schema (orders, order_items, clients, products, zones, localities, dispatches, caches)
- [x] Initialize SQLite DB via SQLAlchemy
- [x] Create zones.json (4 zones + visualization colors)
- [x] Create localities.json (~80 localities + coordinates + zone mapping)
- [x] Populate products table (3 main + 1 special)
- [x] Generate synthetic clients and historical orders
- [x] Notebook 01: setup + Plotly EDA + Folium client map

### Document Extraction
- [x] Process PDF receipts
- [x] LLM extraction (Gemini) with robust prompting for variable formats
- [x] Pydantic validation + fallback handling for partial failures
- [x] Geocode delivery address (Nominatim)
- [x] Cache geocoding results
- [x] Assign zone based on locality / reference data
- [x] Persist processed receipts + extraction artifacts
- [x] Notebook 02: extraction demo + persisted orders

### Priority & Optimization (Planned)
- [ ] Priority scoring module with configurable weights
- [ ] Mandatory-order enforcement (hard constraint)
- [ ] Dispatch selection (produce K candidate dispatches)
- [ ] Zone dispersion penalties + “incompatible zone” constraints
- [ ] Candidate comparison metrics (utilization, zone mix, priority)

### Routing & Outputs (Planned)
- [ ] TSP route sequencing per candidate dispatch
- [ ] Time window handling (soft/heuristic ordering)
- [ ] Route distance/time estimation
- [ ] Folium route maps per candidate
- [ ] Export confirmed dispatch as JSON
- [ ] Full pipeline demo notebook


---


## Project Structure

```text
Eco-Bags-Delivery-Optimizer/
├── README.md
├── AGENTS.md
├── pyproject.toml
├── poetry.lock
├── .env.example
│
├── src/
│   ├── __init__.py
│   ├── database.py
│   ├── schemas.py
│   ├── geo.py
│   └── extraction.py
│
├── notebooks/
│   ├── 01_base_data_setup.ipynb
│   ├── 02_receipt_extraction.ipynb
│   └── demo_showcase.ipynb
│
├── data/
│   ├── geo/
│   │   ├── zones.json
│   │   └── localities.json
│   ├── raw/
│   │   └── receipts/  # PDF examples
│   └── processed/
│       └── delivery.db
│
└── output/
    ├── dispatches/
    ├── maps/
    └── streamlit_apps/
    └── showcase_map.py
```

---


## How to Run Locally

```bash
git clone https://github.com/ssantioviedo/Eco-Bags-Delivery-Optimizer.git
cd Eco-Bags-Delivery-Optimizer

poetry install

# Notebooks
poetry run jupyter lab

# Streamlit showcase (renders the generated HTML map)
poetry run streamlit run output/streamlit_apps/showcase_map.py
```

Environment:
- Copy .env.example → .env
- Set GEMINI_API_KEY in .env (the real .env is ignored by git)


---


## Next Steps
Immediate next implementation milestones:

- Implement priority scoring (configurable weights)
- Implement dispatch selection (capacity constraint + zone penalties)
- Implement routing (sequencing + distance/time estimation)
- Add export + maps for candidate dispatches


---


## What This Project Demonstrates
- Designing around messy real inputs (documents), not idealized datasets
- Schema-first engineering with validation and auditability (Pydantic + SQLite)
- Geospatial enrichment for operational decision-making (geocoding + zoning)
- A modular path from raw receipts → dispatch optimization → route planning

## Author

**Santiago Oviedo** | *Data Scientist*

🔗 **LinkedIn**: https://linkedin.com/in/ssantioviedo