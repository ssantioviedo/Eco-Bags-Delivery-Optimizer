# Project Prompt: Delivery Optimization System

## Context

You are building a delivery optimization system for an eco-friendly bag factory located in Buenos Aires, Argentina. The factory produces reusable bags and needs to optimize their delivery operations.

## Business Constraints

### Factory Operations
- Single fixed depot (factory location)
- Single truck for all deliveries
- Truck capacity: 8 pallets
- Average deliveries per trip: ~10 orders
- Products are packs of 250 bags, shaped like bricks/pillows
- 3 main bag types + occasional special types
- Work in pallets as unit of measure (not actual volume)

### Sales Process
- ~10 different salespeople, each with their own receipt format
- Receipts can be DOCX or Excel files
- Receipt structure varies significantly between salespeople
- New salespeople may be added over time
- Delivery address may differ from client's billing address (e.g., shipping to a transport company)

### Geographic Zones
- CABA (Buenos Aires City)
- NORTH_ZONE (Vicente López, San Isidro, Tigre, Pilar, etc.)
- SOUTH_ZONE (Avellaneda, Quilmes, Lanús, Lomas de Zamora, etc.)
- WEST_ZONE (Morón, La Matanza, Moreno, Merlo, etc.)
- Prefer dispatches with orders from the same zone
- Avoid mixing incompatible zones (e.g., NORTH_ZONE + SOUTH_ZONE)

### Delivery Rules
- All orders should be delivered within 7 days
- Paid orders: deliver within 3 days
- Star clients: higher priority
- New clients: higher priority (good first impression)
- User can mark orders as mandatory (must go out today)
- Clients may have delivery time windows (e.g., 10:00-14:00)

## System Requirements

### Module 1: Receipt Extraction (Document AI)
- Extract data from DOCX/Excel receipts using LLM (OpenAI GPT-4o)
- Handle variable formats from different salespeople
- Extract: delivery address, items (product type + quantity), issue date, billing data, total amount
- Validate extracted data with Pydantic schemas
- Geocode delivery addresses using Geopy + Nominatim
- Assign geographic zone based on locality
- Cache geocoding results to avoid repeated API calls
- Handle extraction failures gracefully with fallback values

### Module 2: Priority Scoring System
- Calculate priority score for each order based on:
  - URGENCY: days until delivery deadline (weight: 0.40)
  - PAYMENT: paid/partial/pending status (weight: 0.25)
  - CLIENT_TYPE: star/new/regular/occasional (weight: 0.20)
  - AGE: days since order was placed (weight: 0.15)
- Weights should be configurable
- Mandatory orders get infinite priority (must be included)
- Score formula: PRIORITY = w1×URGENCY + w2×PAYMENT + w3×CLIENT + w4×AGE

### Module 3: Dispatch Generator (Optimization)
- Generate K dispatch candidates (not just one optimal solution)
- Each dispatch is a subset of orders that fit in the truck (≤8 pallets)
- Optimization objectives:
  - Maximize total priority of included orders
  - Maximize truck utilization (closer to 100% = better)
  - Minimize zone dispersion (penalize mixing zones)
- Constraints:
  - Total pallets ≤ 8
  - All mandatory orders must be included
- Use OR-Tools CP-SAT solver
- Output multiple candidates for user to choose from
- Show combinations of dispatches that together achieve ~100% utilization

### Module 4: Route Optimization (TSP)
- For each dispatch candidate, optimize delivery sequence
- Use simple TSP (not full VRP, orders are already selected)
- Calculate distances using Haversine (straight line, sufficient for ranking)
- Estimate total route distance (km) and time (minutes)
- Use OR-Tools TSP or python-tsp library
- Order by time windows when applicable

### Output Requirements
- List of dispatch candidates with metrics:
  - Orders included
  - Total pallets
  - Utilization percentage
  - Priority score
  - Zone composition (% per zone)
  - Estimated route distance (km)
- Interactive map showing route (Folium)
- Allow user to compare and select dispatch
- Export selected dispatch as JSON

## Technical Stack

### Core
- Python 3.11+
- Poetry for dependency management
- SQLite database (via SQLAlchemy)
- Pydantic v2 for data validation

### Document Processing
- OpenAI API (GPT-4o) for extraction
- python-docx for DOCX parsing
- openpyxl for Excel parsing

### Optimization
- OR-Tools (CP-SAT for selection, TSP for routing)
- NumPy for matrix operations

### Geographic
- Geopy for geocoding (Nominatim provider)
- Haversine for distance calculations
- Folium for map visualization

### Visualization
- Plotly Express / Plotly Graph Objects for all charts
- DO NOT use matplotlib or seaborn
- Folium for geographic maps only

### Development
- JupyterLab for notebooks
- One notebook per module/phase

## Database Schema

### Tables Required

**zones**
- zone_id (TEXT, PK): CABA, NORTH_ZONE, SOUTH_ZONE, WEST_ZONE
- name (TEXT)
- color (TEXT): for visualization

**localities**
- locality_id (TEXT, PK)
- name (TEXT)
- zone_id (TEXT, FK)
- latitude (REAL)
- longitude (REAL)

**clients**
- client_id (TEXT, PK)
- business_name (TEXT)
- tax_id (TEXT)
- billing_address (TEXT)
- zone_id (TEXT, FK)
- latitude (REAL)
- longitude (REAL)
- time_window_start (TIME, nullable)
- time_window_end (TIME, nullable)
- is_star_client (BOOLEAN)
- is_new_client (BOOLEAN)
- first_order_date (DATE)

**products**
- product_id (TEXT, PK)
- name (TEXT)
- bag_type (TEXT): A, B, C, special
- packs_per_pallet (INTEGER)
- description (TEXT)

**orders**
- order_id (TEXT, PK)
- client_id (TEXT, FK)
- issue_date (DATE)
- delivery_deadline (DATE)
- delivery_address (TEXT)
- delivery_latitude (REAL)
- delivery_longitude (REAL)
- delivery_zone_id (TEXT, FK)
- total_amount (REAL)
- payment_status (TEXT): pending, partial, paid
- is_mandatory (BOOLEAN)
- total_pallets (REAL)
- priority_score (REAL)
- status (TEXT): pending, assigned, delivered
- created_at (TIMESTAMP)

**order_items**
- item_id (INTEGER, PK, autoincrement)
- order_id (TEXT, FK)
- product_id (TEXT, FK)
- quantity_packs (INTEGER)
- pallets (REAL)

**dispatches**
- dispatch_id (TEXT, PK)
- dispatch_date (DATE)
- total_pallets (REAL)
- utilization_percentage (REAL)
- total_priority_score (REAL)
- estimated_distance_km (REAL)
- estimated_time_min (INTEGER)
- status (TEXT): candidate, confirmed, completed
- created_at (TIMESTAMP)

**dispatch_orders**
- dispatch_id (TEXT, FK)
- order_id (TEXT, FK)
- visit_sequence (INTEGER)
- estimated_arrival_time (TIME)
- PRIMARY KEY (dispatch_id, order_id)

**geocoding_cache**
- address_hash (TEXT, PK)
- raw_address (TEXT)
- latitude (REAL)
- longitude (REAL)
- locality (TEXT)
- zone_id (TEXT)
- confidence (TEXT): high, medium, low
- cached_at (TIMESTAMP)

**processed_receipts**
- receipt_id (TEXT, PK)
- source_file (TEXT)
- processed_at (TIMESTAMP)
- raw_extraction (JSON)
- generated_order_id (TEXT, FK)
- extraction_confidence (REAL)

## Project Structure
```text
delivery-optimizer/
├── README.md
├── AGENTS.md
├── pyproject.toml
├── .gitignore
├── .env.example
│
├── notebooks/
│ ├── 01_base_data_setup.ipynb
│ ├── 02_receipt_extraction.ipynb
│ ├── 03_priority_scoring.ipynb
│ ├── 04_dispatch_generator.ipynb
│ ├── 05_routing_visualization.ipynb
│ └── 06_full_demo.ipynb
│
├── src/
│ ├── init.py
│ ├── extraction.py
│ ├── scoring.py
│ ├── optimizer.py
│ ├── routing.py
│ ├── geo.py
│ ├── schemas.py
│ ├── database.py
│ └── pipeline.py
│
├── data/
│ ├── raw/
│ │ └── receipts/
│ ├── processed/
│ │ └── delivery.db
│ └── geo/
│ ├── zones.json
│ └── localities.json
│
├── output/
│ ├── dispatches/
│ └── maps/
│
└── docs/
├── architecture.md
└── data_dictionary.md
```

## Geographic Reference Data

### zones.json structure
```json
{
  "CABA": {
    "name": "Buenos Aires City",
    "color": "#FF6B6B"
  },
  "NORTH_ZONE": {
    "name": "North Zone",
    "color": "#4ECDC4"
  },
  "SOUTH_ZONE": {
    "name": "South Zone", 
    "color": "#45B7D1"
  },
  "WEST_ZONE": {
    "name": "West Zone",
    "color": "#96CEB4"
  }
}
```

### localities.json structure
```json
{
  "palermo": {
    "name": "Palermo",
    "zone_id": "CABA",
    "latitude": -34.5875,
    "longitude": -58.4311
  },
  "vicente_lopez": {
    "name": "Vicente López",
    "zone_id": "NORTH_ZONE", 
    "latitude": -34.5257,
    "longitude": -58.4738
  }
}
```

Populate with ~80 localities:

CABA: ~25 main neighborhoods
NORTH_ZONE: ~18 localities
SOUTH_ZONE: ~18 localities
WEST_ZONE: ~18 localities

## Synthetic Data Requirements
Generate realistic synthetic data for portfolio demonstration:

4 zones (defined above)
80 localities with coordinates
4 products (3 main bag types + 1 special)
30 clients distributed across zones
50 historical orders with varied statuses
15 sample receipts (DOCX format) with different structures

## Current Phase: Phase 1 - Base Data Setup

### Deliverables for Phase 1

1. Initialize SQLite database with all tables
1. Create zones.json with 4 zones
1. Create localities.json with ~80 localities and coordinates
1. Populate products table (4 products)
1. Generate synthetic clients (30)
1. Generate synthetic orders (50)
1. Create notebook 01_base_data_setup.ipynb demonstrating:
    - Database initialization
    - Data loading
    - Basic exploratory queries
    - Plotly visualization of orders by zone
    - Folium map showing client distribution

### Code Conventions

1. All code in English (zero Spanish)
1. Type hints on all functions
1. Google-style docstrings
1. PEP 8 compliant
1. Max line length: 100 characters
1. Use Pydantic for all schemas
1. Use Plotly for all charts (never matplotlib)
1. Use Folium for maps