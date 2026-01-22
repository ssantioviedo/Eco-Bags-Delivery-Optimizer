# AGENTS.md

## Project Overview

Delivery optimization system for an eco-friendly bag factory in Buenos Aires. The system provides a complete end-to-end pipeline:

1. **Document Extraction**: Processes sales receipts (PDF) using Google Gemini AI
2. **Priority Scoring**: Calculates delivery priorities based on configurable weights
3. **Dispatch Selection**: Generates optimal dispatch candidates using multiple strategies
4. **Route Optimization**: Solves TSP/VRP problems using OR-Tools for efficient routing

## Tech Stack

- **Python**: 3.11+
- **Package Manager**: Poetry
- **Database**: SQLite (via SQLAlchemy)
- **Data Validation**: Pydantic v2
- **Optimization**: OR-Tools (TSP/VRP solvers)
- **LLM**: Google Gemini 2.5 Flash (via `google-genai`)
- **Geocoding**: Geopy + Nominatim
- **Visualization**: Plotly (charts) + Folium (maps)
- **Notebooks**: JupyterLab
- **Showcase**: Streamlit

## Code Conventions

### Language
- All code, comments, docstrings, variable names, and data in **English**
- Zero Spanish in codebase

### Style
- Follow PEP 8
- Use type hints for all functions
- Docstrings in Google format
- Max line length: 100 characters

### Visualization
- **DO NOT** use matplotlib or seaborn
- **USE** Plotly Express or Plotly Graph Objects for all charts
- Display Plotly figures with `fig.show()` in notebooks
- Use Folium exclusively for geographic maps

### Data Handling
- Use Pydantic models for all data schemas
- Validate inputs at boundaries
- Use pandas DataFrames for tabular operations
- SQLite for persistence, never raw CSV for storage

### Notebooks
- One notebook per phase/module (01_, 02_, 03_, 04_, 05_)
- Clear markdown headers for sections
- Keep cells focused and short
- Include outputs in commits for demo purposes

## Project Structure

```text
Eco-Bags-Delivery-Optimizer/
├── README.md                    # Project documentation
├── AGENTS.md                    # AI agent instructions (this file)
├── pyproject.toml               # Poetry dependencies
├── streamlit_app.py             # Interactive showcase application
│
├── src/                         # Core Python modules
│   ├── __init__.py
│   ├── database.py              # SQLAlchemy models & DatabaseManager
│   ├── schemas.py               # Pydantic validation schemas
│   ├── extraction.py            # LLM-based document parsing (Gemini)
│   ├── geo.py                   # Geocoding and distance utilities
│   ├── scoring.py               # Priority calculation engine
│   ├── order_selector.py        # Dispatch candidate generation (knapsack)
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
│   ├── config/                  # Configuration files (JSON)
│   │   ├── scoring_weights.json
│   │   ├── order_selector_config.json
│   │   └── routing_config.json
│   ├── geo/                     # Geographic reference data
│   │   ├── zones.json
│   │   └── localities.json
│   ├── raw/receipts/            # Input PDF/image receipts
│   └── processed/               # Extraction results cache
│
├── output/
│   ├── priority_scores.csv      # Calculated scores
│   ├── dispatches/              # Dispatch outputs (JSON, CSV)
│   └── maps/                    # Generated Folium HTML maps
│
└── prompts/                     # Development prompts archive
```

## Key Domain Concepts

| Term | Description |
|------|-------------|
| **Receipt** | Sales document (PDF) with variable formats per vendor |
| **Order** | Extracted and validated data from a receipt |
| **Dispatch** | Subset of orders selected for a single truck trip (max 8 pallets) |
| **Zone** | Geographic area (CABA, NORTH_ZONE, SOUTH_ZONE, WEST_ZONE) |
| **Priority Score** | Calculated value based on urgency, payment, client type, age |
| **Mandatory Order** | Order flagged by user that must be included in dispatch |
| **Candidate** | A potential dispatch configuration with metrics |

## Constraints

- Truck capacity: 8 pallets
- Single truck, single depot (fixed location in Buenos Aires)
- ~10 deliveries per trip
- Orders grouped by zone to minimize route dispersion
- Some orders may be marked as mandatory (must go out)

## Module Responsibilities

### extraction.py
- LLM-based document parsing using Google Gemini
- PDF/image ingestion with base64 encoding
- Pydantic validation of extracted data
- Geocoding integration for delivery addresses

### scoring.py
- Priority calculation with 4 weighted components:
  - Urgency (40%): Days until deadline
  - Payment (25%): Order value × payment status
  - Client (20%): Client type hierarchy
  - Age (15%): Days since order placement
- Configurable via `scoring_weights.json`
- Mandatory orders get score of 999,999

### order_selector.py
- Multiple selection strategies:
  - Greedy (efficiency, priority, zone-based)
  - Dynamic Programming (optimal)
  - Mandatory-first, best-fit
- Zone dispersion penalties
- Generates K candidate dispatches
- Configurable via `order_selector_config.json`

### routing.py
- TSP/VRP solving using OR-Tools
- Multiple solver strategies (nearest neighbor, guided local search)
- Haversine distance calculations
- Persistent route cache in SQLite
- Folium map generation with numbered stops

### database.py
- SQLAlchemy ORM models for all entities
- DatabaseManager class for CRUD operations
- Caching tables (geocoding, routes)
- Transaction management

### app_utils.py
- Streamlit helper functions
- Chart creation (Plotly)
- Map creation (Folium)
- Data loading and caching

## Configuration Files

### scoring_weights.json
```json
{
  "weights": {
    "urgency": 0.40,
    "payment": 0.25,
    "client": 0.20,
    "age": 0.15
  },
  "payment_status_multipliers": {...},
  "client_scores": {...},
  "mandatory_score": 999999
}
```

### order_selector_config.json
- Truck capacity settings
- Zone dispersion penalties
- Strategy-specific parameters

### routing_config.json
- Solver settings
- Time limits
- Distance calculation parameters

## LLM Usage Guidelines

- Use LLM only for document extraction (variable formats)
- Always validate LLM output with Pydantic
- Cache geocoding results to avoid repeated API calls
- Handle extraction failures gracefully with fallback values
- Store extraction artifacts for traceability

## Testing

- Notebooks serve as integration tests / demos
- Use synthetic data for all examples
- Focus on unit tests for scoring and optimization logic
- Validate outputs with Pydantic schemas

## Do Not

- Do not use matplotlib/seaborn for plotting
- Do not hardcode file paths (use relative paths or config)
- Do not store API keys in code (use .env)
- Do not over-engineer (this is a portfolio project)
- Do not implement deployment/CI-CD (out of scope)

## Running the Project

```bash
# Install dependencies
poetry install

# Run notebooks
poetry run jupyter lab

# Launch Streamlit showcase
poetry run streamlit run streamlit_app.py
```

## Environment Variables

Create `.env` file with:
```
GEMINI_API_KEY=your_api_key_here
```
