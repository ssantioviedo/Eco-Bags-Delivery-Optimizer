# AGENTS.md

## Project Overview

Delivery optimization system for an eco-friendly bag factory. The system processes sales receipts, extracts data using LLMs, calculates delivery priorities, and generates optimized dispatch candidates with route estimation.

## Tech Stack

- **Python**: 3.11+
- **Package Manager**: Poetry
- **Database**: SQLite (via SQLAlchemy)
- **Data Validation**: Pydantic v2
- **Optimization**: OR-Tools (CP-SAT)
- **LLM**: OpenAI API (GPT-4o) for document extraction
- **Geocoding**: Geopy + Nominatim
- **Visualization**: Plotly + Folium for maps
- **Notebooks**: JupyterLab

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
- One notebook per phase/module
- Clear markdown headers for sections
- Keep cells focused and short
- Include outputs in commits for demo purposes

## Project Structure

```text
src/
├── extraction.py # LLM-based document parsing
├── scoring.py # Priority calculation
├── optimizer.py # Dispatch generation (knapsack)
├── routing.py # TSP for route optimization
├── geo.py # Geocoding and distance utilities
├── schemas.py # Pydantic models
└── pipeline.py # End-to-end orchestration

notebooks/ # Development and demos (01_, 02_, etc.)
data/raw/ # Input files (receipts)
data/processed/ # SQLite database
data/geo/ # Geographic reference data (JSON)
output/ # Generated dispatches and maps
```

## Key Domain Concepts

| Term | Description |
|------|-------------|
| **Receipt** | Sales document (DOCX/Excel) with variable formats per vendor |
| **Order** | Extracted and validated data from a receipt |
| **Dispatch** | Subset of orders selected for a single truck trip (max 8 pallets) |
| **Zone** | Geographic area (CABA, NORTH_ZONE, SOUTH_ZONE, WEST_ZONE) |
| **Priority Score** | Calculated value based on urgency, payment, client type, age |
| **Mandatory Order** | Order flagged by user that must be included in dispatch |

## Constraints

- Truck capacity: 8 pallets
- Single truck, single depot (fixed location)
- ~10 deliveries per trip
- Orders grouped by zone to minimize route dispersion
- Some orders may be marked as mandatory (must go out)

## LLM Usage Guidelines

- Use LLM only for document extraction (variable formats)
- Always validate LLM output with Pydantic
- Cache geocoding results to avoid repeated API calls
- Handle extraction failures gracefully with fallback values

## Testing

- Focus on unit tests for scoring and optimization logic
- Use synthetic data for all examples
- Notebooks serve as integration tests / demos

## Do Not

- Do not use matplotlib/seaborn for plotting
- Do not hardcode file paths (use relative paths or config)
- Do not store API keys in code (use .env)
- Do not over-engineer (this is a portfolio project)
- Do not implement deployment/CI-CD (out of scope)

