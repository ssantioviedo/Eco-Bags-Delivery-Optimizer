# Phase 2: Receipt Extraction Module

## Context

Phase 1 is complete. The SQLite database is initialized with:
- zones and localities tables (populated)
- clients table (30 synthetic clients)
- products table (4 products)
- orders table (50 synthetic orders)
- geocoding_cache table (empty)
- processed_receipts table (empty)

Sample receipt PDFs are in `data/raw/receipts/`.

## Objective

Build the receipt extraction pipeline:

PDF → Gemini 1.5 Flash → JSON → Validación → Nominatim → SQLite


## Technical Requirements

### 1. LLM Integration
- Use Google Gemini 1.5 Flash (free tier)
- Library: `google-generativeai`
- Send PDF directly to Gemini (it supports PDF input)
- Use the extraction prompt provided below
- Parse JSON response

### 2. Pydantic Validation
Create schemas in `src/schemas.py` for:
- `ExtractedClient`: business_name, tax_id, delivery_address
- `ExtractedDocument`: issue_date, document_number
- `ExtractedItem`: bag_type_raw, bag_type_normalized, quantity_packs
- `ExtractedReceipt`: Full extraction with confidence score

### 3. Client Matching Logic
Before creating a new client:
1. Search by `tax_id` (exact match) → if found, use existing client_id
2. Search by `business_name` (fuzzy match, lowercase) → if found, use existing client_id
3. If no match → create new client with new UUID

### 4. Order Deduplication
Before creating a new order:
1. Check if `document_number` already exists in orders table
2. If exists → skip or update (log warning)
3. If not exists → create new order with new UUID

### 5. Geocoding Pipeline
For delivery addresses:
1. Check `geocoding_cache` table first (by address hash)
2. If cached → use cached coordinates
3. If not cached:
   - Call Nominatim via Geopy (respect 1 req/sec rate limit)
   - Parse locality from response
   - Match locality to zone using `localities` table
   - Cache result in `geocoding_cache`
4. If geocoding fails → try with just locality name
5. If still fails → set `requires_review = true`

### 6. Product Matching
Map `bag_type_normalized` to product_id:
- "large" → product with bag_type = "A"
- "medium" → product with bag_type = "B"  
- "small" → product with bag_type = "C"
- "unknown" → flag for review, use default or null

### 7. Pallet Calculation
- Get `packs_per_pallet` from products table
- Calculate: `pallets = quantity_packs / packs_per_pallet`

## Files to Create/Modify

### New Files
- `src/extraction.py` - Main extraction logic
- `src/geo.py` - Geocoding utilities
- `src/schemas.py` - Pydantic models (add extraction schemas)
- `notebooks/02_receipt_extraction.ipynb` - Demo notebook

### Modify
- `.env.example` - Add GEMINI_API_KEY

## extraction.py Functions

```python
def extract_from_pdf(pdf_path: Path) -> ExtractedReceipt
def find_or_create_client(extracted: ExtractedClient, db: Session) -> str  # returns client_id
def geocode_address(address: str, db: Session) -> GeocodingResult
def process_receipt(pdf_path: Path, db: Session) -> ProcessingResult
def process_all_receipts(receipts_dir: Path, db: Session) -> list[ProcessingResult]
```

## geo.py Functions

```python
def get_address_hash(address: str) -> str
def check_geocoding_cache(address_hash: str, db: Session) -> Optional[GeocodingResult]
def geocode_with_nominatim(address: str) -> Optional[GeocodingResult]
def match_locality_to_zone(locality: str, db: Session) -> Optional[str]  # returns zone_id
def cache_geocoding_result(result: GeocodingResult, db: Session) -> None
```

## Notebook 02 Structure

1. Setup - Load environment, connect to DB
1. Single Receipt Demo - Process one PDF step by step
1. LLM Response Analysis - Show raw JSON, validate with Pydantic
1. Geocoding Demo - Show address → coordinates → zone
1. Client Matching Demo - Show duplicate detection
1. Batch Processing - Process all receipts in folder
1. Results Summary - Plotly charts:
    - Extraction confidence distribution
    - Orders by zone (new vs existing)
    - Items requiring review
1. Folium Map - Show extracted delivery locations

## LLM Prompt to Use

You are a data extraction assistant specialized in processing Argentine delivery receipts (remitos) and invoices. Your task is to extract structured information from the attached document and return it in JSON format.

## Context

This data will be used to:
1. Search for existing clients in the database by business name or tax ID (CUIT)
2. Create new orders linked to existing or new clients
3. Match product references to actual product IDs

## Data to Extract

### 1. Client Information
- **business_name**: Company or client name (razón social)
- **tax_id**: Argentine CUIT number (format: XX-XXXXXXXX-X)
- **delivery_address**: Full delivery address (may appear as "entregar en", "domicilio", "dirección", "lugar de entrega", "enviar a", etc.)

### 2. Document Information
- **issue_date**: Document date (may appear as "fecha", "emitido", etc.)
- **document_number**: Receipt or invoice number (may appear as "remito nro", "factura", "comprobante", etc.)

### 3. Order Items
For each product line in the document, extract:
- **bag_type_raw**: The exact text describing the bag/product type as it appears
- **bag_type_normalized**: Normalize to one of: "large", "medium", "small", "unknown"
- **quantity_packs**: Number of packs/units ordered

### 4. Totals
- **total_amount**: Total monetary amount (if present)
- **total_packs**: Sum of all packs (if explicitly stated)

## Normalization Rules for bag_type

Map the Spanish terms to normalized values:
| Spanish terms | Normalized value |
|---------------|------------------|
| "grande", "gde", "g", "bolsa grande", "tamaño grande" | "large" |
| "mediana", "med", "m", "bolsa mediana", "tamaño mediano" | "medium" |
| "chica", "pequeña", "peq", "ch", "p", "bolsa chica", "tamaño chico" | "small" |
| Any other or unclear | "unknown" |

## Response Format

Return ONLY a valid JSON object with this exact structure:

```json
{
  "extraction_confidence": <float between 0.0 and 1.0>,
  "client": {
    "business_name": <string or null>,
    "tax_id": <string or null>,
    "delivery_address": <string or null>
  },
  "document": {
    "issue_date": <string in "YYYY-MM-DD" format or null>,
    "document_number": <string or null>
  },
  "items": [
    {
      "bag_type_raw": <string - exact text from document>,
      "bag_type_normalized": <"large" | "medium" | "small" | "unknown">,
      "quantity_packs": <integer or null>
    }
  ],
  "totals": {
    "total_amount": <float or null>,
    "total_packs": <integer or null>
  },
  "extraction_notes": <string with observations about ambiguous or missing data>,
  "requires_review": <boolean - true if any critical data is missing or uncertain>
}
```

## Critical Rules

1. **Never invent data.** If a field is not found, use `null`.
2. **Preserve original text** in `bag_type_raw` exactly as written.
3. **Quantities must be integers**, not strings.
4. **Amounts must be floats** without currency symbols.
5. **Set `requires_review: true`** if:
   - `business_name` is null
   - `delivery_address` is null
   - Any item has `bag_type_normalized: "unknown"`
   - Any item has `quantity_packs: null`
   - `extraction_confidence` < 0.7
6. **In `extraction_notes`**, explain any ambiguities, alternative interpretations, or data you couldn't extract with certainty.
7. **Do not assume or complete partial addresses.** Extract only what is explicitly written.

## Example

Input document text:
"Remito Nro 0001-00004521
Fecha: 15/03/2024
Cliente: Panadería Don José
CUIT: 30-12345678-9
Entregar en: Av. San Martín 1234, Quilmes
50 bolsas grandes
30 medianas
Total: $45.000"

Expected output:
```json
{
  "extraction_confidence": 0.95,
  "client": {
    "business_name": "Panadería Don José",
    "tax_id": "30-12345678-9",
    "delivery_address": "Av. San Martín 1234, Quilmes"
  },
  "document": {
    "issue_date": "2024-03-15",
    "document_number": "0001-00004521"
  },
  "items": [
    {
      "bag_type_raw": "bolsas grandes",
      "bag_type_normalized": "large",
      "quantity_packs": 50
    },
    {
      "bag_type_raw": "medianas",
      "bag_type_normalized": "medium",
      "quantity_packs": 30
    }
  ],
  "totals": {
    "total_amount": 45000.00,
    "total_packs": 80
  },
  "extraction_notes": "All fields extracted successfully.",
  "requires_review": false
}
```

## Error Handling

- Log all LLM calls and responses
- If JSON parsing fails → retry once with simplified prompt
- If geocoding fails → continue with null coordinates, flag for review
- If client matching is ambiguous → flag for review
- Never crash on single receipt failure, continue with batch

## Rate Limiting

- Nominatim: 1 request per second (use time.sleep)
- Gemini free tier: 15 requests per minute (add delay if needed)

## Dependencies to Add

- google-generativeai = "^0.5"
- geopy = "^2.4"
- python-dotenv = "^1.0"

## Expected Output
After processing all receipts:

- New clients added to clients table (if not existing)
- New orders added to orders table (with priority_score = null for now)
- Order items added to order_items table
- Geocoding results cached in geocoding_cache
- Processing log in processed_receipts table

## Code Conventions

- All code in English
- Type hints on all functions
- Google-style docstrings
- Use Plotly for visualizations (never matplotlib)
- Handle all edge cases gracefully