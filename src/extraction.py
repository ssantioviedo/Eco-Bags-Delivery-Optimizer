"""Receipt extraction module for the Delivery Optimization System.

This module provides functions for extracting structured data from PDF receipts
using Google Gemini, validating with Pydantic, and persisting to the database.
"""

import json
import os
import re
import time
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from .database import (
    ClientModel,
    OrderItemModel,
    OrderModel,
    ProcessedReceiptModel,
    ProductModel,
)
from .geo import geocode_address, extract_locality_from_address
from .schemas import (
    ExtractedClient,
    ExtractedDocument,
    ExtractedItem,
    ExtractedReceipt,
    ExtractedTotals,
    GeocodingResult,
    NormalizedBagType,
    ProcessingResult,
)


# Load environment variables
load_dotenv()

# Gemini rate limiting (15 requests per minute for free tier)
_last_gemini_call: float = 0.0
GEMINI_DELAY = 4.0  # seconds between requests (to stay well under limit)


# The extraction prompt for Gemini
EXTRACTION_PROMPT = '''You are a data extraction assistant specialized in processing Argentine delivery receipts (remitos) and invoices. Your task is to extract structured information from the attached document and return it in JSON format.

## IMPORTANT: Document Context

**We are Eco-Bags**, a company that manufactures and sells eco-friendly bags.
- If you see "Eco-Bags", "ECO BAGS", or similar - that is US (the seller)
- The CLIENT is the OTHER party - the one receiving/buying the products
- Extract the CLIENT's information, NOT Eco-Bags' information

## Context

This data will be used to:
1. Search for existing clients in the database by business name or tax ID (CUIT)
2. Create new orders linked to existing or new clients
3. Match product references to actual product IDs

## Data to Extract

### 1. Client Information (THE BUYER, not Eco-Bags)
- **business_name**: Company or client name (razón social) - REQUIRED - This is the BUYER, not the seller
- **tax_id**: Argentine CUIT number (format: XX-XXXXXXXX-X) - may appear as "CUIT", "CIF", "RUT"
- **delivery_address**: COMPLETE delivery address - CRITICAL - Look for:
  - "Entregar en", "Domicilio", "Dirección", "Lugar de entrega", "Enviar a", "Dir.", "Despacho"
  - Street name and number (Av. Corrientes 1234, Calle Mitre 500)
  - Locality/Suburb/Barrio (Palermo, Recoleta, Quilmes, CABA, etc.)
  - Always include complete address with street, number, AND locality
  - IMPORTANT: If CABA is mentioned, also try to find the barrio/neighbourhood
  - DO NOT include postal codes (CP, C.P., Código Postal) in the address - extract them separately
  - Clean format: "Street Name 1234, Locality" (no CP, no extra punctuation)

### 2. Document Information
- **issue_date**: Document date (may appear as "fecha", "emitido", "fechado") - format YYYY-MM-DD
- **document_number**: Receipt or invoice number (may appear as "remito nro", "factura", "comprobante", "número")

### 3. Order Items
For each product line in the document, extract:
- **bag_type_raw**: The exact text describing the bag/product type as it appears
- **bag_type_normalized**: Normalize to one of: "large", "medium", "small", "unknown"
- **quantity_packs**: Number of packs/units ordered

### 4. Totals
- **total_amount**: Total monetary amount (if present) - no currency symbols
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
5. **Address is CRITICAL** - If you cannot find the full delivery address, set `requires_review: true`.
6. **Set `requires_review: true`** if:
   - `business_name` is null
   - `delivery_address` is null or incomplete
   - Any item has `bag_type_normalized: "unknown"`
   - Any item has `quantity_packs: null`
   - `extraction_confidence` < 0.7
7. **In `extraction_notes`**, explain any ambiguities, alternative interpretations, or data you couldn\'t extract with certainty.
8. **Do not assume or complete partial addresses.** Extract only what is explicitly written.'''


def _get_gemini_client() -> genai.Client:
    """Get configured Gemini client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    return genai.Client(api_key=api_key)


def _respect_gemini_rate_limit() -> None:
    """Ensure we respect Gemini's rate limit."""
    global _last_gemini_call
    elapsed = time.time() - _last_gemini_call
    if elapsed < GEMINI_DELAY:
        time.sleep(GEMINI_DELAY - elapsed)
    _last_gemini_call = time.time()


def _parse_json_response(response_text: str) -> dict:
    """Parse JSON from Gemini response, handling markdown code blocks.

    Args:
        response_text: Raw response text from Gemini.

    Returns:
        Parsed JSON dictionary.

    Raises:
        ValueError: If JSON parsing fails.
    """
    # Remove markdown code blocks if present
    text = response_text.strip()

    # Try to find JSON in code blocks
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        text = json_match.group(1).strip()

    # Try to parse as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try to find JSON object in text
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            try:
                return json.loads(text[json_start:json_end])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Failed to parse JSON response: {e}")


def extract_from_pdf(pdf_path: Path) -> ExtractedReceipt:
    """Extract structured data from a PDF receipt using Gemini.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        ExtractedReceipt with all extracted information.

    Raises:
        FileNotFoundError: If PDF file doesn't exist.
        ValueError: If extraction fails.
    """
    import shutil
    import tempfile
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    gemini_client = _get_gemini_client()
    _respect_gemini_rate_limit()

    # Handle filenames with special characters (accents, etc.)
    # by creating a temporary copy with an ASCII-safe name
    temp_file = None
    upload_path = pdf_path
    
    # Check if filename has non-ASCII characters
    original_name = pdf_path.name
    try:
        original_name.encode('ascii')
    except UnicodeEncodeError:
        # Create a temp file with safe name
        temp_dir = tempfile.mkdtemp()
        safe_name = f"receipt_{uuid.uuid4().hex[:8]}.pdf"
        temp_file = Path(temp_dir) / safe_name
        shutil.copy2(pdf_path, temp_file)
        upload_path = temp_file

    # Upload the PDF to Gemini
    uploaded_file = gemini_client.files.upload(file=upload_path)

    try:
        # Generate response using the SDK API
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[EXTRACTION_PROMPT, uploaded_file],
        )
        response_text = response.text

        # Parse the JSON response
        data = _parse_json_response(response_text)

        # Validate with Pydantic
        client = ExtractedClient(**data.get("client", {}))
        document = ExtractedDocument(**data.get("document", {}))

        items = []
        for item_data in data.get("items", []):
            bag_type = item_data.get("bag_type_normalized", "unknown")
            try:
                normalized = NormalizedBagType(bag_type)
            except ValueError:
                normalized = NormalizedBagType.UNKNOWN

            items.append(ExtractedItem(
                bag_type_raw=item_data.get("bag_type_raw", ""),
                bag_type_normalized=normalized,
                quantity_packs=item_data.get("quantity_packs"),
            ))

        totals = ExtractedTotals(**data.get("totals", {}))

        return ExtractedReceipt(
            extraction_confidence=data.get("extraction_confidence", 0.5),
            client=client,
            document=document,
            items=items,
            totals=totals,
            extraction_notes=data.get("extraction_notes"),
            requires_review=data.get("requires_review", True),
        )

    except Exception as e:
        # Return a failed extraction result
        return ExtractedReceipt(
            extraction_confidence=0.0,
            client=ExtractedClient(),
            document=ExtractedDocument(),
            items=[],
            totals=ExtractedTotals(),
            extraction_notes=f"Extraction failed: {str(e)}",
            requires_review=True,
        )
    finally:
        # Clean up uploaded file from Gemini
        try:
            gemini_client.files.delete(name=uploaded_file.name)
        except Exception:
            pass
        
        # Clean up temporary file if created
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
                temp_file.parent.rmdir()
            except Exception:
                pass


def find_client_by_tax_id(tax_id: str, session: Session) -> Optional[ClientModel]:
    """Find a client by exact tax ID match.

    Args:
        tax_id: The CUIT to search for.
        session: SQLAlchemy database session.

    Returns:
        ClientModel if found, None otherwise.
    """
    if not tax_id:
        return None

    # Normalize tax_id (remove spaces, ensure dashes)
    tax_id_clean = tax_id.strip().replace(" ", "")

    return session.query(ClientModel).filter(
        ClientModel.tax_id == tax_id_clean
    ).first()


def find_client_by_name(business_name: str, session: Session) -> Optional[ClientModel]:
    """Find a client by fuzzy business name match.

    Args:
        business_name: The business name to search for.
        session: SQLAlchemy database session.

    Returns:
        ClientModel if found, None otherwise.
    """
    if not business_name:
        return None

    # Normalize the search name
    name_lower = business_name.lower().strip()
    # Remove common business suffixes for better matching
    name_clean = (
        name_lower
        .replace("s.a.", "").replace("sa", "")
        .replace("s.r.l.", "").replace("srl", "")
        .replace("s.a.s.", "").replace("sas", "")
        .replace(".", "").replace(",", "")
        .strip()
    )

    # Try exact match first (case insensitive) with wildcards
    result = session.query(ClientModel).filter(
        ClientModel.business_name.ilike(f"%{name_lower}%")
    ).first()

    if result:
        return result

    # Try with cleaned name
    result = session.query(ClientModel).filter(
        ClientModel.business_name.ilike(f"%{name_clean}%")
    ).first()

    if result:
        return result

    # Try partial match - check all clients
    all_clients = session.query(ClientModel).all()
    for client in all_clients:
        client_name_lower = client.business_name.lower()
        client_name_clean = (
            client_name_lower
            .replace("s.a.", "").replace("sa", "")
            .replace("s.r.l.", "").replace("srl", "")
            .replace("s.a.s.", "").replace("sas", "")
            .replace(".", "").replace(",", "")
            .strip()
        )
        # Check if significant overlap (either direction)
        if (name_clean in client_name_clean or 
            client_name_clean in name_clean or
            name_lower in client_name_lower or 
            client_name_lower in name_lower):
            return client

    return None


def find_or_create_client(
    extracted: ExtractedClient,
    geocoding: GeocodingResult,
    session: Session
) -> tuple[str, bool]:
    """Find an existing client or create a new one.

    Args:
        extracted: Extracted client information.
        geocoding: Geocoding result for the delivery address.
        session: SQLAlchemy database session.

    Returns:
        Tuple of (client_id, is_new_client).
    """
    # Try to find by tax_id first (most reliable)
    if extracted.tax_id:
        existing = find_client_by_tax_id(extracted.tax_id, session)
        if existing:
            return existing.client_id, False

    # Try to find by business name
    if extracted.business_name:
        existing = find_client_by_name(extracted.business_name, session)
        if existing:
            return existing.client_id, False

    # Create new client
    client_id = f"CLI-{uuid.uuid4().hex[:8].upper()}"

    new_client = ClientModel(
        client_id=client_id,
        business_name=extracted.business_name or "Unknown Client",
        tax_id=extracted.tax_id or f"00-00000000-0",
        billing_address=extracted.delivery_address or "Address pending",
        zone_id=geocoding.zone_id or "CABA",  # Default to CABA
        latitude=geocoding.latitude or -34.6037,  # Default to Buenos Aires center
        longitude=geocoding.longitude or -58.3816,
        is_star_client=False,
        is_new_client=True,
        first_order_date=date.today(),
    )

    session.add(new_client)
    session.commit()

    return client_id, True


def check_order_exists(document_number: str, session: Session) -> Optional[str]:
    """Check if an order with the given document number already exists.

    Args:
        document_number: The document/receipt number to check.
        session: SQLAlchemy database session.

    Returns:
        order_id if exists, None otherwise.
    """
    if not document_number:
        return None

    # Check in processed_receipts table first
    existing = session.query(ProcessedReceiptModel).filter(
        ProcessedReceiptModel.source_file.contains(document_number)
    ).first()

    if existing and existing.generated_order_id:
        return existing.generated_order_id

    return None


def get_product_by_bag_type(bag_type: NormalizedBagType, session: Session) -> Optional[ProductModel]:
    """Get product by normalized bag type.

    Args:
        bag_type: Normalized bag type.
        session: SQLAlchemy database session.

    Returns:
        ProductModel if found, None otherwise.
    """
    type_mapping = {
        NormalizedBagType.LARGE: "large",
        NormalizedBagType.MEDIUM: "medium",
        NormalizedBagType.SMALL: "small",
        NormalizedBagType.UNKNOWN: None,
    }

    db_type = type_mapping.get(bag_type)
    if not db_type:
        return None

    return session.query(ProductModel).filter(
        ProductModel.bag_type == db_type
    ).first()


def calculate_pallets(quantity_packs: int, packs_per_pallet: int) -> float:
    """Calculate the number of pallets for a given quantity.

    Args:
        quantity_packs: Number of packs ordered.
        packs_per_pallet: Packs that fit on one pallet.

    Returns:
        Number of pallets (can be fractional).
    """
    if packs_per_pallet <= 0:
        return 0.0
    return round(quantity_packs / packs_per_pallet, 2)


def process_receipt(pdf_path: Path, session: Session) -> ProcessingResult:
    """Process a single receipt PDF end-to-end.

    Args:
        pdf_path: Path to the PDF file.
        session: SQLAlchemy database session.

    Returns:
        ProcessingResult with all processing information.
    """
    start_time = time.time()
    extraction = None  # Store extraction early so we can include it in error results
    geocoding = None

    try:
        # Step 1: Extract from PDF
        extraction = extract_from_pdf(pdf_path)

        if extraction.extraction_confidence == 0.0:
            return ProcessingResult(
                receipt_path=str(pdf_path),
                success=False,
                extraction=extraction,
                error_message=extraction.extraction_notes,
                processing_time_seconds=time.time() - start_time,
            )

        # Step 2: Check for duplicate order
        if extraction.document.document_number:
            existing_order_id = check_order_exists(
                extraction.document.document_number, session
            )
            if existing_order_id:
                return ProcessingResult(
                    receipt_path=str(pdf_path),
                    success=True,
                    order_id=existing_order_id,
                    order_is_duplicate=True,
                    extraction=extraction,
                    processing_time_seconds=time.time() - start_time,
                )

        # Step 3: Geocode address
        geocoding = None
        if extraction.client.delivery_address:
            geocoding = geocode_address(extraction.client.delivery_address, session)
        else:
            geocoding = GeocodingResult(
                address_hash="",
                raw_address="",
                success=False,
            )

        # Step 4: Find or create client
        client_id, client_is_new = find_or_create_client(
            extraction.client, geocoding, session
        )

        # Step 5: Create order
        order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"

        # Calculate total pallets and total packs
        total_pallets = 0.0
        total_packs = 0
        order_items = []

        for item in extraction.items:
            if item.quantity_packs:
                total_packs += item.quantity_packs
                product = get_product_by_bag_type(item.bag_type_normalized, session)
                if product:
                    pallets = calculate_pallets(item.quantity_packs, product.packs_per_pallet)
                    total_pallets += pallets

                    order_items.append(OrderItemModel(
                        order_id=order_id,
                        product_id=product.product_id,
                        quantity_packs=item.quantity_packs,
                        pallets=pallets,
                    ))

        # Determine delivery deadline (7 days from issue or today)
        issue_date = extraction.document.issue_date or date.today()
        delivery_deadline = issue_date + __import__("datetime").timedelta(days=7)

        new_order = OrderModel(
            order_id=order_id,
            client_id=client_id,
            issue_date=issue_date,
            delivery_deadline=delivery_deadline,
            delivery_address=extraction.client.delivery_address or "Address pending",
            delivery_latitude=geocoding.latitude or -34.6037,
            delivery_longitude=geocoding.longitude or -58.3816,
            delivery_zone_id=geocoding.zone_id or "CABA",
            total_amount=extraction.totals.total_amount or 0.0,
            payment_status="pending",
            is_mandatory=False,
            quantity_packs=total_packs or extraction.totals.total_packs or 0,
            total_pallets=total_pallets,
            priority_score=None,  # Will be calculated in Phase 3
            status="pending",
        )

        session.add(new_order)

        for item in order_items:
            session.add(item)

        # Step 6: Log to processed_receipts
        processed_receipt = ProcessedReceiptModel(
            receipt_id=f"RCP-{uuid.uuid4().hex[:8].upper()}",
            source_file=str(pdf_path),
            raw_extraction=json.dumps(extraction.model_dump(), default=str),
            generated_order_id=order_id,
            extraction_confidence=extraction.extraction_confidence,
        )

        session.add(processed_receipt)
        session.commit()

        return ProcessingResult(
            receipt_path=str(pdf_path),
            success=True,
            client_id=client_id,
            client_is_new=client_is_new,
            order_id=order_id,
            order_is_duplicate=False,
            extraction=extraction,
            geocoding=geocoding,
            processing_time_seconds=time.time() - start_time,
        )

    except Exception as e:
        session.rollback()
        return ProcessingResult(
            receipt_path=str(pdf_path),
            success=False,
            extraction=extraction,  # Include extraction if we got one
            geocoding=geocoding,    # Include geocoding if we got one
            error_message=str(e),
            processing_time_seconds=time.time() - start_time,
        )


def process_all_receipts(
    receipts_dir: Path,
    session: Session,
    file_pattern: str = "*.pdf"
) -> list[ProcessingResult]:
    """Process all receipts in a directory.

    Args:
        receipts_dir: Directory containing PDF receipts.
        session: SQLAlchemy database session.
        file_pattern: Glob pattern for files to process.

    Returns:
        List of ProcessingResult for each receipt.
    """
    results = []

    pdf_files = sorted(receipts_dir.glob(file_pattern))

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        result = process_receipt(pdf_path, session)
        results.append(result)

        status = "✓" if result.success else "✗"
        if result.order_is_duplicate:
            status = "⊘ (duplicate)"

        confidence = result.extraction.extraction_confidence if result.extraction else 0
        
        # Show client info
        client_info = ""
        if result.success:
            if result.client_is_new:
                client_info = " | New client created"
            else:
                client_info = " | Existing client matched"
        
        print(f"  {status} - Confidence: {confidence:.2f}{client_info}")
        
        # Show error if failed
        if not result.success and result.error_message:
            print(f"  ❌ Error: {result.error_message}")

    return results


def get_processing_summary(results: list[ProcessingResult]) -> dict:
    """Generate a summary of processing results.

    Args:
        results: List of ProcessingResult objects.

    Returns:
        Dictionary with summary statistics.
    """
    total = len(results)
    successful = sum(1 for r in results if r.success and not r.order_is_duplicate)
    duplicates = sum(1 for r in results if r.order_is_duplicate)
    failed = sum(1 for r in results if not r.success)
    new_clients = sum(1 for r in results if r.client_is_new)

    avg_confidence = 0.0
    confidence_values = [r.extraction.extraction_confidence for r in results if r.extraction]
    if confidence_values:
        avg_confidence = sum(confidence_values) / len(confidence_values)

    requires_review = sum(
        1 for r in results
        if r.extraction and r.extraction.requires_review
    )

    total_time = sum(r.processing_time_seconds for r in results)

    return {
        "total_receipts": total,
        "successful": successful,
        "duplicates": duplicates,
        "failed": failed,
        "new_clients_created": new_clients,
        "average_confidence": round(avg_confidence, 2),
        "requires_review": requires_review,
        "total_processing_time_seconds": round(total_time, 2),
    }
