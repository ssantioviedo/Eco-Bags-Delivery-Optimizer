"""Extract data from all receipts and save to JSON for app preview.

This script processes all receipt images/PDFs in data/raw/receipts/
and saves the extracted data to data/processed/extracted_receipts.json
for use in the Streamlit app preview.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import date

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction import extract_from_pdf
from google import genai
from dotenv import load_dotenv

load_dotenv()

def extract_all_receipts() -> Dict[str, Any]:
    """Extract all receipts and return structured data."""
    
    receipts_dir = Path(__file__).parent.parent / "data" / "raw" / "receipts"
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_data = {}
    
    # Get all PDF files (prioritize PDF over JPG)
    pdf_files = sorted(receipts_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {receipts_dir}")
        return extracted_data
    
    print(f"📋 Found {len(pdf_files)} receipt files to process")
    print()
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        
        try:
            # Extract from PDF
            extraction_result = extract_from_pdf(pdf_path)
            
            # Convert to dictionary for JSON serialization
            extracted_dict = extraction_result.model_dump()
            
            # Extract just the display-friendly format
            display_data = {
                "client_name": extraction_result.client.business_name or "Unknown",
                "tax_id": extraction_result.client.tax_id or "Not found",
                "issue_date": str(extraction_result.document.issue_date) if extraction_result.document.issue_date else "Not found",
                "delivery_address": extraction_result.client.delivery_address or "Not found",
                "items": [
                    {
                        "product": f"ECO-{item.bag_type_normalized.upper()}-{item.quantity_packs:03d}" if item.quantity_packs else "ECO-UNK-000",
                        "quantity": item.quantity_packs or 0,
                        "bag_type": item.bag_type_normalized or "unknown",
                        "bag_type_raw": item.bag_type_raw,
                    }
                    for item in extraction_result.items
                ],
                "total_amount": float(extraction_result.totals.total_amount) if extraction_result.totals.total_amount else 0.0,
                "total_packs": extraction_result.totals.total_packs or 0,
                "payment_status": "unknown",
                "extraction_confidence": float(extraction_result.extraction_confidence),
                "requires_review": extraction_result.requires_review,
                "extraction_notes": extraction_result.extraction_notes,
            }
            
            extracted_data[pdf_path.name] = display_data
            
            print(f"  ✓ Success | Confidence: {extraction_result.extraction_confidence:.0%}")
            print(f"    Client: {extraction_result.client.business_name}")
            print(f"    Items: {len(extraction_result.items)} | Total: {extraction_result.totals.total_packs} packs")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            # Add fallback data so app doesn't break
            extracted_data[pdf_path.name] = {
                "client_name": pdf_path.stem.replace("_", " "),
                "tax_id": "Not found",
                "issue_date": "Not found",
                "delivery_address": "Not found",
                "items": [],
                "total_amount": 0.0,
                "total_packs": 0,
                "payment_status": "unknown",
                "extraction_confidence": 0.0,
                "requires_review": True,
                "extraction_notes": f"Extraction failed: {str(e)}",
            }
        
        print()
    
    # Save to JSON
    output_path = processed_dir / "extracted_receipts.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Extracted data saved to: {output_path}")
    print(f"📊 Total receipts processed: {len(extracted_data)}")
    
    return extracted_data


if __name__ == "__main__":
    extract_all_receipts()
