# Fixes Applied - January 13, 2026

## Summary of Changes

Fixed critical issues with receipt address extraction and geocoding pipeline for the Delivery Optimization System.

## 1. Enhanced Address Extraction Prompt

**File:** `src/extraction.py`

**Changes:**
- Updated `EXTRACTION_PROMPT` with explicit instructions for address extraction
- Added emphasis on **CRITICAL** importance of delivery address field
- Included specific keywords to look for: "Entregar en", "Domicilio", "Dirección", "Lugar de entrega", etc.
- Added instruction to capture complete address with street, number, AND locality
- Special instruction for CABA: "If CABA is mentioned, also try to find the barrio/neighbourhood"
- Clarified that addresses should NOT be partial - must include everything found
- Added rule: "If you cannot find the full delivery address, set `requires_review: true`"

## 2. Fixed Duplicate Function in Geocoding Module

**File:** `src/geo.py`

**Issue:** Function `_extract_locality_from_nominatim` was defined twice with slightly different implementations

**Solution:** Removed duplicate and kept the improved version that:
- Prioritizes suburb/neighbourhood for CABA addresses
- Falls back to city/municipality for addresses outside CABA
- Better handles missing locality data

## 3. Improved CABA Zone Matching

**File:** `src/geo.py`

**Changes to `match_locality_to_zone` function:**
- Added special case handling: If locality is only "CABA", "Buenos Aires", or "C.A.B.A", automatically return "CABA" zone
- This fixes the issue where "Av. Regimiento de Patricios 1030, CABA" (without specific barrio) now correctly assigns CABA zone
- Enhanced fuzzy matching by removing additional prefixes ("la", "el") to improve matching of various address formats
- Added handling for abbreviations like "C.A.B.A"

## 4. Model Updated to Gemini 2.5 Flash

**File:** `src/extraction.py`

**Status:** ✅ Already configured to use `gemini-2.5-flash` (line 222)
- The model was already updated from gemini-1.5-flash to gemini-2.5-flash
- No additional changes needed

## Why These Fixes Improve Address Extraction

### Problem 1: Addresses Not Being Captured
- **Root cause:** The LLM prompt didn't emphasize address extraction enough
- **Solution:** Made address extraction CRITICAL with specific keywords and examples

### Problem 2: CABA Addresses Without Suburb Not Matching
- **Root cause:** When only "CABA" was provided without a barrio, the matching logic failed
- **Solution:** Added explicit handling for CABA-only addresses to assign CABA zone by default

### Problem 3: Missing Suburbs in Nominatim Results
- **Root cause:** For some addresses, Nominatim doesn't return a suburb field
- **Solution:** Function now gracefully falls back to city/municipality while still trying to get barrios first

## Testing Recommendations

1. **Test CABA Address Without Suburb:**
   - Address: "Av. Regimiento de Patricios 1030, CABA"
   - Expected: Zone = "CABA"

2. **Test Complete CABA Address With Barrio:**
   - Address: "Av. Corrientes 1234, Palermo, CABA"
   - Expected: Zone = "CABA", Locality = "Palermo"

3. **Test Address Outside CABA:**
   - Address: "Calle Mitre 500, Quilmes"
   - Expected: Zone = "SOUTH_ZONE", Locality = "Quilmes"

## Database Suburbs for Optimization

**Answer to the original question:** "Do I really need the suburb for optimization?"

**YES, the suburb/barrio IS important for optimization because:**
1. **Route Planning:** Different barrios in the same zone may require different routes
2. **Delivery Clustering:** Grouping deliveries by barrio before zone is more efficient
3. **Service Quality:** Accurate barrio-level geocoding improves delivery accuracy
4. **Future Phases:** Phases 4-5 (routing optimization) will benefit from precise barrio coordinates

**However:** If the suburb is missing, CABA as a zone is still valid for delivery grouping - it just means slightly less optimized routes but still functional.

## Files Modified

1. ✅ `src/extraction.py` - Enhanced prompt
2. ✅ `src/geo.py` - Removed duplicate function, improved zone matching
3. ✅ `notebooks/02_receipt_extraction.ipynb` - (No changes needed)

## Next Steps

1. Test with actual PDF receipts to verify address extraction works
2. Monitor geocoding cache hits to ensure Nominatim responses include suburb information
3. Consider adding a fallback address extraction method if Gemini 2.5 Flash still misses some addresses
