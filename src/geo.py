"""Geocoding utilities for the Delivery Optimization System.

This module provides functions for geocoding addresses using Nominatim,
caching results, and matching localities to zones.
"""

import hashlib
import re
import time
from datetime import datetime
from typing import Optional

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from sqlalchemy.orm import Session

from .database import GeocodingCacheModel, LocalityModel, ZoneModel
from .schemas import GeocodingConfidence, GeocodingResult


# Rate limiting for Nominatim (1 request per second)
_last_nominatim_call: float = 0.0
NOMINATIM_DELAY = 1.0  # seconds between requests

# Nominatim user agent (required by their policy)
NOMINATIM_USER_AGENT = "eco_bags_delivery_optimizer_v1"


def clean_address(address: str) -> str:
    """Clean an address by removing postal codes and normalizing format.
    
    Args:
        address: Raw address string.
        
    Returns:
        Cleaned address string.
    """
    if not address:
        return address
    
    cleaned = address
    
    # Remove postal code patterns: CP: 1234, CP 1234, C.P.: 1234, C.P. 1234, (1234), etc.
    postal_patterns = [
        r'\bCP\s*:?\s*\d{4,5}\b',      # CP: 1234, CP 1234
        r'\bC\.P\.?\s*:?\s*\d{4,5}\b', # C.P.: 1234, C.P. 1234
        r'\bCódigo\s*Postal\s*:?\s*\d{4,5}\b',  # Código Postal: 1234
        r'\bCodigo\s*Postal\s*:?\s*\d{4,5}\b',  # Codigo Postal: 1234
        r'\(\d{4,5}\)',                 # (1234)
        r'\[\d{4,5}\]',                 # [1234]
        r',\s*\d{4,5}\s*$',             # , 1234 at end
    ]
    
    for pattern in postal_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Clean up extra spaces and punctuation
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single
    cleaned = re.sub(r',\s*,', ',', cleaned)  # Double commas
    cleaned = re.sub(r',\s*$', '', cleaned)   # Trailing comma
    cleaned = cleaned.strip()
    
    return cleaned


def get_address_hash(address: str) -> str:
    """Generate a hash for an address to use as cache key.

    Args:
        address: The raw address string.

    Returns:
        SHA256 hash of the normalized address.
    """
    normalized = address.lower().strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def check_geocoding_cache(address_hash: str, session: Session) -> Optional[GeocodingResult]:
    """Check if an address is already cached in the database.

    Args:
        address_hash: The hash of the address to look up.
        session: SQLAlchemy database session.

    Returns:
        GeocodingResult if found in cache, None otherwise.
    """
    cached = session.query(GeocodingCacheModel).filter(
        GeocodingCacheModel.address_hash == address_hash
    ).first()

    if cached:
        return GeocodingResult(
            address_hash=cached.address_hash,
            raw_address=cached.raw_address,
            latitude=cached.latitude,
            longitude=cached.longitude,
            locality=cached.locality,
            zone_id=cached.zone_id,
            confidence=GeocodingConfidence(cached.confidence),
            success=cached.latitude is not None and cached.longitude is not None,
        )
    return None


def _respect_rate_limit() -> None:
    """Ensure we respect Nominatim's rate limit of 1 request per second."""
    global _last_nominatim_call
    elapsed = time.time() - _last_nominatim_call
    if elapsed < NOMINATIM_DELAY:
        time.sleep(NOMINATIM_DELAY - elapsed)
    _last_nominatim_call = time.time()

def _extract_locality_from_nominatim(address_details: dict, raw_address: str = "") -> Optional[str]:
    """Extract the most specific locality from Nominatim address details.
    
    Prioritizes suburb/neighbourhood (barrios) over city for Buenos Aires addresses.
    For CABA addresses, tries to get the barrio/neighbourhood. For other areas,
    falls back to city/municipality.
    
    Args:
        address_details: The 'address' dict from Nominatim response.
        raw_address: Original address string for fallback detection.
        
    Returns:
        The most appropriate locality name, or None.
    """
    # First try to get the barrio/neighbourhood (most specific)
    locality = (
        address_details.get("suburb")           # Barrio (e.g., "Palermo", "San Nicolas")
        or address_details.get("neighbourhood") # More specific neighborhood
        or address_details.get("quarter")       # Some areas use this
    )
    
    # If no barrio, use city/municipality (for addresses outside CABA)
    if not locality:
        locality = (
            address_details.get("city")
            or address_details.get("town")
            or address_details.get("municipality")
            or address_details.get("county")
        )
    
    # Check if we're in CABA based on state/city field
    state = address_details.get("state", "").lower()
    city = address_details.get("city", "").lower()
    is_caba = (
        "ciudad autónoma de buenos aires" in state or
        "ciudad autonoma de buenos aires" in state or
        "caba" in state or
        "capital federal" in state or
        "ciudad autónoma de buenos aires" in city or
        "buenos aires" == city
    )
    
    # If we're in CABA and got "Buenos Aires" or similar as locality,
    # but have no specific barrio, that's still valid - it maps to CABA zone
    if not locality and is_caba:
        locality = "Buenos Aires"
    
    # Special fallback: if we detected CABA in raw address but still no locality
    if not locality and raw_address:
        raw_lower = raw_address.lower()
        if "caba" in raw_lower or "capital federal" in raw_lower or "c.a.b.a" in raw_lower:
            locality = "Buenos Aires"
    
    return locality


def _extract_street_from_address(address: str) -> Optional[str]:
    """Extract just the street name and number from an address.
    
    Args:
        address: Full address string.
        
    Returns:
        Street portion of the address.
    """
    if not address:
        return None
    
    # Clean the address first
    cleaned = clean_address(address)
    
    # Split by comma and take the first part (usually the street)
    parts = [p.strip() for p in cleaned.split(',')]
    if parts:
        return parts[0]
    
    return cleaned


def _normalize_street_name(street: str) -> list[str]:
    """Generate variations of a street name for geocoding attempts.
    
    In Buenos Aires, streets like "Florida" might be searched as "Calle Florida"
    or vice versa. This generates multiple variations to try.
    
    Args:
        street: The street name with number.
        
    Returns:
        List of street name variations to try.
    """
    if not street:
        return []
    
    variations = [street]  # Original first
    
    # Common street type prefixes in Spanish
    prefixes = [
        "calle ", "av. ", "av ", "avenida ", "pasaje ", "paseo ", 
        "boulevard ", "blvd ", "diagonal "
    ]
    
    street_lower = street.lower()
    
    # Try removing prefixes
    for prefix in prefixes:
        if street_lower.startswith(prefix):
            without_prefix = street[len(prefix):]
            if without_prefix not in variations:
                variations.append(without_prefix)
            break
    
    return variations


def _is_in_caba(address_details: dict) -> bool:
    """Check if Nominatim result is in CABA.
    
    Args:
        address_details: The 'address' dict from Nominatim response.
        
    Returns:
        True if address is in CABA.
    """
    state = address_details.get("state", "").lower()
    city = address_details.get("city", "").lower()
    
    caba_indicators = [
        "ciudad autónoma de buenos aires",
        "ciudad autonoma de buenos aires",
        "caba",
        "capital federal"
    ]
    
    return (
        any(ind in state for ind in caba_indicators) or
        any(ind in city for ind in caba_indicators) or
        city == "buenos aires"
    )


def _extract_expected_zone_from_address(address: str) -> Optional[str]:
    """Extract the expected zone from an address based on keywords.
    
    Args:
        address: The raw address string.
        
    Returns:
        Expected zone_id or None if no zone indicator found.
    """
    if not address:
        return None
    
    address_lower = address.lower()
    
    # CABA indicators
    caba_indicators = ["caba", "capital federal", "c.a.b.a", "ciudad autonoma", "ciudad autónoma"]
    if any(ind in address_lower for ind in caba_indicators):
        return "CABA"
    
    # Zone indicators (from address text, not locality)
    if "zona norte" in address_lower or "north zone" in address_lower:
        return "NORTH_ZONE"
    if "zona sur" in address_lower or "south zone" in address_lower:
        return "SOUTH_ZONE"
    if "zona oeste" in address_lower or "west zone" in address_lower:
        return "WEST_ZONE"
    
    return None


def _validate_result_matches_expected_zone(
    address_details: dict,
    expected_zone: Optional[str],
) -> bool:
    """Validate that geocoding result matches the expected zone from address.
    
    Args:
        address_details: Nominatim address dict.
        expected_zone: Expected zone from address parsing.
        
    Returns:
        True if result matches or no expectation set.
    """
    if expected_zone is None:
        return True  # No expectation, accept any result
    
    if expected_zone == "CABA":
        is_match = _is_in_caba(address_details)
        return is_match
    
    # For other zones, we'd need more complex logic based on locality
    # For now, accept if no specific validation needed
    return True


def geocode_with_nominatim(
    address: str,
    user_agent: str = NOMINATIM_USER_AGENT,
    timeout: int = 10,
) -> Optional[GeocodingResult]:
    """Geocode an address using Nominatim with smart fallbacks and validation.
    
    If the address mentions a specific zone (e.g., CABA), validates that the
    API result matches the expected zone before accepting it.
    """
    _respect_rate_limit()

    # Clean the address first
    cleaned_address = clean_address(address)
    address_hash = get_address_hash(address)
    geolocator = Nominatim(user_agent=user_agent, timeout=timeout)
    
    # Determine expected zone from address text
    expected_zone = _extract_expected_zone_from_address(address)

    def _try_geocode(search_query: str, attempt_name: str) -> Optional[GeocodingResult]:
        """Helper to try geocoding with a search query and validate result."""
        nonlocal expected_zone, address_hash, address
        
        _respect_rate_limit()
        location = geolocator.geocode(search_query, addressdetails=True)
        
        if not location:
            return None
        
        address_details = location.raw.get("address", {})
        
        if expected_zone and not _validate_result_matches_expected_zone(address_details, expected_zone):
            return None
        
        locality = _extract_locality_from_nominatim(address_details, address)
        
        return GeocodingResult(
            address_hash=address_hash,
            raw_address=address,
            latitude=location.latitude,
            longitude=location.longitude,
            locality=locality,
            zone_id=None,
            confidence=GeocodingConfidence.HIGH if attempt_name == "Try 1" else GeocodingConfidence.MEDIUM,
            success=True,
        )

    try:
        # Try 1: Direct geocoding with cleaned address
        result = _try_geocode(cleaned_address, "Try 1 (direct)")
        if result:
            return result

        # Try 2: Add Buenos Aires context
        address_with_context = f"{cleaned_address}, Buenos Aires, Argentina"
        result = _try_geocode(address_with_context, "Try 2 (+ Buenos Aires)")
        if result:
            return result
        
        # Try 3: If CABA expected, search specifically in CABA with street variations
        if expected_zone == "CABA":
            street = _extract_street_from_address(address)
            if street:
                # Try multiple street name variations
                street_variations = _normalize_street_name(street)
                for i, street_var in enumerate(street_variations):
                    caba_search = f"{street_var}, Ciudad Autónoma de Buenos Aires, Argentina"
                    result = _try_geocode(caba_search, f"Try 3.{i+1} (CABA: '{street_var}')")
                    if result:
                        # Force locality to Buenos Aires if not set
                        if not result.locality:
                            result.locality = "Buenos Aires"
                        return result
        
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        pass

    return GeocodingResult(
        address_hash=address_hash,
        raw_address=address,
        latitude=None,
        longitude=None,
        locality=None,
        zone_id=None,
        confidence=GeocodingConfidence.LOW,
        success=False,
    )


def match_locality_to_zone(locality: str, session: Session) -> Optional[str]:
    """Match a locality name to a zone_id using the localities table.
    
    Special handling for CABA:
    - If only "CABA" or "Buenos Aires" is provided without a barrio,
      defaults to CABA zone
    - Tries to match specific barrios to their zones

    Args:
        locality: The locality name to match.
        session: SQLAlchemy database session.

    Returns:
        zone_id if a match is found, None otherwise.
    """
    if not locality:
        return None

    locality_lower = locality.lower().strip()
    
    # Special case: CABA without a specific neighbourhood → CABA zone
    caba_variants = [
        "caba", "ciudad autónoma de buenos aires", "buenos aires", 
        "c.a.b.a", "c.a.b.a.", "capital federal", "ciudad de buenos aires"
    ]
    if locality_lower in caba_variants:
        return "CABA"

    # Try exact match first (case insensitive) - use ilike with %
    result = session.query(LocalityModel).filter(
        LocalityModel.name.ilike(f"%{locality_lower}%")
    ).first()

    if result:
        return result.zone_id

    # Try partial match (locality contains or is contained in db entry)
    all_localities = session.query(LocalityModel).all()
    for loc in all_localities:
        db_name_lower = loc.name.lower()
        # Check both directions for partial match
        if locality_lower in db_name_lower or db_name_lower in locality_lower:
            return loc.zone_id

    # Try matching by removing common suffixes/prefixes and accents
    locality_clean = (
        locality_lower
        .replace("partido de ", "")
        .replace("ciudad de ", "")
        .replace("barrio ", "")
        .replace("la ", "")
        .replace("el ", "")
        .replace("á", "a").replace("é", "e").replace("í", "i")
        .replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    )
    for loc in all_localities:
        db_name_lower = loc.name.lower()
        db_name_clean = (
            db_name_lower
            .replace("á", "a").replace("é", "e").replace("í", "i")
            .replace("ó", "o").replace("ú", "u").replace("ñ", "n")
        )
        if locality_clean == db_name_clean or locality_clean in db_name_clean:
            return loc.zone_id

    return None


def cache_geocoding_result(result: GeocodingResult, session: Session) -> None:
    """Save a geocoding result to the cache table.
    
    Only caches successful results with valid coordinates.
    """
    # ← FIX: No cachear resultados fallidos
    if not result.success or result.latitude is None or result.longitude is None:
        return
    
    cache_entry = GeocodingCacheModel(
        address_hash=result.address_hash,
        raw_address=result.raw_address,
        latitude=result.latitude,
        longitude=result.longitude,
        locality=result.locality,
        zone_id=result.zone_id,
        confidence=result.confidence.value,
    )
    session.merge(cache_entry)
    session.commit()


def _infer_zone_from_coordinates(lat: float, lon: float) -> str:
    """Infer a zone based on coordinates relative to Buenos Aires center.
    
    This is a rough heuristic for auto-adding new localities.
    CABA center is approximately -34.6037, -58.3816
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Inferred zone_id
    """
    # CABA approximate bounding box
    caba_lat_min, caba_lat_max = -34.71, -34.52
    caba_lon_min, caba_lon_max = -58.53, -58.33
    
    if caba_lat_min <= lat <= caba_lat_max and caba_lon_min <= lon <= caba_lon_max:
        return "CABA"
    
    # Buenos Aires center reference
    ba_center_lat = -34.6037
    ba_center_lon = -58.3816
    
    # North: higher latitude (closer to 0)
    if lat > ba_center_lat:
        return "NORTH_ZONE"
    # South: lower latitude (more negative)
    elif lat < ba_center_lat - 0.15:
        return "SOUTH_ZONE"
    # West: lower longitude (more negative)
    elif lon < ba_center_lon - 0.15:
        return "WEST_ZONE"
    
    # Default to CABA for ambiguous cases near center
    return "CABA"


def auto_add_locality(
    locality_name: str,
    latitude: float,
    longitude: float,
    session: Session,
    zone_id: Optional[str] = None
) -> Optional[str]:
    """Automatically add a new locality to the database.
    
    If zone_id is not provided, infers it from coordinates.
    
    Args:
        locality_name: Name of the locality.
        latitude: Latitude coordinate.
        longitude: Longitude coordinate.
        session: SQLAlchemy session.
        zone_id: Optional zone_id, inferred if not provided.
        
    Returns:
        The zone_id of the added locality, or None if failed.
    """
    if not locality_name or latitude is None or longitude is None:
        return None
    
    # Clean locality name
    locality_clean = locality_name.strip()
    if not locality_clean:
        return None
    
    # Check if already exists (case insensitive)
    existing = session.query(LocalityModel).filter(
        LocalityModel.name.ilike(f"%{locality_clean}%")
    ).first()
    
    if existing:
        return existing.zone_id
    
    # Infer zone if not provided
    if zone_id is None:
        zone_id = _infer_zone_from_coordinates(latitude, longitude)
    
    # Generate locality_id from name
    import re
    locality_id = re.sub(r'[^a-z0-9]+', '_', locality_clean.lower()).strip('_')
    
    # Ensure zone exists
    zone_exists = session.query(ZoneModel).filter(ZoneModel.zone_id == zone_id).first()
    if not zone_exists:
        zone_id = "CABA"
    
    try:
        new_locality = LocalityModel(
            locality_id=locality_id,
            name=locality_clean,
            zone_id=zone_id,
            latitude=latitude,
            longitude=longitude,
        )
        session.add(new_locality)
        session.commit()
        return zone_id
    except Exception as e:
        session.rollback()
        return None


def geocode_address(address: str, session: Session) -> GeocodingResult:
    """Full geocoding pipeline: check cache, geocode if needed, match zone, cache result.

    Args:
        address: The address to geocode.
        session: SQLAlchemy database session.

    Returns:
        GeocodingResult with all available information.
    """
    # Clean the address first (remove postal codes, extra whitespace)
    cleaned_address = clean_address(address)
    
    address_hash = get_address_hash(cleaned_address)

    # Step 1: Check cache
    cached = check_geocoding_cache(address_hash, session)
    if cached:
        return cached

    # Step 2: Geocode with Nominatim (use original address for zone detection)
    result = geocode_with_nominatim(address)

    if result is None:
        result = GeocodingResult(
            address_hash=address_hash,
            raw_address=cleaned_address,
            success=False,
        )

    # Step 3: Match locality to zone
    if result.locality:
        result.zone_id = match_locality_to_zone(result.locality, session)
        
        # Step 3a: Auto-add new locality if not found in DB
        if result.zone_id is None and result.success and result.latitude and result.longitude:
            result.zone_id = auto_add_locality(
                locality_name=result.locality,
                latitude=result.latitude,
                longitude=result.longitude,
                session=session,
            )
    
    # Step 3b: Fallback - if no zone matched but address contains CABA indicators,
    # trust the address and assign CABA zone directly
    if result.zone_id is None and result.success:
        address_lower = address.lower()  # Use original address for zone detection
        caba_indicators = ["caba", "capital federal", "c.a.b.a", "ciudad autonoma", "ciudad autónoma"]
        if any(indicator in address_lower for indicator in caba_indicators):
            result.zone_id = "CABA"
            result.locality = result.locality or "Buenos Aires"

    # Step 4: Cache the result (only if successful)
    cache_geocoding_result(result, session)

    return result


def extract_locality_from_address(address: str) -> Optional[str]:
    """Try to extract a locality name from an address string.

    This is a fallback for when geocoding fails.

    Args:
        address: The address string.

    Returns:
        Extracted locality name or None.
    """
    if not address:
        return None

    # Common Buenos Aires localities to look for
    known_localities = [
        "palermo", "recoleta", "belgrano", "caballito", "almagro",
        "villa crespo", "san telmo", "la boca", "barracas", "flores",
        "vicente lopez", "olivos", "san isidro", "martinez", "tigre",
        "pilar", "avellaneda", "quilmes", "lanus", "lomas de zamora",
        "moron", "haedo", "ramos mejia", "san justo", "merlo", "moreno",
        "san miguel", "hurlingham", "ituzaingo", "caseros", "banfield",
        "temperley", "berazategui", "florencio varela", "bernal",
    ]

    address_lower = address.lower()

    for locality in known_localities:
        if locality in address_lower:
            return locality.title()

    # Try to extract from comma-separated parts
    parts = [p.strip() for p in address.split(",")]
    if len(parts) >= 2:
        # Usually locality is the second-to-last part before "Buenos Aires"
        for part in reversed(parts[:-1]):
            part_clean = part.strip()
            if part_clean and not part_clean.isdigit():
                # Skip street numbers and common words
                if not any(word in part_clean.lower() for word in ["av.", "calle", "argentina"]):
                    return part_clean

    return None
