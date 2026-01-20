"""Pydantic schemas for the Delivery Optimization System.

This module defines all data validation schemas used throughout the system,
including zones, localities, clients, products, orders, and dispatches.
"""

from datetime import date, datetime, time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ZoneId(str, Enum):
    """Geographic zone identifiers for the Buenos Aires metropolitan area."""

    CABA = "CABA"
    NORTH_ZONE = "NORTH_ZONE"
    SOUTH_ZONE = "SOUTH_ZONE"
    WEST_ZONE = "WEST_ZONE"


class PaymentStatus(str, Enum):
    """Payment status for orders."""

    PENDING = "pending"
    PARTIAL = "partial"
    PAID = "paid"


class OrderStatus(str, Enum):
    """Order fulfillment status."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    DELIVERED = "delivered"


class DispatchStatus(str, Enum):
    """Dispatch lifecycle status."""

    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"


class BagType(str, Enum):
    """Types of bags produced by the factory based on dimensions."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    SPECIAL = "special"


class GeocodingConfidence(str, Enum):
    """Confidence level for geocoded addresses."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ClientType(str, Enum):
    """Client classification for priority scoring."""

    STAR = "star"
    NEW = "new"
    REGULAR = "regular"
    OCCASIONAL = "occasional"


# ============================================================================
# Geographic Schemas
# ============================================================================


class ZoneBase(BaseModel):
    """Base schema for geographic zones."""

    model_config = ConfigDict(from_attributes=True)

    zone_id: ZoneId
    name: str
    color: str = Field(..., pattern=r"^#[0-9A-Fa-f]{6}$")


class Zone(ZoneBase):
    """Zone schema with all attributes."""

    pass


class LocalityBase(BaseModel):
    """Base schema for localities/neighborhoods."""

    model_config = ConfigDict(from_attributes=True)

    locality_id: str
    name: str
    zone_id: ZoneId
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class Locality(LocalityBase):
    """Locality schema with all attributes."""

    pass


# ============================================================================
# Product Schemas
# ============================================================================


class ProductBase(BaseModel):
    """Base schema for products."""

    model_config = ConfigDict(from_attributes=True)

    product_id: str
    name: str
    bag_type: BagType
    packs_per_pallet: int = Field(..., gt=0)
    description: Optional[str] = None


class Product(ProductBase):
    """Product schema with all attributes."""

    pass


# ============================================================================
# Client Schemas
# ============================================================================


class ClientBase(BaseModel):
    """Base schema for clients."""

    model_config = ConfigDict(from_attributes=True)

    client_id: str
    business_name: str
    tax_id: str
    billing_address: str
    zone_id: ZoneId
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    time_window_start: Optional[time] = None
    time_window_end: Optional[time] = None
    is_star_client: bool = False
    is_new_client: bool = False
    first_order_date: Optional[date] = None


class ClientCreate(ClientBase):
    """Schema for creating a new client."""

    pass


class Client(ClientBase):
    """Client schema with all attributes."""

    pass


# ============================================================================
# Order Schemas
# ============================================================================


class OrderItemBase(BaseModel):
    """Base schema for order line items."""

    model_config = ConfigDict(from_attributes=True)

    product_id: str
    quantity_packs: int = Field(..., gt=0)
    pallets: float = Field(..., gt=0)


class OrderItemCreate(OrderItemBase):
    """Schema for creating an order item."""

    pass


class OrderItem(OrderItemBase):
    """Order item schema with all attributes."""

    item_id: int
    order_id: str


class OrderBase(BaseModel):
    """Base schema for orders."""

    model_config = ConfigDict(from_attributes=True)

    order_id: str
    client_id: str
    issue_date: date
    delivery_deadline: date
    delivery_address: str
    delivery_latitude: float = Field(..., ge=-90, le=90)
    delivery_longitude: float = Field(..., ge=-180, le=180)
    delivery_zone_id: ZoneId
    total_amount: float = Field(..., ge=0)
    payment_status: PaymentStatus = PaymentStatus.PENDING
    is_mandatory: bool = False
    total_pallets: float = Field(..., gt=0)
    priority_score: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING

    @field_validator("delivery_deadline")
    @classmethod
    def deadline_after_issue(cls, v: date, info) -> date:
        """Validate that delivery deadline is after or equal to issue date."""
        if "issue_date" in info.data and v < info.data["issue_date"]:
            raise ValueError("delivery_deadline must be after or equal to issue_date")
        return v


class OrderCreate(OrderBase):
    """Schema for creating a new order."""

    items: list[OrderItemCreate] = Field(default_factory=list)


class Order(OrderBase):
    """Order schema with all attributes."""

    items: list[OrderItem] = Field(default_factory=list)


# ============================================================================
# Dispatch Schemas
# ============================================================================


class DispatchOrderBase(BaseModel):
    """Base schema for dispatch-order relationship."""

    model_config = ConfigDict(from_attributes=True)

    dispatch_id: str
    order_id: str
    visit_sequence: int = Field(..., ge=1)
    estimated_arrival_time: Optional[time] = None


class DispatchOrder(DispatchOrderBase):
    """Dispatch-order relationship schema."""

    pass


class DispatchBase(BaseModel):
    """Base schema for dispatches."""

    model_config = ConfigDict(from_attributes=True)

    dispatch_id: str
    dispatch_date: date
    total_pallets: float = Field(..., ge=0, le=8)
    utilization_percentage: float = Field(..., ge=0, le=100)
    total_priority_score: float = Field(..., ge=0)
    estimated_distance_km: Optional[float] = Field(default=None, ge=0)
    estimated_time_min: Optional[int] = Field(default=None, ge=0)
    status: DispatchStatus = DispatchStatus.CANDIDATE


class DispatchCreate(DispatchBase):
    """Schema for creating a new dispatch."""

    order_ids: list[str] = Field(default_factory=list)


class Dispatch(DispatchBase):
    """Dispatch schema with all attributes."""

    orders: list[DispatchOrder] = Field(default_factory=list)


# ============================================================================
# Geocoding Cache Schema
# ============================================================================


class GeocodingCacheEntry(BaseModel):
    """Schema for geocoding cache entries."""

    model_config = ConfigDict(from_attributes=True)

    address_hash: str
    raw_address: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    locality: Optional[str] = None
    zone_id: Optional[ZoneId] = None
    confidence: GeocodingConfidence = GeocodingConfidence.MEDIUM


# ============================================================================
# Processed Receipt Schema
# ============================================================================


class ProcessedReceipt(BaseModel):
    """Schema for processed receipt records."""

    model_config = ConfigDict(from_attributes=True)

    receipt_id: str
    source_file: str
    raw_extraction: dict
    generated_order_id: Optional[str] = None
    extraction_confidence: float = Field(..., ge=0, le=1)


# ============================================================================
# Priority Scoring Configuration
# ============================================================================


class ScoringWeights(BaseModel):
    """Configuration for priority scoring weights."""

    urgency: float = Field(default=0.40, ge=0, le=1)
    payment: float = Field(default=0.25, ge=0, le=1)
    client_type: float = Field(default=0.20, ge=0, le=1)
    age: float = Field(default=0.15, ge=0, le=1)

    @field_validator("age")
    @classmethod
    def weights_sum_to_one(cls, v: float, info) -> float:
        """Validate that all weights sum to approximately 1.0."""
        total = v
        for field in ["urgency", "payment", "client_type"]:
            if field in info.data:
                total += info.data[field]
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        return v


# ============================================================================
# Dispatch Candidate Schema (for optimization output)
# ============================================================================


class DispatchCandidate(BaseModel):
    """Schema for dispatch candidate output from optimizer."""

    candidate_id: int
    order_ids: list[str]
    total_pallets: float
    utilization_percentage: float
    total_priority_score: float
    zone_composition: dict[str, float]  # zone_id -> percentage
    estimated_distance_km: Optional[float] = None
    estimated_time_min: Optional[int] = None
    route_sequence: Optional[list[str]] = None  # ordered list of order_ids


# ============================================================================
# Extraction Schemas (for LLM-based document extraction)
# ============================================================================


class NormalizedBagType(str, Enum):
    """Normalized bag type values from extraction."""

    LARGE = "large"
    MEDIUM = "medium"
    SMALL = "small"
    UNKNOWN = "unknown"


class ExtractedClient(BaseModel):
    """Schema for client information extracted from receipts."""

    business_name: Optional[str] = None
    tax_id: Optional[str] = None
    delivery_address: Optional[str] = None


class ExtractedDocument(BaseModel):
    """Schema for document metadata extracted from receipts."""

    issue_date: Optional[date] = None
    document_number: Optional[str] = None

    @field_validator("issue_date", mode="before")
    @classmethod
    def parse_date(cls, v):
        """Parse date from string if needed."""
        if v is None:
            return None
        if isinstance(v, date):
            return v
        if isinstance(v, str):
            try:
                return datetime.strptime(v, "%Y-%m-%d").date()
            except ValueError:
                return None
        return None


class ExtractedItem(BaseModel):
    """Schema for order item extracted from receipts."""

    bag_type_raw: str
    bag_type_normalized: NormalizedBagType = NormalizedBagType.UNKNOWN
    quantity_packs: Optional[int] = None


class ExtractedTotals(BaseModel):
    """Schema for totals extracted from receipts."""

    total_amount: Optional[float] = None
    total_packs: Optional[int] = None


class ExtractedReceipt(BaseModel):
    """Full extraction result from a receipt document."""

    extraction_confidence: float = Field(..., ge=0.0, le=1.0)
    client: ExtractedClient
    document: ExtractedDocument
    items: list[ExtractedItem] = Field(default_factory=list)
    totals: ExtractedTotals = Field(default_factory=ExtractedTotals)
    extraction_notes: Optional[str] = None
    requires_review: bool = False

    @field_validator("requires_review", mode="after")
    @classmethod
    def check_requires_review(cls, v, info):
        """Automatically set requires_review based on data quality."""
        data = info.data
        if data.get("extraction_confidence", 1.0) < 0.7:
            return True
        client = data.get("client")
        if client:
            if client.business_name is None or client.delivery_address is None:
                return True
        items = data.get("items", [])
        for item in items:
            if item.bag_type_normalized == NormalizedBagType.UNKNOWN:
                return True
            if item.quantity_packs is None:
                return True
        return v


# ============================================================================
# Geocoding Result Schema
# ============================================================================


class GeocodingResult(BaseModel):
    """Result from geocoding an address."""

    model_config = ConfigDict(from_attributes=True)

    address_hash: str
    raw_address: str
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    locality: Optional[str] = None
    zone_id: Optional[str] = None
    confidence: GeocodingConfidence = GeocodingConfidence.LOW
    success: bool = False


# ============================================================================
# Processing Result Schema
# ============================================================================


class ProcessingResult(BaseModel):
    """Result from processing a single receipt."""

    receipt_path: str
    success: bool
    client_id: Optional[str] = None
    client_is_new: bool = False
    order_id: Optional[str] = None
    order_is_duplicate: bool = False
    extraction: Optional[ExtractedReceipt] = None
    geocoding: Optional[GeocodingResult] = None
    error_message: Optional[str] = None
    processing_time_seconds: float = 0.0