"""Eco-Bags Delivery Optimizer - Source Package.

This package provides modules for delivery optimization including:
- schemas: Pydantic models for data validation
- database: SQLAlchemy models and database utilities
- extraction: LLM-based receipt parsing (PDF → structured data)
- geo: Geocoding and distance utilities

Planned modules (not yet implemented in src/): scoring, optimizer, routing, pipeline.
"""

from .database import (
    Base,
    ClientModel,
    DatabaseManager,
    DispatchModel,
    DispatchOrderModel,
    GeocodingCacheModel,
    LocalityModel,
    OrderItemModel,
    OrderModel,
    ProcessedReceiptModel,
    ProductModel,
    ZoneModel,
    get_database_manager,
)
from .schemas import (
    BagType,
    Client,
    ClientCreate,
    ClientType,
    Dispatch,
    DispatchCandidate,
    DispatchCreate,
    DispatchOrder,
    DispatchStatus,
    ExtractedClient,
    ExtractedDocument,
    ExtractedItem,
    ExtractedReceipt,
    ExtractedTotals,
    GeocodingCacheEntry,
    GeocodingConfidence,
    GeocodingResult,
    Locality,
    NormalizedBagType,
    Order,
    OrderCreate,
    OrderItem,
    OrderStatus,
    PaymentStatus,
    ProcessedReceipt,
    ProcessingResult,
    Product,
    ScoringWeights,
    Zone,
    ZoneId,
)

__version__ = "0.1.0"

__all__ = [
    # Database
    "Base",
    "ClientModel",
    "DatabaseManager",
    "DispatchModel",
    "DispatchOrderModel",
    "GeocodingCacheModel",
    "LocalityModel",
    "OrderItemModel",
    "OrderModel",
    "ProcessedReceiptModel",
    "ProductModel",
    "ZoneModel",
    "get_database_manager",
    # Schemas - Enums
    "BagType",
    "ClientType",
    "DispatchStatus",
    "GeocodingConfidence",
    "OrderStatus",
    "PaymentStatus",
    "ZoneId",
    # Schemas - Models
    "Client",
    "ClientCreate",
    "Dispatch",
    "DispatchCandidate",
    "DispatchCreate",
    "DispatchOrder",
    "GeocodingCacheEntry",
    "Locality",
    "Order",
    "OrderCreate",
    "OrderItem",
    "ProcessedReceipt",
    "Product",
    "ScoringWeights",
    "Zone",
    # Extraction Schemas
    "ExtractedClient",
    "ExtractedDocument",
    "ExtractedItem",
    "ExtractedReceipt",
    "ExtractedTotals",
    "GeocodingResult",
    "NormalizedBagType",
    "ProcessingResult",
]
