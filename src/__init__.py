"""Eco-Bags Delivery Optimizer - Source Package.

This package provides modules for delivery optimization including:
- schemas: Pydantic models for data validation
- database: SQLAlchemy models and database utilities
- extraction: LLM-based document parsing
- scoring: Priority calculation
- optimizer: Dispatch generation (knapsack)
- routing: TSP for route optimization
- geo: Geocoding and distance utilities
- pipeline: End-to-end orchestration
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
    GeocodingCacheEntry,
    GeocodingConfidence,
    Locality,
    Order,
    OrderCreate,
    OrderItem,
    OrderStatus,
    PaymentStatus,
    ProcessedReceipt,
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
]
