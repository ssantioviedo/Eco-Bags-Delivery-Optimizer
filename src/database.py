"""Database module for the Delivery Optimization System.

This module provides SQLAlchemy models and database utilities for managing
zones, localities, clients, products, orders, and dispatches.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Time,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


# ============================================================================
# SQLAlchemy Models
# ============================================================================


class ZoneModel(Base):
    """SQLAlchemy model for geographic zones."""

    __tablename__ = "zones"

    zone_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    color = Column(String, nullable=False)

    # Relationships
    localities = relationship("LocalityModel", back_populates="zone")
    clients = relationship("ClientModel", back_populates="zone")


class LocalityModel(Base):
    """SQLAlchemy model for localities/neighborhoods."""

    __tablename__ = "localities"

    locality_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    zone_id = Column(String, ForeignKey("zones.zone_id"), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    # Relationships
    zone = relationship("ZoneModel", back_populates="localities")


class ClientModel(Base):
    """SQLAlchemy model for clients."""

    __tablename__ = "clients"

    client_id = Column(String, primary_key=True)
    business_name = Column(String, nullable=False)
    tax_id = Column(String, nullable=False)
    billing_address = Column(String, nullable=False)
    zone_id = Column(String, ForeignKey("zones.zone_id"), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    time_window_start = Column(Time, nullable=True)
    time_window_end = Column(Time, nullable=True)
    is_star_client = Column(Boolean, default=False)
    is_new_client = Column(Boolean, default=False)
    first_order_date = Column(Date, nullable=True)

    # Relationships
    zone = relationship("ZoneModel", back_populates="clients")
    orders = relationship("OrderModel", back_populates="client")


class ProductModel(Base):
    """SQLAlchemy model for products."""

    __tablename__ = "products"

    product_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    bag_type = Column(String, nullable=False)
    packs_per_pallet = Column(Integer, nullable=False)
    description = Column(Text, nullable=True)

    # Relationships
    order_items = relationship("OrderItemModel", back_populates="product")


class OrderModel(Base):
    """SQLAlchemy model for orders."""

    __tablename__ = "orders"

    order_id = Column(String, primary_key=True)
    client_id = Column(String, ForeignKey("clients.client_id"), nullable=False)
    issue_date = Column(Date, nullable=False)
    delivery_deadline = Column(Date, nullable=False)
    delivery_address = Column(String, nullable=False)
    delivery_latitude = Column(Float, nullable=False)
    delivery_longitude = Column(Float, nullable=False)
    delivery_zone_id = Column(String, ForeignKey("zones.zone_id"), nullable=False)
    total_amount = Column(Float, nullable=False)
    payment_status = Column(String, default="pending")
    is_mandatory = Column(Boolean, default=False)
    total_pallets = Column(Float, nullable=False)
    priority_score = Column(Float, nullable=True)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    client = relationship("ClientModel", back_populates="orders")
    items = relationship("OrderItemModel", back_populates="order", cascade="all, delete-orphan")
    dispatch_orders = relationship("DispatchOrderModel", back_populates="order")


class OrderItemModel(Base):
    """SQLAlchemy model for order line items."""

    __tablename__ = "order_items"

    item_id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String, ForeignKey("orders.order_id"), nullable=False)
    product_id = Column(String, ForeignKey("products.product_id"), nullable=False)
    quantity_packs = Column(Integer, nullable=False)
    pallets = Column(Float, nullable=False)

    # Relationships
    order = relationship("OrderModel", back_populates="items")
    product = relationship("ProductModel", back_populates="order_items")


class DispatchModel(Base):
    """SQLAlchemy model for dispatches."""

    __tablename__ = "dispatches"

    dispatch_id = Column(String, primary_key=True)
    dispatch_date = Column(Date, nullable=False)
    total_pallets = Column(Float, nullable=False)
    utilization_percentage = Column(Float, nullable=False)
    total_priority_score = Column(Float, nullable=False)
    estimated_distance_km = Column(Float, nullable=True)
    estimated_time_min = Column(Integer, nullable=True)
    status = Column(String, default="candidate")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    dispatch_orders = relationship(
        "DispatchOrderModel", back_populates="dispatch", cascade="all, delete-orphan"
    )


class DispatchOrderModel(Base):
    """SQLAlchemy model for dispatch-order relationship."""

    __tablename__ = "dispatch_orders"

    dispatch_id = Column(String, ForeignKey("dispatches.dispatch_id"), primary_key=True)
    order_id = Column(String, ForeignKey("orders.order_id"), primary_key=True)
    visit_sequence = Column(Integer, nullable=False)
    estimated_arrival_time = Column(Time, nullable=True)

    # Relationships
    dispatch = relationship("DispatchModel", back_populates="dispatch_orders")
    order = relationship("OrderModel", back_populates="dispatch_orders")


class GeocodingCacheModel(Base):
    """SQLAlchemy model for geocoding cache."""

    __tablename__ = "geocoding_cache"

    address_hash = Column(String, primary_key=True)
    raw_address = Column(Text, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    locality = Column(String, nullable=True)
    zone_id = Column(String, nullable=True)
    confidence = Column(String, default="medium")
    cached_at = Column(DateTime, default=datetime.utcnow)


class ProcessedReceiptModel(Base):
    """SQLAlchemy model for processed receipts."""

    __tablename__ = "processed_receipts"

    receipt_id = Column(String, primary_key=True)
    source_file = Column(String, nullable=False)
    processed_at = Column(DateTime, default=datetime.utcnow)
    raw_extraction = Column(Text, nullable=False)  # JSON string
    generated_order_id = Column(String, ForeignKey("orders.order_id"), nullable=True)
    extraction_confidence = Column(Float, nullable=False)


# ============================================================================
# Database Utilities
# ============================================================================


class DatabaseManager:
    """Manager class for database operations."""

    def __init__(self, db_path: str | Path):
        """Initialize the database manager.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(self.engine)

    def drop_tables(self) -> None:
        """Drop all database tables."""
        Base.metadata.drop_all(self.engine)

    def get_session(self) -> Session:
        """Get a new database session.

        Returns:
            A SQLAlchemy session object.
        """
        return self.SessionLocal()

    def load_zones_from_json(self, json_path: str | Path) -> list[ZoneModel]:
        """Load zones from a JSON file into the database.

        Args:
            json_path: Path to the zones.json file.

        Returns:
            List of created ZoneModel instances.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            zones_data = json.load(f)

        zones = []
        with self.get_session() as session:
            for zone_id, zone_info in zones_data.items():
                zone = ZoneModel(
                    zone_id=zone_id,
                    name=zone_info["name"],
                    color=zone_info["color"],
                )
                session.merge(zone)
                zones.append(zone)
            session.commit()
        return zones

    def load_localities_from_json(self, json_path: str | Path) -> list[LocalityModel]:
        """Load localities from a JSON file into the database.

        Args:
            json_path: Path to the localities.json file.

        Returns:
            List of created LocalityModel instances.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            localities_data = json.load(f)

        localities = []
        with self.get_session() as session:
            for locality_id, locality_info in localities_data.items():
                locality = LocalityModel(
                    locality_id=locality_id,
                    name=locality_info["name"],
                    zone_id=locality_info["zone_id"],
                    latitude=locality_info["latitude"],
                    longitude=locality_info["longitude"],
                )
                session.merge(locality)
                localities.append(locality)
            session.commit()
        return localities

    def get_all_zones(self) -> list[ZoneModel]:
        """Get all zones from the database.

        Returns:
            List of all ZoneModel instances.
        """
        with self.get_session() as session:
            return session.query(ZoneModel).all()

    def get_all_localities(self) -> list[LocalityModel]:
        """Get all localities from the database.

        Returns:
            List of all LocalityModel instances.
        """
        with self.get_session() as session:
            return session.query(LocalityModel).all()

    def get_all_clients(self) -> list[ClientModel]:
        """Get all clients from the database.

        Returns:
            List of all ClientModel instances.
        """
        with self.get_session() as session:
            return session.query(ClientModel).all()

    def get_all_products(self) -> list[ProductModel]:
        """Get all products from the database.

        Returns:
            List of all ProductModel instances.
        """
        with self.get_session() as session:
            return session.query(ProductModel).all()

    def get_all_orders(self) -> list[OrderModel]:
        """Get all orders from the database.

        Returns:
            List of all OrderModel instances.
        """
        with self.get_session() as session:
            return session.query(OrderModel).all()

    def get_pending_orders(self) -> list[OrderModel]:
        """Get all pending orders from the database.

        Returns:
            List of pending OrderModel instances.
        """
        with self.get_session() as session:
            return session.query(OrderModel).filter(OrderModel.status == "pending").all()

    def add_product(self, product: ProductModel) -> ProductModel:
        """Add a product to the database.

        Args:
            product: ProductModel instance to add.

        Returns:
            The added ProductModel instance.
        """
        with self.get_session() as session:
            session.merge(product)
            session.commit()
        return product

    def add_client(self, client: ClientModel) -> ClientModel:
        """Add a client to the database.

        Args:
            client: ClientModel instance to add.

        Returns:
            The added ClientModel instance.
        """
        with self.get_session() as session:
            session.merge(client)
            session.commit()
        return client

    def add_order(self, order: OrderModel) -> OrderModel:
        """Add an order to the database.

        Args:
            order: OrderModel instance to add.

        Returns:
            The added OrderModel instance.
        """
        with self.get_session() as session:
            session.merge(order)
            session.commit()
        return order

    def add_order_item(self, item: OrderItemModel) -> OrderItemModel:
        """Add an order item to the database.

        Args:
            item: OrderItemModel instance to add.

        Returns:
            The added OrderItemModel instance.
        """
        with self.get_session() as session:
            session.add(item)
            session.commit()
            session.refresh(item)
        return item


def get_database_manager(db_path: Optional[str | Path] = None) -> DatabaseManager:
    """Get a DatabaseManager instance with the default or specified path.

    Args:
        db_path: Optional path to the database file. Defaults to data/processed/delivery.db.

    Returns:
        A DatabaseManager instance.
    """
    if db_path is None:
        # Default path relative to project root
        project_root = Path(__file__).parent.parent
        db_path = project_root / "data" / "processed" / "delivery.db"
    return DatabaseManager(db_path)
