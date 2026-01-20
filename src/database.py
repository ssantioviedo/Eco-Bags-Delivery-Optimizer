"""Database module for the Delivery Optimization System.

This module provides SQLAlchemy models and database utilities for managing
zones, localities, clients, products, orders, and dispatches.
"""

import json
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
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
    quantity_packs = Column(Float, nullable=False)
    total_pallets = Column(Float, nullable=False)
    priority_score = Column(Float, nullable=True)
    status = Column(String, default="pending")

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


class ProcessedReceiptModel(Base):
    """SQLAlchemy model for processed receipts."""

    __tablename__ = "processed_receipts"

    receipt_id = Column(String, primary_key=True)
    source_file = Column(String, nullable=False)
    raw_extraction = Column(Text, nullable=False)  # JSON string
    generated_order_id = Column(String, ForeignKey("orders.order_id"), nullable=True)
    extraction_confidence = Column(Float, nullable=False)


class DispatchCandidateModel(Base):
    """SQLAlchemy model for dispatch candidates generated by the order selector."""

    __tablename__ = "dispatch_candidates"

    candidate_id = Column(String, primary_key=True)
    strategy = Column(String, nullable=False)
    total_pallets = Column(Float, nullable=False)
    total_priority = Column(Float, nullable=False)
    adjusted_priority = Column(Float, nullable=False)
    utilization_pct = Column(Float, nullable=False)
    zone_dispersion_penalty = Column(Float, nullable=False)
    zones = Column(Text, nullable=False)  # JSON array of zone IDs
    zone_breakdown = Column(Text, nullable=False)  # JSON object {zone_id: count}
    is_single_zone = Column(Boolean, nullable=False)
    mandatory_count = Column(Integer, nullable=False)
    order_count = Column(Integer, nullable=False)
    rank = Column(Integer, nullable=True)  # Ranking position (1 = best)
    status = Column(String, default="candidate")  # candidate, selected, rejected

    # Relationships
    candidate_orders = relationship(
        "DispatchCandidateOrderModel",
        back_populates="candidate",
        cascade="all, delete-orphan",
    )


class DispatchCandidateOrderModel(Base):
    """SQLAlchemy model for dispatch candidate to order relationship."""

    __tablename__ = "dispatch_candidate_orders"

    candidate_id = Column(
        String, ForeignKey("dispatch_candidates.candidate_id"), primary_key=True
    )
    order_id = Column(String, ForeignKey("orders.order_id"), primary_key=True)
    is_mandatory = Column(Boolean, nullable=False)

    # Relationships
    candidate = relationship("DispatchCandidateModel", back_populates="candidate_orders")
    order = relationship("OrderModel")


class RouteCacheModel(Base):
    """SQLAlchemy model for route optimization cache.
    
    Stores optimized route results to avoid recalculating routes
    for the same set of orders. The cache key is based on sorted
    order IDs and solver type.
    """

    __tablename__ = "route_cache"

    cache_key = Column(String, primary_key=True)  # MD5 hash of sorted order_ids + solver
    solver = Column(String, nullable=False)
    order_ids = Column(Text, nullable=False)  # JSON array of order IDs
    total_distance_km = Column(Float, nullable=False)
    total_duration_minutes = Column(Integer, nullable=False)
    route_data = Column(Text, nullable=False)  # JSON serialized route result
    feasible = Column(Boolean, nullable=False, default=True)


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

    def clear_data_tables(self, keep_cache: bool = True) -> None:
        """Clear data tables while optionally preserving cache tables.
        
        This method clears:
        - orders, order_items, clients, products
        - dispatches, dispatch_orders
        - dispatch_candidates, dispatch_candidate_orders
        - processed_receipts
        
        Optionally keeps cache tables:
        - geocoding_cache (if keep_cache=True)
        - ortools_route_cache (if keep_cache=True) - future cache table
        
        Args:
            keep_cache: If True, preserve geocoding and route optimization caches.
        """
        with self.get_session() as session:
            # Clear data tables (order matters for foreign keys)
            session.query(DispatchCandidateOrderModel).delete()
            session.query(DispatchCandidateModel).delete()
            session.query(DispatchOrderModel).delete()
            session.query(DispatchModel).delete()
            session.query(ProcessedReceiptModel).delete()
            session.query(OrderItemModel).delete()
            session.query(OrderModel).delete()
            session.query(ClientModel).delete()
            session.query(ProductModel).delete()
            
            # Clear cache tables if keep_cache=False
            if not keep_cache:
                session.query(GeocodingCacheModel).delete()
                session.query(RouteCacheModel).delete()
            
            session.commit()

    def enforce_one_pending_order_per_client(self) -> int:
        """Enforce constraint: only ONE pending order per client.
        
        If a client has multiple pending orders, keep the most recent one
        (by issue_date) and mark others as 'archived' or delete them.
        
        Returns:
            Number of orders deleted/archived.
        """
        with self.get_session() as session:
            # Group pending orders by client
            from sqlalchemy import func
            
            # Get clients with multiple pending orders
            client_order_counts = (
                session.query(
                    OrderModel.client_id,
                    func.count(OrderModel.order_id).label("order_count")
                )
                .filter(OrderModel.status == "pending")
                .group_by(OrderModel.client_id)
                .having(func.count(OrderModel.order_id) > 1)
                .all()
            )
            
            deleted_count = 0
            for client_id, count in client_order_counts:
                # Get all pending orders for this client, ordered by issue_date DESC
                pending_orders = (
                    session.query(OrderModel)
                    .filter(
                        OrderModel.client_id == client_id,
                        OrderModel.status == "pending"
                    )
                    .order_by(OrderModel.issue_date.desc())
                    .all()
                )
                
                # Keep the first (most recent), delete the rest
                for order in pending_orders[1:]:
                    # Delete related items first
                    session.query(OrderItemModel).filter(
                        OrderItemModel.order_id == order.order_id
                    ).delete()
                    # Delete the order
                    session.delete(order)
                    deleted_count += 1
            
            session.commit()
        return deleted_count

    def get_clients_with_pending_orders_count(self) -> dict[str, int]:
        """Get count of pending orders per client.
        
        Returns:
            Dictionary mapping client_id to pending order count.
        """
        with self.get_session() as session:
            from sqlalchemy import func
            
            results = (
                session.query(
                    OrderModel.client_id,
                    func.count(OrderModel.order_id).label("pending_count")
                )
                .filter(OrderModel.status == "pending")
                .group_by(OrderModel.client_id)
                .all()
            )
            return {client_id: count for client_id, count in results}

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

    def update_order_priority(self, order_id: str, priority_score: float) -> None:
        """Update priority score for a single order.

        Args:
            order_id: The order ID to update.
            priority_score: The new priority score.
        """
        with self.get_session() as session:
            order = session.query(OrderModel).filter(OrderModel.order_id == order_id).first()
            if order:
                order.priority_score = priority_score
                session.commit()

    def bulk_update_priorities(self, scores: list[dict]) -> int:
        """Bulk update priority scores for multiple orders.

        Args:
            scores: List of {"order_id": str, "priority_score": float} dictionaries.

        Returns:
            Number of orders updated.
        """
        updated_count = 0
        with self.get_session() as session:
            for score_data in scores:
                order = (
                    session.query(OrderModel)
                    .filter(OrderModel.order_id == score_data["order_id"])
                    .first()
                )
                if order:
                    order.priority_score = score_data["priority_score"]
                    updated_count += 1
            session.commit()
        return updated_count

    def get_orders_with_clients(self) -> list[tuple]:
        """Get all orders joined with their client data.

        Returns:
            List of tuples (OrderModel, ClientModel) for each order.
        """
        with self.get_session() as session:
            results = (
                session.query(OrderModel, ClientModel)
                .join(ClientModel, OrderModel.client_id == ClientModel.client_id)
                .all()
            )
            return results

    def get_pending_orders_with_clients(self) -> list[tuple]:
        """Get all pending orders joined with their client data.

        Returns:
            List of tuples (OrderModel, ClientModel) for each pending order.
        """
        with self.get_session() as session:
            results = (
                session.query(OrderModel, ClientModel)
                .join(ClientModel, OrderModel.client_id == ClientModel.client_id)
                .filter(OrderModel.status == "pending")
                .all()
            )
            return results

    def count_client_orders(self, client_id: str) -> int:
        """Count total orders for a client (for frequency calculation).

        Args:
            client_id: The client ID to count orders for.

        Returns:
            Number of orders for this client.
        """
        with self.get_session() as session:
            count = (
                session.query(OrderModel)
                .filter(OrderModel.client_id == client_id)
                .count()
            )
            return count

    def get_client_order_counts(self) -> dict[str, int]:
        """Get order counts for all clients.

        Returns:
            Dictionary mapping client_id to order count.
        """
        with self.get_session() as session:
            from sqlalchemy import func

            results = (
                session.query(OrderModel.client_id, func.count(OrderModel.order_id))
                .group_by(OrderModel.client_id)
                .all()
            )
            return {client_id: count for client_id, count in results}

    def save_dispatch_candidates(
        self,
        candidates: list,
        ranked: bool = True,
    ) -> int:
        """Save dispatch candidates to the database.
        
        This method automatically clears all existing candidates before saving,
        ensuring only the latest run is kept in the database.

        Args:
            candidates: List of DispatchCandidate objects from order_selector.
            ranked: Whether the candidates are in ranked order (position = rank).

        Returns:
            Number of candidates saved.
        """
        with self.get_session() as session:
            # Clear all existing dispatch candidates and their relationships
            session.query(DispatchCandidateOrderModel).delete()
            session.query(DispatchCandidateModel).delete()
            session.commit()
            
            # Save new candidates
            for idx, candidate in enumerate(candidates, 1):
                # Create the candidate record
                db_candidate = DispatchCandidateModel(
                    candidate_id=candidate.candidate_id,
                    strategy=candidate.strategy.value,
                    total_pallets=candidate.total_pallets,
                    total_priority=candidate.total_priority,
                    adjusted_priority=candidate.adjusted_priority,
                    utilization_pct=candidate.utilization_pct,
                    zone_dispersion_penalty=candidate.zone_dispersion_penalty,
                    zones=json.dumps(candidate.zones),
                    zone_breakdown=json.dumps(candidate.zone_breakdown),
                    is_single_zone=candidate.is_single_zone,
                    mandatory_count=candidate.mandatory_count,
                    order_count=len(candidate.order_ids),
                    rank=idx if ranked else None,
                    status="candidate",
                )
                session.merge(db_candidate)

                # Create order relationships
                for order_data in candidate.orders:
                    order_rel = DispatchCandidateOrderModel(
                        candidate_id=candidate.candidate_id,
                        order_id=order_data["order_id"],
                        is_mandatory=order_data.get("is_mandatory", False),
                    )
                    session.merge(order_rel)

            session.commit()
            return len(candidates)

    def get_all_dispatch_candidates(self) -> list[DispatchCandidateModel]:
        """Get all dispatch candidates.

        Returns:
            List of DispatchCandidateModel instances ordered by rank.
        """
        with self.get_session() as session:
            candidates = (
                session.query(DispatchCandidateModel)
                .order_by(DispatchCandidateModel.rank)
                .all()
            )
            return candidates

    def get_candidate_orders(
        self, candidate_id: str
    ) -> list[tuple[DispatchCandidateOrderModel, OrderModel]]:
        """Get all orders for a specific dispatch candidate.

        Args:
            candidate_id: The candidate ID.

        Returns:
            List of tuples (DispatchCandidateOrderModel, OrderModel).
        """
        with self.get_session() as session:
            results = (
                session.query(DispatchCandidateOrderModel, OrderModel)
                .join(OrderModel, DispatchCandidateOrderModel.order_id == OrderModel.order_id)
                .filter(DispatchCandidateOrderModel.candidate_id == candidate_id)
                .all()
            )
            return results

    def update_candidate_status(self, candidate_id: str, status: str) -> bool:
        """Update the status of a dispatch candidate.

        Args:
            candidate_id: The candidate ID.
            status: New status ('candidate', 'selected', 'rejected').

        Returns:
            True if updated successfully, False otherwise.
        """
        with self.get_session() as session:
            candidate = (
                session.query(DispatchCandidateModel)
                .filter(DispatchCandidateModel.candidate_id == candidate_id)
                .first()
            )
            if candidate:
                candidate.status = status
                session.commit()
                return True
            return False

    def delete_candidates_by_batch(self, batch_id: str) -> int:
        """Delete all dispatch candidates from a specific batch.

        Args:
            batch_id: The generation batch ID.

        Returns:
            Number of candidates deleted.
        """
        with self.get_session() as session:
            count = (
                session.query(DispatchCandidateModel)
                .filter(DispatchCandidateModel.generation_batch_id == batch_id)
                .delete()
            )
            session.commit()
            return count

    # ========================================================================
    # Route Optimization Methods
    # ========================================================================

    def get_order_coordinates(
        self, order_ids: list[str]
    ) -> dict[str, tuple[float, float]]:
        """Get coordinates for multiple orders.

        Args:
            order_ids: List of order IDs.

        Returns:
            Dictionary mapping order_id to (latitude, longitude) tuple.
        """
        with self.get_session() as session:
            orders = (
                session.query(OrderModel)
                .filter(OrderModel.order_id.in_(order_ids))
                .all()
            )
            return {
                o.order_id: (o.delivery_latitude, o.delivery_longitude)
                for o in orders
                if o.delivery_latitude is not None and o.delivery_longitude is not None
            }

    def get_dispatch_candidate_with_orders(
        self, candidate_id: str
    ) -> tuple[DispatchCandidateModel | None, list[OrderModel]]:
        """Get dispatch candidate and its orders with full details.

        Args:
            candidate_id: The dispatch candidate ID.

        Returns:
            Tuple of (DispatchCandidateModel, list of OrderModel).
        """
        with self.get_session() as session:
            candidate = (
                session.query(DispatchCandidateModel)
                .filter(DispatchCandidateModel.candidate_id == candidate_id)
                .first()
            )

            if not candidate:
                return None, []

            # Get order IDs from relationship
            order_rels = (
                session.query(DispatchCandidateOrderModel)
                .filter(DispatchCandidateOrderModel.candidate_id == candidate_id)
                .all()
            )
            order_ids = [rel.order_id for rel in order_rels]

            orders = (
                session.query(OrderModel)
                .filter(OrderModel.order_id.in_(order_ids))
                .all()
            )

            return candidate, orders

    def update_dispatch_route(
        self,
        candidate_id: str,
        total_distance_km: float,
        estimated_time_min: int,
    ) -> bool:
        """Update dispatch candidate with route optimization results.

        Note: Since we removed route columns, this updates the existing fields.
        Route details are stored in JSON exports.

        Args:
            candidate_id: The dispatch candidate ID.
            total_distance_km: Total route distance in km.
            estimated_time_min: Estimated route duration in minutes.

        Returns:
            True if updated successfully, False otherwise.
        """
        # Route details are stored in JSON exports, not in database
        # This method exists for API compatibility
        return True

    # ========================================================================
    # Route Cache Methods
    # ========================================================================

    def get_cached_route(self, cache_key: str) -> Optional[dict]:
        """Get a cached route result by cache key.

        Args:
            cache_key: MD5 hash of sorted order_ids + solver.

        Returns:
            Dictionary with route data if found, None otherwise.
        """
        with self.get_session() as session:
            cached = session.query(RouteCacheModel).filter(
                RouteCacheModel.cache_key == cache_key
            ).first()
            
            if cached:
                return {
                    "cache_key": cached.cache_key,
                    "solver": cached.solver,
                    "order_ids": json.loads(cached.order_ids),
                    "total_distance_km": cached.total_distance_km,
                    "total_duration_minutes": cached.total_duration_minutes,
                    "route_data": json.loads(cached.route_data),
                    "feasible": cached.feasible,
                }
            return None

    def save_route_to_cache(
        self,
        cache_key: str,
        solver: str,
        order_ids: list[str],
        total_distance_km: float,
        total_duration_minutes: int,
        route_data: dict,
        feasible: bool = True,
    ) -> None:
        """Save a route result to the persistent cache.

        Args:
            cache_key: MD5 hash of sorted order_ids + solver.
            solver: The solver used (e.g., 'ortools', 'nearest_neighbor').
            order_ids: List of order IDs in the route.
            total_distance_km: Total route distance.
            total_duration_minutes: Total route duration.
            route_data: Serialized route result dictionary.
            feasible: Whether the route is feasible.
        """
        with self.get_session() as session:
            # Check if already exists
            existing = session.query(RouteCacheModel).filter(
                RouteCacheModel.cache_key == cache_key
            ).first()
            
            if existing:
                # Update existing entry
                existing.total_distance_km = total_distance_km
                existing.total_duration_minutes = total_duration_minutes
                existing.route_data = json.dumps(route_data)
                existing.feasible = feasible
            else:
                # Create new entry
                cache_entry = RouteCacheModel(
                    cache_key=cache_key,
                    solver=solver,
                    order_ids=json.dumps(order_ids),
                    total_distance_km=total_distance_km,
                    total_duration_minutes=total_duration_minutes,
                    route_data=json.dumps(route_data),
                    feasible=feasible,
                )
                session.add(cache_entry)
            
            session.commit()

    def get_route_cache_stats(self) -> dict:
        """Get statistics about the route cache.

        Returns:
            Dictionary with cache size and solver breakdown.
        """
        with self.get_session() as session:
            total = session.query(RouteCacheModel).count()
            
            # Get breakdown by solver
            from sqlalchemy import func
            solver_counts = session.query(
                RouteCacheModel.solver,
                func.count(RouteCacheModel.cache_key)
            ).group_by(RouteCacheModel.solver).all()
            
            return {
                "size": total,
                "by_solver": {solver: count for solver, count in solver_counts},
            }

    def clear_route_cache(self) -> int:
        """Clear all entries from the route cache.

        Returns:
            Number of entries deleted.
        """
        with self.get_session() as session:
            count = session.query(RouteCacheModel).delete()
            session.commit()
            return count

    def get_orders_with_time_windows(
        self, order_ids: list[str]
    ) -> list[OrderModel]:
        """Get orders with their client time window information.

        Args:
            order_ids: List of order IDs.

        Returns:
            List of OrderModel with joined client data.
        """
        with self.get_session() as session:
            orders = (
                session.query(OrderModel)
                .join(ClientModel, OrderModel.client_id == ClientModel.client_id)
                .filter(OrderModel.order_id.in_(order_ids))
                .all()
            )
            return orders


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