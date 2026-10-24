"""PostgreSQL prediction history with short, independent transactions."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    Integer,
    String,
    Uuid,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("status IN ('success', 'rejected', 'error')"),
        CheckConstraint("threshold >= 0 AND threshold <= 1"),
        CheckConstraint(
            "ai_probability IS NULL OR (ai_probability >= 0 AND ai_probability <= 1)"
        ),
        CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    filename: Mapped[str | None] = mapped_column(String(180))
    image_sha256: Mapped[str | None] = mapped_column(String(64))
    image_bytes: Mapped[int | None] = mapped_column(Integer)
    image_width: Mapped[int | None] = mapped_column(Integer)
    image_height: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), index=True)
    error_code: Mapped[str | None] = mapped_column(String(40))
    error_message: Mapped[str | None] = mapped_column(String(240))
    label: Mapped[str | None] = mapped_column(String(20))
    ai_probability: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    processing_ms: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(30))
    model_run_id: Mapped[str] = mapped_column(String(64))
    model_sha256: Mapped[str] = mapped_column(String(64))
    preprocessing_policy: Mapped[str] = mapped_column(String(80))

    def as_dict(self):
        timestamp = (
            self.created_at.replace(tzinfo=UTC)
            if self.created_at.tzinfo is None
            else self.created_at
        )
        return {
            "id": str(self.id),
            "created_at": timestamp.isoformat(),
            "filename": self.filename,
            "status": self.status,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "label": self.label,
            "ai_probability": self.ai_probability,
            "confidence": self.confidence,
            "threshold": self.threshold,
            "processing_ms": self.processing_ms,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_run_id": self.model_run_id,
            "model_sha256": self.model_sha256,
            "preprocessing_policy": self.preprocessing_policy,
            "image_sha256": self.image_sha256,
        }


class HistoryStore:
    def __init__(self, url, testing=False):
        options = {"pool_pre_ping": True, "hide_parameters": True}
        if testing and url.startswith("sqlite:"):
            options.update(
                poolclass=StaticPool, connect_args={"check_same_thread": False}
            )
        elif not url.startswith("postgresql+psycopg://"):
            raise ValueError("DATABASE_URL must use postgresql+psycopg")
        else:
            options.update(
                pool_size=3,
                max_overflow=1,
                pool_timeout=3,
                connect_args={"connect_timeout": 3},
            )
        self.engine = create_engine(url, **options)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def initialize(self):
        Base.metadata.create_all(self.engine)

    def add(self, **values):
        with self.sessions.begin() as session:
            row = Prediction(**values)
            session.add(row)
        return row

    def get(self, identifier):
        with self.sessions() as session:
            return session.get(Prediction, identifier)

    def page(self, page, per_page):
        with self.sessions() as session:
            total = session.scalar(select(func.count()).select_from(Prediction))
            rows = session.scalars(
                select(Prediction)
                .order_by(Prediction.created_at.desc(), Prediction.id.desc())
                .offset((page - 1) * per_page)
                .limit(per_page)
            ).all()
        return rows, total

    def healthy(self):
        with self.engine.connect() as connection:
            return connection.execute(text("SELECT 1")).scalar_one() == 1
