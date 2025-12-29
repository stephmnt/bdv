from __future__ import annotations

import os
from typing import Optional

import sqlalchemy as sa
from sqlalchemy import Column, Date, Float, ForeignKey, Integer, MetaData, String, Table, UniqueConstraint
from sqlalchemy.engine import Engine

metadata = MetaData()

communes = Table(
    "communes",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name_normalized", String(255), nullable=True),
    Column("insee_code", String(12), nullable=False, unique=True, index=True),
)

bureaux = Table(
    "bureaux",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("commune_id", Integer, ForeignKey("communes.id"), nullable=False),
    Column("bureau_code", String(32), nullable=False),
    Column("bureau_label", String(255), nullable=True),
    UniqueConstraint("commune_id", "bureau_code", name="uq_bureau_commune_code"),
)

elections = Table(
    "elections",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("election_type", String(32), nullable=False),
    Column("election_year", Integer, nullable=False),
    Column("round", Integer, nullable=True),
    Column("date", Date, nullable=True),
    UniqueConstraint("election_type", "election_year", "round", name="uq_election_unique"),
)

categories = Table(
    "categories",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(64), nullable=False, unique=True),
)

results_local = Table(
    "results_local",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("bureau_id", Integer, ForeignKey("bureaux.id"), nullable=False),
    Column("election_id", Integer, ForeignKey("elections.id"), nullable=False),
    Column("category_id", Integer, ForeignKey("categories.id"), nullable=False),
    Column("share_pct", Float, nullable=True),
    Column("votes", Float, nullable=True),
    Column("expressed", Float, nullable=True),
    Column("turnout_pct", Float, nullable=True),
    UniqueConstraint("bureau_id", "election_id", "category_id", name="uq_local_bureau_election_category"),
)

results_national = Table(
    "results_national",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("election_id", Integer, ForeignKey("elections.id"), nullable=False),
    Column("category_id", Integer, ForeignKey("categories.id"), nullable=False),
    Column("share_pct", Float, nullable=True),
    Column("votes", Float, nullable=True),
    Column("expressed", Float, nullable=True),
    Column("turnout_pct", Float, nullable=True),
    UniqueConstraint("election_id", "category_id", name="uq_nat_election_category"),
)


def _build_url_from_env() -> Optional[str]:
    user = os.getenv("DB_USER") or os.getenv("POSTGRES_USER")
    password = os.getenv("DB_PASSWORD") or os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", os.getenv("POSTGRES_PORT", "5432"))
    db_name = os.getenv("DB_NAME") or os.getenv("POSTGRES_DB")
    if user and password and db_name:
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
    return None


def get_engine(url: Optional[str] = None) -> Engine:
    db_url = url or os.getenv("DATABASE_URL") or _build_url_from_env()
    if not db_url:
        raise RuntimeError("DATABASE_URL or DB_* env vars must be set.")
    return sa.create_engine(db_url)


def create_schema(engine: Engine) -> None:
    metadata.create_all(engine)
