# app/database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ----------------------------
# Database URL (default SQLite)
# ----------------------------
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./papers.db")

# For SQLite, allow multithreading
connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}

# Create engine
engine = create_engine(DB_URL, connect_args=connect_args)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
