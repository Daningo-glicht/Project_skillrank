# app/models.py
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class Paper(Base):
    __tablename__ = "papers"
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    title = Column(String)
    authors = Column(String)
    abstract = Column(Text)
    text = Column(Text)            # full extracted text
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)

class Summary(Base):
    __tablename__ = "summaries"
    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, nullable=False)
    short_summary = Column(Text)
    contributions = Column(Text)   # JSON encoded list or newline bullets
    methods = Column(Text)
    results = Column(Text)
    limitations = Column(Text)
    future_work = Column(Text)
