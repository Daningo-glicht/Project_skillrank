# app/crud.py
from sqlalchemy.orm import Session
from app.models import Paper, Summary

# ----------------------------
# Paper CRUD
# ----------------------------
def create_paper(db: Session, filename: str, title: str, authors: str, abstract: str, text: str):
    paper = Paper(
        filename=filename,
        title=title,
        authors=authors,
        abstract=abstract,
        text=text
    )
    db.add(paper)
    db.commit()
    db.refresh(paper)
    return paper


def get_paper(db: Session, paper_id: int):
    return db.query(Paper).filter(Paper.id == paper_id).first()


def list_papers(db: Session, skip: int = 0, limit: int = 20):
    return db.query(Paper).offset(skip).limit(limit).all()


# ----------------------------
# Summary CRUD
# ----------------------------
def create_summary(db: Session, paper_id: int, short_summary: str,
                   contributions: str, methods: str, results: str,
                   limitations: str, future_work: str):
    summary = Summary(
        paper_id=paper_id,
        short_summary=short_summary,
        contributions=contributions,
        methods=methods,
        results=results,
        limitations=limitations,
        future_work=future_work
    )
    db.add(summary)
    db.commit()
    db.refresh(summary)
    return summary


def get_summary_by_paper(db: Session, paper_id: int):
    return db.query(Summary).filter(Summary.paper_id == paper_id).first()
