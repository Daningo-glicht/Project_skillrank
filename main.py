# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil, os
from app.pdf_utils import extract_pdf_text_and_meta
from app.models import Paper
from app.crud import get_paper, create_summary
from app.llm import summarize_text
from app.gap_analysis import detect_and_resolve

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------
# Upload endpoint
# -------------------------------
@app.post("/upload")
async def upload_paper(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF allowed")
    
    dest = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    title, authors, abstract, full_text = extract_pdf_text_and_meta(dest)
    
    # Create paper entry in DB
    paper = Paper(
        filename=file.filename,
        title=title,
        authors=authors,
        abstract=abstract,
        text=full_text
    )
    from app.database import SessionLocal
    db = SessionLocal()
    db.add(paper)
    db.commit()
    db.refresh(paper)
    db.close()
    
    return {
        "paper_id": paper.id,
        "title": title,
        "authors": authors,
        "abstract": abstract
    }

# -------------------------------
# Summarize endpoint
# -------------------------------
@app.post("/papers/{paper_id}/summarize")
async def summarize_paper(paper_id: int):
    paper = get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    summary = summarize_text(paper.text)
    create_summary(paper_id, summary)
    
    return {"paper_id": paper_id, "summary": summary}

# -------------------------------
# Gap Analysis endpoint
# -------------------------------
@app.post("/papers/{paper_id}/gap_analysis")
async def gap_analysis(paper_id: int):
    paper = get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    gaps = detect_and_resolve(paper)
    return {"paper_id": paper_id, "gaps": gaps}
