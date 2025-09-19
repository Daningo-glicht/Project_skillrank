# app/pdf_utils.py
import fitz  # pymupdf
import re

def extract_pdf_text_and_meta(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    # Heuristics: title = first big line, authors = line after title, abstract = find "abstract"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    title = lines[0] if lines else ""
    authors = lines[1] if len(lines) > 1 else ""
    # find abstract block
    m = re.search(r'(?i)\babstract\b[:\s]*(.+?)(?=\n\s*(?:1\.|introduction|keywords)\b)', text, re.S)
    abstract = m.group(1).strip() if m else ""
    return title, authors, abstract, text
