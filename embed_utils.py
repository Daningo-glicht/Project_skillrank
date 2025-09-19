# app/embed_utils.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# ----------------------------
# Load embedding model
# ----------------------------
# Good small model for semantic search
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Chunking function
# ----------------------------
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50):
    """
    Split text into overlapping word chunks.
    Default: 200 words per chunk with 50-word overlap.
    """
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# ----------------------------
# Build FAISS index
# ----------------------------
def build_faiss_index(chunks):
    """
    Build FAISS index from list of text chunks.
    Returns (index, embeddings, mapping).
    """
    embs = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    mapping = {i: chunk for i, chunk in enumerate(chunks)}
    return index, embs, mapping

# ----------------------------
# Query embedding
# ----------------------------
def embed_query(query: str):
    """
    Embed a query string into vector space.
    """
    return model.encode([query], convert_to_numpy=True)[0]

# ----------------------------
# FAISS search
# ----------------------------
def faiss_search(index, query_emb, k=5):
    """
    Perform similarity search in FAISS index.
    Returns (indices, distances).
    """
    D, I = index.search(np.array([query_emb]).astype("float32"), k)
    return I[0], D[0]
