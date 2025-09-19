# app/gap_analysis.py
import faiss
import numpy as np
from app.llm import call_local_llm  # reuse mistral wrapper
from app.embed_utils import embed_query  # your embedding util

# ----------------------------
# Detector/Resolver AC-RAG loop
# ----------------------------
def detect_and_resolve(
    paper_id: int,
    full_text: str,
    index: faiss.IndexFlatL2,
    chunks_mapping: dict,
    max_rounds: int = 3,
    top_k: int = 5
):
    """
    Perform research gap detection using an adversarial Detector/Resolver loop.
    Args:
        paper_id: ID of paper in DB
        full_text: full extracted text of paper
        index: FAISS index built over paper chunks
        chunks_mapping: dict {faiss_id: chunk_text}
        max_rounds: max number of probing rounds
        top_k: number of chunks retrieved per query
    Returns:
        memory: list of (sub_question, gap_statement)
    """
    memory = []

    for rnd in range(max_rounds):
        # ----------------------------
        # Step 1: Detector generates a probing sub-question
        # ----------------------------
        det_prompt = f"""
        You are a Research Gap Detector.

        Given the paper content and memory of discovered gaps so far:
        Memory: {memory}

        Propose ONE new probing question that might reveal a missing limitation,
        unresolved issue, or overlooked area in this paper.
        """
        subq = call_local_llm(det_prompt, max_tokens=200)

        # ----------------------------
        # Step 2: Resolver retrieves evidence
        # ----------------------------
        q_emb = embed_query(subq)  # convert subq to vector
        D, I = index.search(np.array([q_emb]).astype("float32"), top_k)
        retrieved = [chunks_mapping[i] for i in I[0]]

        # ----------------------------
        # Step 3: Resolver synthesizes a gap
        # ----------------------------
        res_prompt = f"""
        You are a Research Gap Resolver.

        The probing question is: {subq}
        The retrieved evidence from the paper is:
        {retrieved}

        Based on this, draft ONE precise limitation, gap, or unresolved question
        in the paper. Keep it 2–3 sentences max.
        """
        gap = call_local_llm(res_prompt, max_tokens=300)

        memory.append((subq, gap))

        # ----------------------------
        # Step 4: Detector decides whether to stop
        # ----------------------------
        stop_prompt = f"""
        Gaps discovered so far: {memory}

        Is this sufficient (at least 2–3 unique, well-formed gaps)?
        Answer with 'yes' or 'no' and a short reason.
        """
        suff = call_local_llm(stop_prompt, max_tokens=100)

        if suff.lower().strip().startswith("yes"):
            break

    return memory


# ----------------------------
# CLI test
# ----------------------------
if __name__ == "__main__":
    from app.embed_utils import model

    # Example text (small sample)
    text = """
    We propose a new deep learning method for image classification.
    It achieves high accuracy on CIFAR-10.
    However, it struggles with robustness against adversarial examples.
    Future work could explore transfer learning.
    """

    # Build FAISS index
    from app.embed_utils import chunk_text
    chunks = chunk_text(text, chunk_size=30, overlap=5)
    embs = model.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)
    chunks_mapping = {i: c for i, c in enumerate(chunks)}

    # Run detector/resolver loop
    results = detect_and_resolve(1, text, index, chunks_mapping)
    print("\n=== Research Gaps Detected ===")
    for q, g in results:
        print(f"Q: {q}\nGap: {g}\n")
