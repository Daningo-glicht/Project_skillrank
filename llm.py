import os, json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ----------------------------
# 1. Load model + tokenizer
# ----------------------------
# Free, public model
MODEL_NAME = os.getenv("LOCAL_LLM_MODEL", "TheBloke/WizardLM-7B-1.0-GPTQ")

print(f"Loading model {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16  # reduces VRAM usage
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=800,
    temperature=0.0,
    do_sample=False
)

# ----------------------------
# 2. Prompt templates
# ----------------------------
CHUNK_PROMPT = """You are an academic summarizer.

Summarize this chunk of an academic paper.
Focus on contributions, methods, results, and limitations.
Keep it short and clear.

Text:
\"\"\"{chunk}\"\"\"

Return a concise summary (plain text).
"""

MERGE_PROMPT = """You are an academic summarizer.

You are given multiple partial summaries of different chunks of the same paper.

Merge them into one coherent JSON object with keys:
- short_summary (string, 2–3 sentences, <=150 words)
- contributions (list of strings, 3–6 items)
- methods (string)
- results (string)
- limitations (string)
- future_work (list of strings, 3 concise items)

Chunk summaries:
{chunk_summaries}

Return ONLY valid JSON, nothing else.
"""

# ----------------------------
# 3. Helper functions
# ----------------------------
def call_local_llm(prompt: str, max_tokens: int = 800) -> str:
    """Call local LLM with a prompt and return text output."""
    outputs = generator(prompt, max_new_tokens=max_tokens)
    return outputs[0]["generated_text"][len(prompt):].strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# ----------------------------
# 4. Main summarizer
# ----------------------------
def summarize_text(text: str) -> dict:
    """
    Summarize a long paper text using chunking + synthesis.
    Returns a Python dict with keys:
    short_summary, contributions, methods, results, limitations, future_work
    """
    chunks = chunk_text(text)
    partial_summaries = []

    for idx, c in enumerate(chunks):
        print(f"Summarizing chunk {idx+1}/{len(chunks)}...")
        prompt = CHUNK_PROMPT.format(chunk=c)
        summary = call_local_llm(prompt, max_tokens=400)
        partial_summaries.append(summary)

    merge_prompt = MERGE_PROMPT.format(
        chunk_summaries="\n\n".join(partial_summaries)
    )
    print("Merging summaries into final output...")
    raw_output = call_local_llm(merge_prompt, max_tokens=1000)

    # Try parsing JSON
    try:
        parsed = json.loads(raw_output)
        return parsed
    except Exception:
        print("⚠️ Warning: Failed to parse JSON. Returning raw text instead.")
        return {"short_summary": raw_output, "contributions": [], "methods": "", "results": "", "limitations": "", "future_work": []}

# ----------------------------
# 5. CLI test
# ----------------------------
if __name__ == "__main__":
    sample_text = """
    Deep learning has transformed natural language processing (NLP).
    In this paper, we present a new transformer architecture with fewer parameters.
    We evaluate it on GLUE, SQuAD, and machine translation tasks.
    Results show competitive accuracy with 30% fewer parameters.
    However, our model struggles with long-context reasoning and multilingual transfer.
    Future work includes scaling to larger datasets and adding retrieval augmentation.
    """
    result = summarize_text(sample_text)
    print("\n=== Final Summary (dict) ===\n", result)
