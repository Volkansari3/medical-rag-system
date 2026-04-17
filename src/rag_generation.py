"""
Part 3 — RAG Generation (Groq Version)
=====================================
Retrieval : Hybrid RRF (BM25 + multilingual-e5-small)
LLM       : Groq — llama-3.3-70b-versatile

Why Groq + LLaMA-3.3-70B?
  - Free tier with generous rate limits
  - ~300 tokens/sec — fastest open-weight inference available
  - 70B size gives strong medical reasoning and reliable citation following
  - Follows system prompt instructions consistently vs smaller models
"""

import os
import time
from groq import Groq
from retrieval_hybrid import hybrid_search, SemanticSearcher, articles

from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

MODEL = "llama-3.3-70b-versatile"   # 70B >> 8B for medical QA and citation accuracy
TOP_K_RETRIEVE  = 5
RRF_K           = 60
CANDIDATE_POOL  = 10

SYSTEM_PROMPT = """You are a medical literature assistant for doctors.

STRICT RULES:
- Answer ONLY from the provided context. Never use outside knowledge.
- Cite every factual claim with (PMID: XXXXX).
- If the context does not contain enough information, say so clearly.
- Be concise. Do not repeat the same information twice.
- CRITICAL: Always respond in the SAME language as the question. If the question is in English, answer in English. If the question is in Turkish, answer in Turkish. Never switch languages."""

# ─────────────────────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────────────────────
def build_context(retrieved: list[dict], corpus_articles: list[dict]) -> str:
    pmid_map = {a["pmid"]: a for a in corpus_articles}
    blocks = []

    for item in retrieved:
        article = pmid_map.get(item["pmid"], {})
        doi_str = f"\n  DOI: {article.get('doi')}" if article.get("doi") else ""

        block = (
            f"[{item['rank']}] PMID: {item['pmid']}\n"
            f"  Title: {article.get('title', 'N/A')}\n"
            f"  Journal: {article.get('journal', 'N/A')} ({article.get('year', 'N/A')}){doi_str}\n"
            f"  Abstract: {article.get('abstract', 'N/A')}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)

# ─────────────────────────────────────────────
# GROQ CALL
# ─────────────────────────────────────────────
def call_groq(query: str, context: str) -> str:
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY not set.\n"
            "Windows : set GROQ_API_KEY=your_key\n"
            "Linux   : export GROQ_API_KEY=your_key"
        )

    user_message = f"IMPORTANT: Respond in the same language as the question below.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{query}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    return response.choices[0].message.content

# ─────────────────────────────────────────────
# RAG PIPELINE
# ─────────────────────────────────────────────
def rag_query(
    query: str,
    searcher: SemanticSearcher,
    top_k: int = TOP_K_RETRIEVE,
    candidate_pool: int = CANDIDATE_POOL,
) -> dict:

    # 1. Retrieval — hybrid RRF (best method per evaluation.py results)
    retrieved = hybrid_search(
        query,
        searcher,
        top_k=top_k,
        candidate_pool=candidate_pool,
    )

    # 2. Build context from full abstracts
    context = build_context(retrieved, articles)

    # 3. Generate cited answer
    answer = call_groq(query, context)

    return {
        "query":     query,
        "retrieved": retrieved,
        "context":   context,
        "answer":    answer,
    }

# ─────────────────────────────────────────────
# PRINT
# ─────────────────────────────────────────────
def print_rag_result(result: dict) -> None:
    print("\n" + "=" * 70)
    print("QUERY:", result["query"])
    print("=" * 70)

    print("\nRETRIEVED ARTICLES (Hybrid RRF):")
    print("-" * 70)
    for r in result["retrieved"]:
        print(f"  [{r['rank']}] PMID {r['pmid']}  (rrf_score: {r['rrf_score']})")
        print(f"       {r['title'][:85]}...")

    print("\nGENERATED ANSWER:")
    print("-" * 70)
    print(result["answer"])
    print()

# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────
DEMO_QUERIES = [
    "What is the clinical significance of the neutrophil/albumin ratio (NAR) in T2DM risk?",
    "Diyabet yönetiminde hasta aktivasyonunun (patient activation) rolü nedir?",
    "What are the diagnostic and treatment approaches for community-acquired pneumonia?"
]

def main():
    print("=" * 70)
    print("DoctorFollow AI — RAG Demo (Groq / LLaMA-3.3-70B)")
    print("=" * 70)
    print(f"Corpus : {len(articles)} articles")
    print(f"Model  : {MODEL}")
    print(f"Top-K  : {TOP_K_RETRIEVE}  |  RRF-k: {RRF_K}\n")

    searcher = SemanticSearcher()

    for query in DEMO_QUERIES:
        try:
            result = rag_query(query, searcher)
            print_rag_result(result)
            time.sleep(2)   # stay within Groq free-tier rate limits
        except Exception as e:
            print(f"\n[ERROR] {e}")

    print("\n✅ Done.")

if __name__ == "__main__":
    main()