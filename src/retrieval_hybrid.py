import json
import os
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

# ─────────────────────────────────────────────
# 1. Load Corpus
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "output.json")

with open(data_path, "r", encoding="utf-8") as f:
    raw_articles = json.load(f)

articles = [a for a in raw_articles if a.get("title") and a.get("abstract")] # Keep only articles that have both title and abstract
print(f"Corpus: {len(articles)} articles loaded.\n")

# ─────────────────────────────────────────────
# 2. Ground Truth
# ─────────────────────────────────────────────
# Relevant PMIDs per query — identified by inspecting corpus titles/abstracts.
# A PMID is marked relevant if its title/abstract directly addresses the query topic.
GROUND_TRUTH = {
    "What are the latest guidelines for managing type 2 diabetes?": [
        "41454671",  # Neutrophil/albumin ratio ... type 2 diabetes mellitus risk
        "41848224",  # Breaking the cycle: patient activation ... diabetes self-care
        "41817101",  # Understanding barriers ... adherence to lifestyle changes in prediabetes
        "41845564",  # Gut microbiome in type 2 diabetes
        "41665666",  # Capillary blood screening for type 1 diabetes (diabetes family)
    ],
    "Çocuklarda akut otitis media tedavisi nasıl yapılır?": [
        "41342403",  # OCT for Pediatric Middle Ear
        "41964158",  # AI-Assisted Otoscopic Image Screening for Pediatric Otitis Media
        "41090395",  # Diagnostic accuracy of otitis media
    ],
    "Iron supplementation dosing for anemia during pregnancy": [
        "41672571",  # Shengxuening tablets for prevention of anemia in pregnancy
        "41922931",  # Serum trace elements and iron deficiency anemia
        "41912474",  # IV ferric carboxymaltose treatment
        "41916414",  # Screening for iron deficiency in young women
    ],
    "Çölyak hastalığı tanı kriterleri nelerdir?": [
        "41782401",  # Immune-related enteropathy
        "41714845",  # Shared mechanisms for IgA nephropathy and celiac disease
        "41665666",  # Capillary blood screening (autoimmune overlap)
    ],
    "Antibiotic resistance patterns in community acquired pneumonia": [
        "41719027",  # External validation of risk scores for CAP diagnosis
        "41684123",  # Clinical features of community-acquired pneumonia
        "41810829",  # Patient risk factors, disease severity in CAP
    ],
}

# ─────────────────────────────────────────────
# 3. BM25 Setup
# ─────────────────────────────────────────────
def tokenize(text: str) -> list[str]:  # Edit text for BM25
    return text.lower().split()

bm25 = BM25Okapi(
    [tokenize((a["title"] or "") + " " + (a["abstract"] or "")) for a in articles], # combine title and abstract than tokenized
    k1=1.2,
    b=0.3,
)
# k1 controls term-frequency saturation; b controls length normalization

def bm25_search(query: str, top_k: int = 5) -> list[dict]: 
    scores = bm25.get_scores(tokenize(query))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {
            "pmid": articles[i]["pmid"],
            "title": articles[i]["title"],
            "score": round(float(scores[i]), 4),
            "rank": r + 1,
        }
        for r, i in enumerate(top_idx)
    ]

# ─────────────────────────────────────────────
# 4. Semantic (E5) Setup
# ─────────────────────────────────────────────
class SemanticSearcher:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        corpus_texts = [f"passage: {a['title']} {a['abstract']}" for a in articles]   # Do not need to define tokenized function
        print("Encoding corpus...")
        self.corpus_embeddings = self.model.encode(corpus_texts, convert_to_tensor=True)  # text => vector
        print("Encoding complete.\n")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.model.encode(f"query: {query}", convert_to_tensor=True)  
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]  # Calculate cosine simularity
        top_results = torch.topk(cos_scores, k=min(top_k, len(articles)))
        return [
            {
                "pmid": articles[idx.item()]["pmid"],
                "title": articles[idx.item()]["title"],
                "score": round(score.item(), 4),
                "rank": r + 1,
            }
            for r, (score, idx) in enumerate(zip(top_results[0], top_results[1]))
        ]

# ─────────────────────────────────────────────
# 5. RRF + Hybrid Search
# ─────────────────────────────────────────────
def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60, top_k: int = 5) -> list[dict]:  # RRF is used for combining the results of multiple IR methods with ranking system
    rrf_scores: dict[str, float] = {}
    pmid_to_title: dict[str, str] = {}
    for ranked_list in ranked_lists:   # ranked_list = bm25 + e5 
        for item in ranked_list:
            pmid = item["pmid"]
            rrf_scores[pmid] = rrf_scores.get(pmid, 0.0) + 1.0 / (k + item["rank"])  # The formula for RRF, rrf_scores keeps the total RRF score for each document
            pmid_to_title[pmid] = item["title"]
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {"rank": i + 1, "pmid": pmid, "title": pmid_to_title[pmid], "rrf_score": round(score, 6)}
        for i, (pmid, score) in enumerate(sorted_docs[:top_k])
    ]

def hybrid_search(query: str, searcher: SemanticSearcher, top_k: int = 5, candidate_pool: int = 10) -> list[dict]: # how many candidates to pull from each method
    return reciprocal_rank_fusion(
        ranked_lists=[
            bm25_search(query, top_k=candidate_pool),
            searcher.search(query, top_k=candidate_pool),
        ],
        top_k=top_k,
    )

# ─────────────────────────────────────────────
# 6. Metric Functions
# ─────────────────────────────────────────────
def precision_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    """
    P@k = (# relevant docs in top-k) / k
    """
    top_k = retrieved[:k]
    hits = sum(1 for pmid in top_k if pmid in relevant)
    return round(hits / k, 4)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """
    RR = 1 / rank_of_first_relevant_doc
    Returns 0.0 if no relevant doc found in results.
    """
    for i, pmid in enumerate(retrieved, start=1):
        if pmid in relevant:
            return round(1.0 / i, 4)
    return 0.0


def evaluate(method_name: str, results_by_query: dict[str, list[str]]) -> dict:
    """
    Compute P@5 and MRR across all queries.

    results_by_query : { query_string: [pmid_rank1, pmid_rank2, ...] }
    """
    p5_scores = []
    rr_scores  = []

    print(f"\n{'='*65}")
    print(f"  Method: {method_name}")
    print(f"{'='*65}")
    print(f"  {'Query':<52} {'P@5':>5}  {'RR':>6}")
    print(f"  {'─'*52} {'─'*5}  {'─'*6}")

    for query, retrieved_pmids in results_by_query.items():
        relevant = GROUND_TRUTH.get(query, [])
        p5  = precision_at_k(retrieved_pmids, relevant, k=5)
        rr  = reciprocal_rank(retrieved_pmids, relevant)
        p5_scores.append(p5)
        rr_scores.append(rr)
        short_q = (query[:50] + "..") if len(query) > 50 else query
        print(f"  {short_q:<52} {p5:>5.2f}  {rr:>6.4f}")

    mean_p5 = round(sum(p5_scores) / len(p5_scores), 4)
    mrr     = round(sum(rr_scores)  / len(rr_scores),  4)

    print(f"  {'─'*52} {'─'*5}  {'─'*6}")
    print(f"  {'MEAN':<52} {mean_p5:>5.2f}  {mrr:>6.4f}")
    print(f"\n  ➜  Mean P@5 = {mean_p5}   |   MRR = {mrr}")

    return {"method": method_name, "mean_p5": mean_p5, "mrr": mrr}

# ─────────────────────────────────────────────
# 7. Run Evaluation
# ─────────────────────────────────────────────
if __name__ == "__main__":
    searcher = SemanticSearcher()

    queries = list(GROUND_TRUTH.keys())

    # Collect results for each method
    bm25_results    = {q: [r["pmid"] for r in bm25_search(q, top_k=5)]           for q in queries}
    e5_results      = {q: [r["pmid"] for r in searcher.search(q, top_k=5)]        for q in queries}
    hybrid_results  = {q: [r["pmid"] for r in hybrid_search(q, searcher, top_k=5)] for q in queries}

    # Evaluate
    scores = [
        evaluate("BM25",          bm25_results),
        evaluate("Semantic (E5)", e5_results),
        evaluate("Hybrid RRF",    hybrid_results),
    ]

    # ── Summary Table ──────────────────────────────────────────────
    print(f"\n\n{'='*45}")
    print("  FINAL COMPARISON SUMMARY")
    print(f"{'='*45}")
    print(f"  {'Method':<18} {'Mean P@5':>8}  {'MRR':>8}")
    print(f"  {'─'*18} {'─'*8}  {'─'*8}")
    best = max(scores, key=lambda x: x["mean_p5"] + x["mrr"])
    for s in scores:
        marker = " ✓" if s["method"] == best["method"] else ""
        print(f"  {s['method']:<18} {s['mean_p5']:>8.4f}  {s['mrr']:>8.4f}{marker}")
    print(f"\n  Best method: {best['method']}  (P@5={best['mean_p5']}, MRR={best['mrr']})")
    print(f"{'='*45}\n")

