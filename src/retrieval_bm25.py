import json
import os
from rank_bm25 import BM25Okapi

# --- Load Data ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "output.json")    

with open(data_path, "r", encoding="utf-8") as f:
    articles = json.load(f)

# Filter the data that title or abstract is empty
articles = [a for a in articles if a.get("title") and a.get("abstract")]   # List comprehension

# --- BM25 ---
def tokenize(text):
    return text.lower().split()  # Convert text to lowercase, splits text according to spaces, BM25 model works with words 

corpus = [tokenize((a["title"] or "") + " " + (a["abstract"] or "")) for a in articles]  # For each article retrieved the title and abstract then tokenized.
bm25 = BM25Okapi(corpus, k1=1.2, b=0.3)

# k1 => How many times must a word appear in the document for the score to increase
# b => How much should we penalize long documents

# --- Search Function ---
def bm25_search(query, top_k=5):  # top_k => How many results will you return
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k] # Choose the best ones
    
    results = []
    for i in top_indices:
        results.append({
            "rank": len(results) + 1,
            "pmid": articles[i]["pmid"],
            "title": articles[i]["title"],
            "score": round(scores[i], 4),
            "matched_terms": articles[i]["matched_terms"]
        })
    return results

# --- Test ---
if __name__ == "__main__":
    test_queries = [
        "What are the latest guidelines for managing type 2 diabetes?",
        "Çocuklarda akut otitis media tedavisi nasıl yapılır?",
        "Iron supplementation dosing for anemia during pregnancy",
        "Çölyak hastalığı tanı kriterleri nelerdir?",
        "Antibiotic resistance patterns in community acquired pneumonia"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        results = bm25_search(query)
        for r in results:
            print(f"  [{r['rank']}] PMID {r['pmid']} (score: {r['score']})")
            print(f"       {r['title'][:80]}...")