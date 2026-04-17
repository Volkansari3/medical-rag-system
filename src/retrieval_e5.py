import json
import os
import torch
from sentence_transformers import SentenceTransformer, util

# --- Load Data ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "output.json")

with open(data_path, "r", encoding="utf-8") as f:
    articles_dict = json.load(f)

articles = [a for a in articles_dict if a.get("title") and a.get("abstract")]  # List Comprehension

# --- Semantic Search Class ---
class SemanticSearcher:
    def __init__(self, model_name='intfloat/multilingual-e5-small'):  # Call the embedding model
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)                  # token => vector
        
        self.corpus_texts = [
            f"passage: {a['title']} {a['abstract']}" for a in articles  # passage: title + abstract, no tokenized process in semantic search
        ]
        
        # All articles are being converted to vector format, 
        print("Encoding corpus (creating embeddings), this may take a moment...")
        self.corpus_embeddings = self.model.encode(self.corpus_texts, convert_to_tensor=True) # convert_to_tensor=True, allows for faster computations using PyTorch
        print("Encoding completed successfully.")

    def search(self, query, top_k=5):
        query_text = f"query: {query}"
        query_embedding = self.model.encode(query_text, convert_to_tensor=True) # The query is being converted to embedding
        
        # Compute Cosine Similarity (Result is between 0 and 1)
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        
        # Retrieve the top_k results with the highest scores
        top_results = torch.topk(cos_scores, k=min(top_k, len(articles)))
        
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            results.append({
                "pmid": articles[idx.item()]["pmid"],
                "title": articles[idx.item()]["title"],
                "score": round(score.item(), 4)
            })
        return results

# --- Main Test Loop ---
if __name__ == "__main__":
    if not articles:
        print("Error: No valid articles found in the corpus. Check your pipeline output.")
    else:
        searcher = SemanticSearcher()
        
        test_queries = [
            "What are the latest guidelines for managing type 2 diabetes?",
            "Çocuklarda akut otitis media tedavisi nasıl yapılır?",
            "Iron supplementation dosing for anemia during pregnancy",
            "Çölyak hastalığı tanı kriterleri nelerdir?",
            "Antibiotic resistance patterns in community acquired pneumonia"
        ]
        
        for query in test_queries:
            print(f"\n{'='*70}")
            print(f"Executing Semantic Query: {query}")
            print(f"{'='*70}")
            
            results = searcher.search(query)
            
            if not results:
                print("No matching articles found.")
                continue 
            
            for i, r in enumerate(results, 1):
                print(f"[{i}] PMID: {r['pmid']} | Score: {r['score']}")
                print(f"    Title: {r['title'][:90]}...")