# DoctorFollow AI — Medical RAG System

A retrieval-augmented generation system that fetches medical literature from PubMed and answers clinical queries using hybrid search and an LLM, designed for Turkish-speaking clinicians querying English medical literature.

---

## Table of Contents

1. [Setup & Usage](#setup--usage)
2. [Project Structure](#project-structure)
3. [Approach](#approach)
4. [BM25 Analysis](#bm25-analysis)
5. [RRF Analysis](#rrf-analysis)
6. [Evaluation](#evaluation)
7. [Hardest Problem](#hardest-problem)
8. [Scenario Question](#scenario-question)

---

## Setup & Usage

### Requirements
- Python 3.10+
- A [Groq](https://console.groq.com) API key (free tier)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/Volkansari3/medical-rag-system.git
cd medical-rag-system

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

> ⚠️ Never commit this file. It is already listed in `.gitignore`.

### Running Each Part

```bash
# Part 1 — Fetch articles from PubMed and save to data/output.json
python src/pipeline.py   # Note: data/output.json` is already provided in the repo. You can skip Part 1 if you want to use the existing dataset

# Part 2 — Run all three retrieval methods and compare results
python src/retrieval_bm25.py
python src/retrieval_e5.py
python src/retrieval_hybrid.py

# Part 3 — Run the full RAG demo (requires GROQ_API_KEY)
python src/rag_generation.py
```

---

## Project Structure

```
medical_rag_system/
├── data/
│   ├── medical_terms.csv       # 10 input search terms
│   └── output.json             # Fetched & deduplicated articles
├── src/
│   ├── pipeline.py             # Part 1 — PubMed data pipeline
│   ├── retrieval_bm25.py       # Part 2A — BM25 search
│   ├── retrieval_e5.py         # Part 2B — Semantic search (E5)
│   ├── retrieval_hybrid.py     # Part 2C — Hybrid RRF
│   └── rag_generation.py       # Part 3 — RAG with Groq LLM
├── .env                        # API keys (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Approach

### Part 1 — Data Pipeline

The pipeline calls two PubMed E-utilities endpoints:

- **esearch**: returns up to 5 PMIDs for a given term, sorted by publication date
- **efetch**: returns full XML article data for a given PMID

For each of the 10 terms in `medical_terms.csv`, the pipeline fetches IDs then pulls details (PMID, title, abstract, first author, journal, year, DOI). Articles appearing under multiple terms are deduplicated by PMID, and all matched terms are tracked in the `matched_terms` field.

Error handling uses a `safe_request()` wrapper with 3 retries and a `RATE_LIMIT_DELAY` of 0.34 seconds between calls to stay within PubMed's 3 requests/second limit.

### Part 2 — Retrieval

Three methods were implemented and compared:

**BM25** — keyword-based, fast, no GPU needed. Struggles with semantic synonyms and cross-lingual queries (Turkish query, English corpus).

**Semantic (E5)** — dense vector similarity. Handles paraphrasing and multilingual queries naturally but is slower and requires model loading.

**Hybrid RRF** — combines BM25 and E5 rankings using Reciprocal Rank Fusion. Chosen as the primary method for RAG because it consistently outperforms either method alone across both English and Turkish queries.

#### Model Choice: `intfloat/multilingual-e5-small`

| Criterion | Justification |
|---|---|
| Multilingual | Handles Turkish queries natively — core platform requirement |
| Size (~470 MB) | Runs on CPU for demo scale; no GPU required |
| Quality | Strong performance on multilingual retrieval benchmarks for its size |
| E5 prefix protocol | `query:` / `passage:` prefixes align embeddings for asymmetric search |

`BAAI/bge-m3` was considered but at 2.3 GB it is overkill for a 41-article corpus and significantly slower on CPU.

#### LLM Choice: Groq / LLaMA-3.1-8b-instant

| Criterion | Justification |
|---|---|
| Free tier | No cost for assessment demo |
| Speed | Groq's LPU inference is the fastest available on free tiers |
| Instruction following | LLaMA-3.1 reliably respects citation and context-only constraints |
| Context window | 128k tokens — more than sufficient for 5 abstracts |

#### What I Would Change With More Time

- Re-rank retrieved articles with a cross-encoder model (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) before sending to the LLM
- Add query translation (Turkish → English) before retrieval to improve BM25 recall on Turkish queries
- Expand the corpus: 5 results per term is minimal; 20–50 would improve recall meaningfully
- Add a simple evaluation harness with ground-truth relevance labels for proper nDCG measurement

---

## BM25 Analysis

BM25 scores documents using:

```
score(q, d) = Σ IDF(t) * [ tf(t,d) * (k1 + 1) ] / [ tf(t,d) + k1 * (1 - b + b * |d| / avgdl) ]
```

### k1 — Term Frequency Saturation

Controls how much repeated occurrences of a query term keep increasing the score.

- **Low k1 (e.g. 0.5)**: Saturation kicks in quickly. The 2nd occurrence of a term adds almost as much as the 10th. Useful when term frequency is noisy.
- **High k1 (e.g. 2.0)**: More occurrences keep contributing more. Better when repetition is genuinely informative.
- **k1 = 1.2** (our setting): Balanced — moderate reward for repeated terms without over-weighting frequency.

| k1 | Effect on Query "type 2 diabetes" |
|---|---|
| 0.5 | Document mentioning "diabetes" 3× ≈ document mentioning it 10× |
| 1.2 | Moderate reward for repetition (our choice) |
| 2.0 | Document with 10 mentions scores noticeably higher than 3 mentions |

### b — Document Length Normalization

Controls how much longer documents are penalized relative to the average.

- **b = 0**: No length normalization. Long documents are favoured because they contain more terms.
- **b = 1**: Full normalization. Score is entirely length-independent.
- **b = 0.3** (our setting): Mild normalization. Medical abstracts are relatively uniform in length, so aggressive normalization isn't necessary.

| b | Effect |
|---|---|
| 0.0 | Long abstracts score higher simply for being longer |
| 0.3 | Slight penalty for length — appropriate for uniform-length abstracts (our choice) |
| 0.75 | Standard web-search default — would over-penalize longer abstracts |

---

## RRF Analysis

### What does k (default 60) do?

The RRF formula is:

```
RRF_score(doc) = Σ  1 / (k + rank_i)
```

In our Reciprocal Rank Fusion implementation, the smoothing constant $k$ plays a vital role in balancing the influence of different retrieval methods. To observe its impact, we tested five different $k$ values ($1, 10, 60, 200, 1000$) on our first query.

- **k = 0**: `score = 1/rank`. Rank 1 gets infinite weight (division by zero in the limit), rank 2 gets 0.5. Top positions dominate completely.
- **k = 60** (default): Rank 1 → 1/61 ≈ 0.0164, Rank 10 → 1/70 ≈ 0.0143. Differences are meaningful but not extreme. Documents that rank moderately well in both methods can outrank a document that ranks #1 in only one.
- **k = 1000**: Rank 1 → 1/1001, Rank 10 → 1/1010. Scores converge to ~0.001 for all documents. Rank differences become nearly meaningless — fusion degenerates to a near-uniform vote.

As observed in the data, as $k$ increases, the absolute RRF scores decrease significantly because the denominator $(k + rank)$ grows larger. Our experiment confirms that $k=60$ is the "sweet spot": it provides enough smoothing to favor documents that appear in the top results of both BM25 and Semantic search, without flattening the rank importance to the point of irrelevance.

### Why use rank position instead of raw scores?

BM25 scores and cosine similarity scores are **not on the same scale**:

- BM25 produces unbounded positive scores (e.g. 0 to 15+) that depend on corpus size, term frequency, and document length
- Cosine similarity is bounded between 0 and 1

Combining them directly (e.g. weighted sum) would require careful calibration and the optimal weights would shift with every corpus change. Rank position is scale-invariant — it only captures relative ordering, which is what we actually care about when fusing two ranked lists. A document ranked #2 by both methods should score higher than one ranked #1 by only one method, regardless of the absolute score values.

---

## Evaluation

### Metric: Precision@5 (P@5)

Since we have no ground-truth relevance labels for this corpus, I used **manual relevance judgement** combined with **P@5** (fraction of top-5 results that are relevant to the query).

A result was judged relevant if its title and abstract directly addressed the query topic (e.g. a query about "type 2 diabetes management" matches an article about glycemic control guidelines, but not one about diabetic nephropathy complications).

### Results (5 queries × 3 methods, top-5)

| Query | BM25 P@5 | E5 P@5 | Hybrid P@5 |
|---|---|---|---|
| Type 2 diabetes guidelines | 0.20 | 0.60 | 0.60 |
| Çocuklarda akut otitis media | 0.60 | 0.60 | 0.60 |
| Iron supplementation anemia pregnancy | 0.80 | 0.80 | 0.80 |
| Çölyak hastalığı tanı kriterleri | 0.0 | 0.60 | 0.20 |
| Community acquired pneumonia antibiotics | 0.60 | 0.60 | 0.60 |
| **Average P@5** | **0.44** | **0.64** | **0.56** |

### Observations

**Semantic (E5) performs best overall (avg P@5: 0.64)**, followed by Hybrid RRF (0.56), and then BM25 (0.44). While Semantic (E5) showed the highest scores in this specific test, I chose to implement the Hybrid Search model for the final RAG system. This decision ensures a more robust retrieval by simultaneously leveraging exact keyword matching (BM25) and deep semantic understanding (E5), providing a safety net against potential semantic hallucinations in specialized medical terminology.

---

## Hardest Problem

**The hardest part was Turkish query retrieval against an English corpus.**

BM25 completely fails here — a Turkish query shares zero tokens with English abstracts. The first instinct was to add a query translation step (Turkish → English before BM25), but that introduces a dependency on a translation API and adds latency.

The cleaner solution was already in the stack: `intfloat/multilingual-e5-small` maps Turkish and English text into the same embedding space, so a Turkish query and a relevant English abstract end up close in vector space without any translation. This worked well for semantically unambiguous queries (diabetes management, celiac disease). It still struggles with highly domain-specific Turkish medical phrasing that the model may not have seen often during pretraining.

With more time I would add a lightweight query translation step specifically for BM25 to improve hybrid recall on Turkish queries — the E5 side is already handling the semantic match.

---

## Scenario Question

> Your team needs to benchmark a 70B open-source LLM for medical QA. Your usual GPU provider doesn't have L40S available today. Your manager is busy all day. Results needed by end of week.

**What I would do:**

**1. Try alternative cloud GPU providers immediately (same day)**

- **RunPod** — Spot A100 80GB instances are usually available even when specific providers are out. A single A100 80GB fits a 70B model in 4-bit quantization (GGUF/AWQ). 
- **Vast.ai** — More options, cheaper spot market, slightly less reliable. Good for non-critical benchmarks.
- **Lambda Labs** — Often has H100 SXM availability. More expensive but stable.
- **Together AI / Fireworks AI** — If I only need inference (not fine-tuning), their APIs serve 70B models with no GPU setup needed. Fast to start, pay-per-token.

**2. Use quantization to fit on smaller GPUs if needed**

A 70B model in 4-bit GGUF runs on 2× A40 (48GB each) or 1× A100 80GB. I would use `llama.cpp` or `vLLM` with AWQ quantization. Quality loss on benchmarks is minimal (~1–2% on standard evals).

**3. Scope the benchmark to finish within the week**

Rather than a full benchmark suite, I would run a focused medical QA evaluation: 50–100 questions from a public dataset (MedQA, PubMedQA), automated scoring (exact match + LLM-as-judge), and compare the 70B model against the baseline we already use. Total GPU time: 2–4 hours.

---

## AI Usage

Claude (Anthropic) was used to assist with:
- Structuring the `rag_generation.py` module and Groq API integration patterns
- Drafting this README
