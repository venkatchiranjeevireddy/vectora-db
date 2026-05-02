# VectoRA — Custom HNSW Vector Database & RAG Engine

> Built from scratch. No FAISS. No Pinecone. 96.2% Recall@10 at 50K vectors.

## What Is This?

**VectoRA** is a production-grade vector search engine and RAG system built entirely from scratch.

Every component is implemented manually:
- The **HNSW graph** — layers, connections, entry points
- The **neighbor selection heuristic** — diversified, not naive
- The **search algorithm** — greedy descent + beam search
- The **RAG pipeline** — PDF → chunks → embeddings → AI answers

No FAISS. No Pinecone. No vector DB libraries. Just Python and NumPy.

---

## Why I Built This

Most vector search tools are black boxes.

I wanted to understand exactly **how HNSW works internally** — the graph structure, neighbor selection, recall tradeoffs. So I built it from scratch, benchmarked it against brute force, and layered a full RAG pipeline on top.

This is not a tutorial project. Every line is written, debugged, and benchmarked by hand.

---

## Quick Demo

### 1. Install
```bash
pip install numpy fastapi uvicorn sentence-transformers pypdf groq pytest matplotlib
```

### 2. Get free Groq API key
```
https://console.groq.com → API Keys → Create → Copy
```

### 3. Set key and start
```bash
export GROQ_API_KEY=your_key_here
uvicorn vector_db.api.main:app --reload
```

### 4. Open browser
```
http://127.0.0.1:8000
```

### 5. Upload any PDF and ask questions
```
Click Upload PDF → select any document
Type your question → click Search

🤖 AI Answer appears at top
📚 Source chunks appear below with similarity scores
```

### Example
```
Question: "What happens if a student rejects an offer?"

🤖 AI Answer:
If a student rejects an on-campus offer, they will be
debarred from participating in any further placement drives.
A student can only reject if they secure another offer
with at least 1.7x the CTC of the previous one.

📚 Retrieved Sources:
#1 • 65.7% — Students cannot reject an on-campus offer...
#2 • 63.7% — Strict action will be taken against students...
#3 • 54.0% — In case of rejection, students are required...

Vector retrieval latency: ~13ms
```

**Works on any PDF — chunks indexed dynamically per document:**
- 📄 Resumes → ~14 chunks
- 📋 Policy documents → ~68 chunks
- 📚 Research papers → ~100+ chunks

---

## Benchmark Results

Run it yourself:
```bash
python -m vector_db.benchmark.run
```

### Dataset: 50,000 vectors | dim=128 | queries=50

| ef_search | Recall@10 | HNSW Latency | Brute Force | Throughput |
|-----------|-----------|--------------|-------------|------------|
| 50        | 55.2%     | 36.8ms       | 2.9ms       | 25 QPS     |
| 100       | 71.0%     | 59.8ms       | 1.2ms       | 16 QPS     |
| 200       | 87.4%     | 97.8ms       | 1.3ms       | 10 QPS     |
| 400       | 96.2%     | 180ms        | 2.2ms       | 5 QPS      |

> Peak recall: **96.2% at ef_search=400**

### Why brute force latency is low:
NumPy brute force uses BLAS (C++ optimized matrix ops).
VectoRA HNSW is pure Python — prioritizes correctness
and readability over raw speed.

At **1M+ vectors**, brute force becomes O(N) and impossible.
HNSW stays O(log N) — this is where it wins in production.

Generate plots:
```bash
python -m vector_db.benchmark.plot
# saves to vector_db/benchmark/plots/
```

---

## Architecture

```
vector_db/
│
├── core/
│   ├── similarity.py         # Vectorized cosine similarity, top-k selection
│   ├── store.py              # In-memory VectorStore (id → vector mapping)
│   └── brute_force.py        # Brute force baseline (ground truth for recall)
│
├── hnsw/
│   ├── graph.py              # HNSW layer structure + level assignment
│   ├── insert.py             # Insertion logic + search_layer primitive
│   ├── neighbor_selection.py # Diversified neighbor heuristic (key to recall)
│   └── search.py             # hnsw_search() — top-down greedy query
│
├── benchmark/
│   ├── metrics.py            # Recall@k, latency, throughput utilities
│   ├── run.py                # Benchmark runner (HNSW vs brute force)
│   └── plot.py               # Matplotlib recall/latency/throughput plots
│
├── rag/
│   └── pipeline.py           # PDF extraction, semantic chunking, retrieval
│
├── api/
│   └── main.py               # FastAPI endpoints
│
├── frontend/
│   └── index.html            # Dark theme search UI
│
├── utils/
│   └── persistence.py        # Save/load index to disk
│
└── tests/
    ├── test_core.py           # 9 tests — similarity, store, brute force
    └── test_hnsw.py           # 3 tests — insert, recall, entry point
```

---

## Data Flow

```
Raw PDF
   ↓
extract_pdf_text()        # page-by-page extraction
   ↓
clean_text()              # normalize, remove noise
   ↓
semantic_chunk()          # split by meaning, not word count
   ↓
embed_batch()             # sentence-transformers all-MiniLM-L6-v2
   ↓
HNSW insert()             # build hierarchical graph
   ↓
User Query
   ↓
expand_query()            # semantic query expansion
   ↓
embed_text()              # embed query vector
   ↓
hnsw_search()             # greedy descent → beam search at layer 0
   ↓
Top-K chunks
   ↓
Groq LLM (LLaMA3)         # generate answer from context
   ↓
AI Answer + Sources
```

---

## Key Technical Decisions

### 1. Diversified Neighbor Selection
The single most important decision for recall quality.

```python
# NAIVE — causes dense clustering, recall ~40%
neighbors = sorted(candidates)[:M]

# DIVERSIFIED — spans multiple directions, recall 96%+
for candidate in sorted_candidates:
    if candidate is closer to query
       than to any existing neighbor:
        add to result  # adds new spatial information
    else:
        skip           # redundant, already covered
```

This single fix took recall from **40% → 96.2%**.

---

### 2. Bidirectional Connections
When inserting node A connected to node B,
VectoRA also adds A into B's neighbor list.

This keeps the graph navigable in both directions.
Without this — recall drops to ~76%.

---

### 3. M0 = 2×M at Layer 0
Layer 0 contains all nodes (densest layer).
Allowing 2×M connections here improves local
connectivity and recall without large memory cost.

---

### 4. Semantic Chunking
PDF text split by meaning — not fixed word count.

```
Fixed:    "...end of idea. Start of ne" | "xt idea..."
Semantic: "Complete idea one."  |  "Complete idea two."
```

Better chunks → better embeddings → better retrieval.

---

## RAG Pipeline

```
PDF Upload
   ↓
Semantic chunks indexed dynamically
   ↓
User question → HNSW retrieves top-3 chunks in <15ms
   ↓
Groq LLaMA3 reads chunks → generates grounded answer
   ↓
UI shows: AI Answer + Source chunks with scores
```

The vector DB does the retrieval.
The LLM does the reasoning.
Clean separation of concerns.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Open frontend UI |
| POST | `/upload-pdf` | Upload PDF, build HNSW index |
| POST | `/ask` | Ask question, get AI answer + sources |
| POST | `/insert` | Insert a single vector |
| POST | `/search` | Search top-k similar vectors |
| GET | `/stats` | Index stats |

### curl examples

**Upload PDF:**
```bash
curl -X POST http://127.0.0.1:8000/upload-pdf \
  -F "file=@your_document.pdf"
```

**Ask a question:**
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main points?", "k": 5}'
```

**Insert vector:**
```bash
curl -X POST http://127.0.0.1:8000/insert \
  -H "Content-Type: application/json" \
  -d '{"id": "v1", "vector": [0.1, 0.2, 0.3]}'
```

**Stats:**
```bash
curl http://127.0.0.1:8000/stats
```

---

## Test Results

```bash
python -m pytest vector_db/tests/ -v
```

```
platform linux -- Python 3.12.3, pytest-9.0.3

test_core.py::test_cosine_similarity_same_vector_returns_one   PASSED
test_core.py::test_cosine_similarity_orthogonal_returns_zero   PASSED
test_core.py::test_cosine_similarity_zero_query_returns_zero   PASSED
test_core.py::test_top_k_indices_returns_k_and_valid_indices   PASSED
test_core.py::test_top_k_indices_top_result_is_most_similar    PASSED
test_core.py::test_vector_store_add_and_get                    PASSED
test_core.py::test_vector_store_add_batch_and_len              PASSED
test_core.py::test_vector_store_get_wrong_id_raises_key_error  PASSED
test_core.py::test_brute_force_search_returns_k_and_top_score  PASSED
test_hnsw.py::test_insert_and_search                           PASSED
test_hnsw.py::test_recall_vs_bruteforce                        PASSED
test_hnsw.py::test_entry_point_is_set                          PASSED

12 passed in 1.20s
```

---

## HNSW Parameters

| Parameter | Meaning | Increase Effect |
|-----------|---------|-----------------|
| `M` | Max connections per node | ↑ recall, ↑ memory |
| `ef_construction` | Candidate pool during build | ↑ recall, ↑ build time |
| `ef_search` | Candidate pool during query | ↑ recall, ↑ latency |

Current: `M=32`, `ef_construction=400`

---

## Stack

| Component | Technology |
|-----------|-----------|
| Core engine | Python, NumPy |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| LLM | Groq API (LLaMA3-8b-8192) — free |
| API | FastAPI + Uvicorn |
| PDF parsing | pypdf |
| Benchmarking | Custom metrics + Matplotlib |
| Tests | pytest |
| Vector DB | **100% built from scratch** |

---

## What I Learned

- Why **diversified neighbor selection** is the core of HNSW quality
- How **bidirectional connections** determine graph navigability
- The exact **efSearch vs recall vs latency** tradeoff
- How **production RAG systems** are structured end to end
- Why **semantic chunking** beats fixed-size chunking
- How to **benchmark ANN indexes** properly using Recall@k

---

## Interview Questions This Project Answers

**Why is HNSW faster than brute force at scale?**
> Brute force is O(N) — checks every vector. HNSW navigates a hierarchical graph in O(log N). At 1M+ vectors brute force becomes impossible, HNSW stays fast.

**Why not KD-Trees?**
> KD-Trees degrade exponentially in high dimensions (curse of dimensionality). HNSW maintains performance at 128-1536 dimensions.

**What happens when M is too small?**
> Graph becomes poorly connected. Nodes cannot navigate to good neighbors. Recall drops significantly.

**How does efSearch affect recall?**
> Higher efSearch = larger candidate pool at layer 0 = more accurate results. VectoRA goes from 55% recall at efSearch=50 to 96.2% at efSearch=400.

**What is the recall vs latency tradeoff?**
> efSearch=50 → 55% recall, 36ms. efSearch=400 → 96.2% recall, 180ms. Choose based on your accuracy requirements.

**How does RAG work end to end?**
> PDF → semantic chunks → embed → store in vector DB → query → retrieve relevant chunks → feed to LLM → grounded answer.

**Why is your HNSW slower than NumPy brute force?**
> NumPy brute force uses BLAS (C++ optimized). VectoRA HNSW is pure Python prioritizing correctness. Production systems like FAISS solve this with C++ HNSW. At 1M+ vectors HNSW wins regardless.



*Built with zero vector DB libraries. Every algorithm implemented from scratch.*
