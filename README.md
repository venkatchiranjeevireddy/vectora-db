# Custom HNSW Vector Database

## Problem
- **Why brute force is slow (O(N) per query)**: A naive nearest-neighbor search scores the query against every stored vector, requiring \(N\) similarity computations per query. As \(N\) grows, latency increases linearly.
- **What vector search is used for**: Semantic search, retrieval-augmented generation (RAG), recommendation, deduplication, clustering, anomaly detection, and any workflow that needs “find the most similar embeddings”.

## Solution
- **What HNSW is**: HNSW (Hierarchical Navigable Small World) is an approximate nearest-neighbor (ANN) index that organizes vectors into a multi-layer proximity graph. Higher layers are sparser; layer 0 is the densest.
- **Why it's faster**: Queries start from an entry point at the top layer and descend greedily through the hierarchy, exploring only a small subset of nodes rather than scoring all vectors.

## Architecture
- `vector_db/core/`
  - `similarity.py`: vectorized cosine similarity + top-k index selection helpers.
  - `store.py`: in-memory `VectorStore` mapping ids → vectors.
  - `brute_force.py`: brute-force cosine search baseline over `VectorStore`.
- `vector_db/hnsw/`
  - `graph.py`: HNSW layer/neighbor adjacency structure and level assignment.
  - `insert.py`: insertion routine + layer search primitive (`search_layer`).
  - `neighbor_selection.py`: diversified neighbor selection heuristic (critical for recall).
  - `search.py`: `hnsw_search()` query routine (top-down + layer-0 beam search).
- `vector_db/benchmark/`
  - `metrics.py`: recall/latency/throughput utilities.
  - `run.py`: benchmark runner comparing HNSW vs brute force.
  - `plot.py`: matplotlib plots saved to `vector_db/benchmark/plots/`.
- `vector_db/api/`
  - `main.py`: FastAPI app exposing `/insert`, `/search`, `/stats`.
- `vector_db/utils/`
  - `persistence.py`: pickle-based save/load of `(graph, vectors)` for disk persistence.

## How to Run

### Install

```bash
python -m pip install --upgrade pip
python -m pip install numpy pytest fastapi uvicorn matplotlib
```

### Run benchmark

```bash
python -m vector_db.benchmark.run
```

### Generate plots

```bash
python -m vector_db.benchmark.plot
```

### Start API

```bash
uvicorn vector_db.api.main:app --reload
```

### Run tests

```bash
python -m pytest vector_db/tests -v
```

## Benchmark Results

| ef_search | Recall@10 | HNSW Latency | Brute Force Latency |
|-----------|-----------|--------------|---------------------|
| 50        | 0.978     | 6.6ms        | 0.9ms               |
| 100       | 0.998     | 7.9ms        | 0.9ms               |
| 200       | 0.999     | 8.4ms        | 0.9ms               |
| 400       | 0.999     | 8.8ms        | 0.9ms               |

## Key Insights
- **What happens when efSearch increases**: Recall generally increases because the layer-0 search tracks and explores a larger candidate set; latency also increases due to more expansions/scoring.
- **What happens when M increases**: The graph becomes denser (more edges per node), which typically improves navigability and recall, but increases memory and can increase build/search cost.
- **Why diversified neighbor selection matters**: If neighbor lists are chosen as only the top-M closest, they can become redundant (many neighbors pointing into the same local cluster). Diversification keeps edges that preserve multiple “directions” in the graph, improving global navigability and recall.
- **Tradeoff between recall and latency**: Higher `ef_search` and larger `M` push recall up, but increase CPU time (more scoring/heap operations) and memory footprint.

## API Endpoints
- **POST `/insert`**: insert a vector by id into the in-memory HNSW index.
- **POST `/search`**: search the in-memory HNSW index for top-k results.
- **GET `/stats`**: basic index stats (count, entry point, entry level).

