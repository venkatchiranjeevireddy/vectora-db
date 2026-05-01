import time

import numpy as np  # type: ignore

from vector_db.benchmark.metrics import compute_recall
from vector_db.core.brute_force import BruteForceSearch
from vector_db.hnsw.graph import HNSWGraph
from vector_db.hnsw.insert import insert
from vector_db.hnsw.search import hnsw_search


def build_index(
    n_vectors: int = 1000,
    dim: int = 128,
    M: int = 32,
    ef_construction: int = 400,
):
    """
    Build an HNSW index from randomly generated vectors.

    Returns:
        (graph, vectors, ids)
        - graph: HNSWGraph instance
        - vectors: dict {id: 1D numpy array}
        - ids: list of IDs in insertion order
    """

    n_vectors = int(n_vectors)
    dim = int(dim)

    graph = HNSWGraph(M=M, ef_construction=ef_construction)
    vectors = {}
    ids = [f"v{i}" for i in range(n_vectors)]

    rng = np.random.default_rng(0)
    for node_id in ids:
        vec = rng.normal(size=(dim,)).astype(float)
        insert(id=node_id, vector=vec, graph=graph, vectors=vectors)

    return graph, vectors, ids


def run_benchmark(
    n_vectors: int = 1000,
    dim: int = 128,
    k: int = 10,
    ef_search_values=(50, 100, 200, 400),
):
    """
    Run an end-to-end benchmark for HNSW vs brute force.

    Steps:
    - build index once
    - generate 50 random query vectors
    - precompute brute-force top-k results for all queries (and brute latency)
    - for each ef_search:
        - run hnsw_search for all queries
        - compute mean Recall@k vs brute results
        - compute mean latency and throughput

    Returns:
        List of dicts with keys:
        ef_search, recall, hnsw_latency_ms, brute_latency_ms, throughput_qps
    """

    k = int(k)
    ef_search_values = list(ef_search_values)

    # Build index once.
    graph, vectors, ids = build_index(
        n_vectors=n_vectors, dim=dim, M=32, ef_construction=400
    )

    # Create 50 random query vectors.
    n_queries = 100
    rng = np.random.default_rng(1)
    query_matrix = rng.normal(size=(n_queries, dim)).astype(float)
    queries = [query_matrix[i] for i in range(n_queries)]

    # Build a VectorStore for brute force scoring.
    # (We precompute brute results once for recall comparisons,
    # but we re-time brute-force inside each ef_search loop per the spec.)
    from vector_db.core.store import VectorStore

    store = VectorStore(dim=dim)
    store.add_batch(ids, np.stack([vectors[node_id] for node_id in ids], axis=0))
    brute_store = BruteForceSearch(store=store)

    # Precompute brute-force top-k results (no timing here).
    brute_results_per_query = []
    for i in range(n_queries):
        q = queries[i]
        brute_results_per_query.append(brute_store.search(query=q, k=k))

    results = []

    # Run HNSW for each ef_search value and compute metrics.
    for ef_search in ef_search_values:
        # Collect HNSW results and time it across all queries.
        hnsw_results_per_query = []
        hnsw_latencies_ms = np.empty((n_queries,), dtype=float)
        hnsw_start = time.perf_counter()

        for i in range(n_queries):
            q = queries[i]
            t0 = time.perf_counter()
            res = hnsw_search(
                query=q, k=k, graph=graph, vectors=vectors, ef_search=ef_search
            )
            t1 = time.perf_counter()
            hnsw_results_per_query.append(res)
            hnsw_latencies_ms[i] = (t1 - t0) * 1000.0

        # Also run and time brute force across all queries for this ef_search.
        # Recall comparisons use precomputed brute_results_per_query, but
        # timing must reflect running the brute search.
        brute_latencies_ms = np.empty((n_queries,), dtype=float)
        brute_start = time.perf_counter()
        for i in range(n_queries):
            q = queries[i]
            t0 = time.perf_counter()
            _ = brute_store.search(query=q, k=k)
            t1 = time.perf_counter()
            brute_latencies_ms[i] = (t1 - t0) * 1000.0
        _ = brute_start  # keep for clarity; total not required

        hnsw_total_s = time.perf_counter() - hnsw_start
        throughput_qps = float(n_queries) / float(hnsw_total_s) if hnsw_total_s > 0 else 0.0
        mean_hnsw_latency_ms = float(np.mean(hnsw_latencies_ms))

        # Compute mean recall@k across all queries.
        recalls = np.empty((n_queries,), dtype=float)
        for i in range(n_queries):
            hnsw_res = hnsw_results_per_query[i]
            brute_res = brute_results_per_query[i]
            recalls[i] = compute_recall(
                hnsw_results=hnsw_res, brute_results=brute_res, k=k
            )

        mean_recall = float(np.mean(recalls))

        results.append(
            {
                "ef_search": int(ef_search),
                "recall": mean_recall,
                "hnsw_latency_ms": mean_hnsw_latency_ms,
                "brute_latency_ms": float(np.mean(brute_latencies_ms)),
                "throughput_qps": throughput_qps,
            }
        )

    return results


if __name__ == "__main__":
    results = run_benchmark()

    # Print a clean table.
    header = "ef_search | recall | hnsw_ms | brute_ms | qps"
    print(header)
    print("-" * len(header))

    for row in results:
        ef_search = row["ef_search"]
        recall = row["recall"]
        hnsw_ms = row["hnsw_latency_ms"]
        brute_ms = row["brute_latency_ms"]
        qps = row["throughput_qps"]
        print(
            f"{ef_search:8d} | {recall:6.3f} | {hnsw_ms:8.3f} | {brute_ms:8.3f} | {qps:8.2f}"
        )

    print("Benchmark complete")

