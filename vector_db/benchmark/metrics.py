import time

import numpy as np  # type: ignore


def compute_recall(hnsw_results, brute_results, k: int) -> float:
    """
    Compute Recall@k as overlap / k.

    Args:
        hnsw_results: list of (id, score) tuples (top-k from HNSW).
        brute_results: list of (id, score) tuples (top-k from brute force).
        k: Number of results requested (Recall@k = overlap / k).

    Returns:
        Recall@k as a float (e.g., 8 common results out of 10 => 0.8).
    """

    k = int(k)
    if k <= 0:
        return 0.0

    # Extract IDs from both result sets.
    hnsw_ids = [rid for (rid, _score) in hnsw_results[:k]]
    brute_ids = [rid for (rid, _score) in brute_results[:k]]

    # Use set intersection to compute how many of the top-k IDs overlap.
    overlap = len(set(hnsw_ids).intersection(set(brute_ids)))

    # Recall@k is overlap divided by k.
    return float(overlap) / float(k)


def measure_latency(search_fn, query, k: int, runs: int = 100) -> dict:
    """
    Measure latency statistics for a search function over repeated runs.

    Args:
        search_fn: callable used to perform search; expected signature compatible with:
            search_fn(query=query, k=k) or search_fn(query, k).
        query: query vector (passed through to search_fn).
        k: number of results (passed through to search_fn).
        runs: number of repetitions.

    Returns:
        Dict containing:
        - mean_ms: mean latency in milliseconds
        - min_ms: minimum latency in milliseconds
        - max_ms: maximum latency in milliseconds
        - p99_ms: 99th percentile latency in milliseconds
    """

    runs = int(runs)
    if runs <= 0:
        return {"mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "p99_ms": 0.0}

    latencies_ms = np.empty((runs,), dtype=float)

    # Run search repeatedly to capture timing distribution.
    for i in range(runs):
        start = time.perf_counter()

        # Attempt keyword-based call first; fall back to positional call.
        try:
            _ = search_fn(query=query, k=k)
        except TypeError:
            _ = search_fn(query, k)

        end = time.perf_counter()

        # Convert seconds to milliseconds.
        latencies_ms[i] = (end - start) * 1000.0

    # Compute basic statistics and p99 percentile from the measured latencies.
    mean_ms = float(np.mean(latencies_ms))
    min_ms = float(np.min(latencies_ms))
    max_ms = float(np.max(latencies_ms))

    # p99 percentile: index-based selection after sorting.
    sorted_ms = np.sort(latencies_ms)
    idx = int(np.ceil(0.99 * (runs - 1)))
    p99_ms = float(sorted_ms[idx])

    return {
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "p99_ms": p99_ms,
    }


def measure_throughput(search_fn, queries, k: int) -> float:
    """
    Measure throughput as queries per second for a given search function.

    Args:
        search_fn: callable used to perform search; expected signature compatible with:
            search_fn(query=query, k=k) or search_fn(query, k).
        queries: list of query vectors.
        k: number of results to request per query.

    Returns:
        Throughput in queries per second (float).
    """

    k = int(k)
    n_queries = len(queries)
    if n_queries == 0:
        return 0.0

    start = time.perf_counter()

    for q in queries:
        # Attempt keyword-based call first; fall back to positional call.
        try:
            _ = search_fn(query=q, k=k)
        except TypeError:
            _ = search_fn(q, k)

    end = time.perf_counter()

    elapsed_s = end - start
    if elapsed_s <= 0:
        return 0.0

    return float(n_queries) / float(elapsed_s)


if __name__ == "__main__":
    # Create dummy top-k result lists with overlap=8 out of k=10.
    k = 10

    hnsw_results = [(f"id{i}", float(1.0 - i * 0.01)) for i in range(10)]
    brute_results = [(f"id{i}", float(1.0 - i * 0.02)) for i in range(8)] + [
        (f"brute_extra_{i}", -1.0 - i) for i in range(2)
    ]

    recall = compute_recall(hnsw_results=hnsw_results, brute_results=brute_results, k=k)
    print("Recall:", recall)

    print("Metrics module working")

