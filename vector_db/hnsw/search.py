import numpy as np

from vector_db.hnsw.graph import HNSWGraph
from vector_db.hnsw.insert import insert, search_layer


def hnsw_search(query, k, graph: HNSWGraph, vectors, ef_search: int = 50):
    """
    Search an HNSW graph for the top-k most similar vectors using cosine similarity.

    Strategy:
    - Start from `graph.entry_point`.
    - From the top layer down to layer 1, perform greedy searches with ef=1
      (i.e., single best step at each layer).
    - At layer 0, search with ef=ef_search to get a wider candidate set.
    - Return the top-k candidates sorted by score descending.

    Args:
        query: 1D NumPy array with shape (D,).
        k: Number of results to return.
        graph: HNSWGraph instance containing layer connectivity.
        vectors: Dict {id: 1D NumPy vector} for scoring.
        ef_search: Search width at layer 0 (higher => more accurate).

    Returns:
        List of (id, score) tuples, sorted by score descending.
    """

    # Normalize types.
    query = np.asarray(query)
    k = int(k)
    ef_search = int(ef_search)

    # Basic validation.
    if query.ndim != 1:
        raise ValueError(f"`query` must be 1D with shape (D,), got shape {query.shape}")
    if k <= 0:
        return []
    if graph.entry_point is None:
        return []
    if query.size == 0:
        raise ValueError("`query` must contain at least one dimension")
    expected_dim = np.asarray(vectors[graph.entry_point]).shape[0]
    if query.shape[0] != expected_dim:
        raise ValueError(
            f"Query dimension mismatch: expected {expected_dim}, got {query.shape[0]}"
        )

    # Identify the top populated layer in the current graph.
    # (This is based on assigned levels recorded in graph.levels.)
    top_layer = max(graph.levels.values()) if graph.levels else 0

    # Start from the entry point.
    current_entry_point = graph.entry_point

    # Greedy descent from top layer down to layer 1.
    for layer in range(top_layer, 0, -1):
        # If the entry point doesn't belong to this layer, we cannot search from it.
        # (This can happen due to differing node levels.)
        if current_entry_point not in graph.layers[layer]:
            continue

        top = search_layer(
            query=query,
            entry_point=current_entry_point,
            ef=1,
            layer=layer,
            graph=graph,
            vectors=vectors,
        )
        if top:
            # Move to the best candidate found at this layer.
            current_entry_point = top[0][0]

    # The candidate beam must be at least as wide as the requested result
    # count; otherwise ef_search < k silently limits the number returned.
    search_width = max(k, ef_search)

    # Final search on layer 0 with an adequate candidate width.
    results = search_layer(
        query=query,
        entry_point=current_entry_point,
        ef=search_width,
        layer=0,
        graph=graph,
        vectors=vectors,
    )

    # `search_layer` already sorts by score descending; take the top-k.
    return results[:k]


if __name__ == "__main__":
    # Create HNSWGraph M=16, ef_construction=50
    graph = HNSWGraph(M=16, ef_construction=50)

    # Insert 500 random 128-dim vectors with ids "v0" to "v499"
    vectors = {}
    dim = 128
    n_vectors = 500
    rng = np.random.default_rng(0)

    for i in range(n_vectors):
        node_id = f"v{i}"
        vec = rng.normal(size=(dim,))
        insert(id=node_id, vector=vec, graph=graph, vectors=vectors)

    # Create random query vector
    query = rng.normal(size=(dim,))

    # Time the search using NumPy-only datetime/timedelta arithmetic.
    k = 5
    ef_search = 50
    start = np.datetime64("now")
    topk = hnsw_search(query=query, k=k, graph=graph, vectors=vectors, ef_search=ef_search)
    elapsed_ms = (np.datetime64("now") - start) / np.timedelta64(1, "ms")

    # Print top 5 results (id, score)
    print(f"Top {k} results (id, score):")
    print(topk)

    # Print time taken in milliseconds
    print(f"Time taken: {elapsed_ms:.3f} ms")

