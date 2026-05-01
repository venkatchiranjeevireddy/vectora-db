import random as pyrandom

import numpy as np
import pytest

from vector_db.core.brute_force import BruteForceSearch
from vector_db.core.store import VectorStore
from vector_db.hnsw.graph import HNSWGraph
from vector_db.hnsw.insert import insert
from vector_db.hnsw.search import hnsw_search


def _build_test_data(dim: int = 128, n_vectors: int = 200, seed: int = 123):
    """
    Create deterministic random vectors for repeatable tests.

    Note: HNSWGraph level assignment uses `random.random()` (stdlib), so we also seed
    Python's `random` to keep the graph structure deterministic.
    """

    # Seed both RNGs: numpy for vectors/query, python-random for HNSW levels.
    np_rng = np.random.default_rng(seed)
    pyrandom.seed(seed)

    ids = [f"v{i}" for i in range(n_vectors)]
    vectors = {node_id: np_rng.normal(size=(dim,)).astype(float) for node_id in ids}
    query = np_rng.normal(size=(dim,)).astype(float)
    return ids, vectors, query


def test_insert_and_search():
    """
    Basic integration: insert vectors into HNSW and verify search output shape/ranges.
    """

    dim = 128
    ids, vectors, query = _build_test_data(dim=dim, n_vectors=200, seed=123)

    # Create HNSWGraph and insert all vectors.
    graph = HNSWGraph(M=16, ef_construction=100)
    hnsw_vectors = {}
    for node_id in ids:
        insert(id=node_id, vector=vectors[node_id], graph=graph, vectors=hnsw_vectors)

    # Search with random query.
    k = 10
    results = hnsw_search(
        query=query, k=k, graph=graph, vectors=hnsw_vectors, ef_search=50
    )

    assert len(results) == k

    inserted_ids = set(ids)
    returned_ids = [rid for (rid, _score) in results]
    assert set(returned_ids).issubset(inserted_ids)

    # Cosine similarity should be within [-1, 1] (allow tiny float tolerance).
    for _rid, score in results:
        assert -1.0 - 1e-6 <= float(score) <= 1.0 + 1e-6


def test_recall_vs_bruteforce():
    """
    Quality check: compare HNSW top-k overlap vs brute-force cosine search.
    """

    dim = 128
    ids, vectors, query = _build_test_data(dim=dim, n_vectors=200, seed=456)

    # Build HNSW.
    graph = HNSWGraph(M=16, ef_construction=100)
    hnsw_vectors = {}
    for node_id in ids:
        insert(id=node_id, vector=vectors[node_id], graph=graph, vectors=hnsw_vectors)

    # HNSW results.
    k = 10
    ef_search = 50
    hnsw_results = hnsw_search(
        query=query, k=k, graph=graph, vectors=hnsw_vectors, ef_search=ef_search
    )
    hnsw_ids = {rid for (rid, _score) in hnsw_results}

    # Brute force results (same vectors and query).
    store = VectorStore(dim=dim)
    vectors_matrix = np.stack([vectors[node_id] for node_id in ids], axis=0)
    store.add_batch(ids, vectors_matrix)

    brute = BruteForceSearch(store=store)
    brute_results = brute.search(query=query, k=k)
    brute_ids = {rid for (rid, _score) in brute_results}

    # Compute recall as overlap / k.
    overlap = len(hnsw_ids.intersection(brute_ids))
    recall = overlap / k

    print(f"Recall@10: {recall}")
    assert recall >= 0.5


def test_entry_point_is_set():
    """
    After inserting vectors, graph.entry_point should be non-null and valid.
    """

    dim = 128
    ids, vectors, _query = _build_test_data(dim=dim, n_vectors=200, seed=789)

    graph = HNSWGraph(M=16, ef_construction=100)
    hnsw_vectors = {}
    for node_id in ids:
        insert(id=node_id, vector=vectors[node_id], graph=graph, vectors=hnsw_vectors)

    assert graph.entry_point is not None
    assert graph.entry_point in ids

