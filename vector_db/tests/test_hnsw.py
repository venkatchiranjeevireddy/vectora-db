import random as pyrandom

import numpy as np
import pytest

from vector_db.core.brute_force import BruteForceSearch
from vector_db.core.store import VectorStore
from vector_db.hnsw.graph import HNSWGraph
from vector_db.hnsw.insert import insert
from vector_db.hnsw.neighbor_selection import select_neighbors_heuristic
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



def test_higher_level_insertion_connects_to_existing_graph(monkeypatch):
    """A new top-level entry point must not link to itself or isolate old nodes."""
    graph = HNSWGraph(M=4, ef_construction=10, max_level=3)
    levels = iter((0, 2))
    monkeypatch.setattr(graph, "get_random_level", lambda: next(levels))
    vectors = {}

    insert("first", np.array([1.0, 0.0]), graph, vectors)
    insert("second", np.array([0.9, 0.1]), graph, vectors)

    assert graph.entry_point == "second"
    assert "first" in graph.get_neighbors("second", 0)
    assert "second" in graph.get_neighbors("first", 0)
    for layer, nodes in enumerate(graph.layers):
        for node_id in nodes:
            assert node_id not in graph.get_neighbors(node_id, layer)


def test_search_returns_k_when_ef_search_is_smaller(monkeypatch):
    """A narrow search setting must not silently truncate top-k results."""
    graph = HNSWGraph(M=4, ef_construction=10, max_level=0)
    monkeypatch.setattr(graph, "get_random_level", lambda: 0)
    vectors = {}
    for node_id, values in (
        ("a", [1.0, 0.0]),
        ("b", [0.8, 0.2]),
        ("c", [0.0, 1.0]),
    ):
        insert(node_id, np.array(values), graph, vectors)

    results = hnsw_search(
        query=np.array([1.0, 0.0]),
        k=3,
        graph=graph,
        vectors=vectors,
        ef_search=1,
    )
    assert len(results) == 3
    assert len({node_id for node_id, _ in results}) == 3


@pytest.mark.parametrize("sample", [0.0, 1e-300, 0.5, 0.999999])
def test_random_level_respects_allocated_layers(monkeypatch, sample):
    """Even extreme random samples must produce an allocated graph level."""
    graph = HNSWGraph(M=4, max_level=2)
    monkeypatch.setattr(pyrandom, "random", lambda: sample)

    level = graph.get_random_level()

    assert 0 <= level <= graph.max_level
    graph.add_node("sampled", level)
    assert graph.levels["sampled"] == level
    assert "sampled" in graph.layers[level]


def test_random_level_zero_sample_is_capped(monkeypatch):
    """A zero random sample must not cause a logarithm error."""
    graph = HNSWGraph(M=4, max_level=1)
    monkeypatch.setattr(pyrandom, "random", lambda: 0.0)

    assert graph.get_random_level() == 1


def test_neighbor_selection_deduplicates_candidate_ids():
    """Repeated candidates must not create duplicate graph edges."""
    vectors = {
        "a": np.array([1.0, 0.0]),
        "b": np.array([0.0, 1.0]),
    }
    selected = select_neighbors_heuristic(
        query=np.array([1.0, 0.0]),
        candidates=[("a", 2.0), ("a", 1.5), ("b", 0.5)],
        M=3,
        layer=0,
        graph=None,
        vectors=vectors,
    )

    assert selected.count("a") == 1
    assert len(selected) == len(set(selected))


@pytest.mark.parametrize("ef_construction", [0, -1, -10])
def test_graph_rejects_nonpositive_construction_width(ef_construction):
    """Invalid construction beam widths should fail at initialization."""
    with pytest.raises(ValueError, match="ef_construction.*positive"):
        HNSWGraph(M=4, ef_construction=ef_construction)


def test_graph_accepts_positive_construction_width():
    graph = HNSWGraph(M=4, ef_construction=1, max_level=0)

    assert graph.ef_construction == 1


def test_hnsw_insert_copies_caller_owned_vector(monkeypatch):
    """Mutating the original input after indexing must not change search data."""
    graph = HNSWGraph(M=4, ef_construction=10, max_level=0)
    monkeypatch.setattr(graph, "get_random_level", lambda: 0)
    vectors = {}
    original = np.array([1.0, 2.0])

    insert("first", original, graph, vectors)
    original[:] = [-5.0, -6.0]

    np.testing.assert_array_equal(vectors["first"], [1.0, 2.0])


def test_hnsw_subsequent_insert_copies_caller_owned_vector(monkeypatch):
    """The same ownership guarantee applies after the entry point exists."""
    graph = HNSWGraph(M=4, ef_construction=10, max_level=0)
    monkeypatch.setattr(graph, "get_random_level", lambda: 0)
    vectors = {}
    insert("first", np.array([1.0, 0.0]), graph, vectors)
    original = np.array([0.0, 1.0])

    insert("second", original, graph, vectors)
    original[:] = [9.0, 9.0]

    np.testing.assert_array_equal(vectors["second"], [0.0, 1.0])
