import numpy as np
import pytest

from vector_db.core.brute_force import BruteForceSearch
from vector_db.core.similarity import cosine_similarity, top_k_indices
from vector_db.core.store import VectorStore


def test_cosine_similarity_same_vector_returns_one():
    # Same non-zero vector should produce cosine similarity 1.0.
    query = np.array([1.0, 2.0, 3.0], dtype=float)
    vectors = np.stack([query, query], axis=0)  # shape: (2, D)

    scores = cosine_similarity(query, vectors)

    assert scores.shape == (2,)
    assert np.isclose(scores[0], 1.0)
    assert np.isclose(scores[1], 1.0)


def test_cosine_similarity_orthogonal_returns_zero():
    # Orthogonal vectors should produce cosine similarity 0.0.
    query = np.array([1.0, 0.0, 0.0], dtype=float)  # aligned with x-axis
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],  # parallel -> cosine 1
            [0.0, 1.0, 0.0],  # orthogonal -> cosine 0
        ],
        dtype=float,
    )

    scores = cosine_similarity(query, vectors)

    assert scores.shape == (2,)
    assert np.isclose(scores[0], 1.0)
    assert np.isclose(scores[1], 0.0)


def test_cosine_similarity_zero_query_returns_zero_scores():
    # A zero query vector has norm 0, so cosine should safely return 0 scores (no crash).
    query = np.zeros(4, dtype=float)
    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    scores = cosine_similarity(query, vectors)

    assert scores.shape == (3,)
    assert np.allclose(scores, 0.0)


def test_top_k_indices_returns_k_and_valid_indices():
    rng = np.random.default_rng(123)
    query = rng.normal(size=(6,))
    vectors = rng.normal(size=(20, 6))
    k = 5

    topk = top_k_indices(query, vectors, k)

    assert isinstance(topk, np.ndarray)
    assert topk.shape == (k,)

    # Indices must be in the valid range [0, N-1].
    assert np.all(topk >= 0)
    assert np.all(topk < vectors.shape[0])


def test_top_k_indices_top_result_is_most_similar():
    rng = np.random.default_rng(321)
    query = rng.normal(size=(8,))
    vectors = rng.normal(size=(50, 8))
    k = 7

    scores = cosine_similarity(query, vectors)
    topk = top_k_indices(query, vectors, k)

    # top_k_indices returns top-k sorted by score descending,
    # so its first index must correspond to the maximum score.
    top_index = topk[0]
    max_score = scores.max()
    top_score = scores[top_index]

    assert np.isclose(top_score, max_score)


def test_vector_store_add_and_get():
    store = VectorStore(dim=3)
    store.add("a", np.array([1.0, 2.0, 3.0], dtype=float))

    got = store.get("a")
    assert got.shape == (3,)
    assert np.allclose(got, np.array([1.0, 2.0, 3.0], dtype=float))
    assert len(store) == 1


def test_vector_store_add_batch_and_len():
    store = VectorStore(dim=2)

    ids = ["v0", "v1", "v2", "v3", "v4"]
    vectors = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=float,
    )

    store.add_batch(ids, vectors)

    assert len(store) == 5
    all_vecs = store.all_vectors()
    assert all_vecs.shape == (5, 2)


def test_vector_store_get_wrong_id_raises_key_error():
    store = VectorStore(dim=2)
    store.add("present", np.array([1.0, 2.0], dtype=float))

    with pytest.raises(KeyError):
        _ = store.get("missing")


def test_brute_force_search_returns_k_and_top_score_max():
    rng = np.random.default_rng(999)
    dim = 16
    n = 200
    k = 10

    store = VectorStore(dim=dim)
    ids = [f"v{i}" for i in range(n)]
    vectors = rng.normal(size=(n, dim))
    store.add_batch(ids, vectors)

    query = rng.normal(size=(dim,))
    searcher = BruteForceSearch(store=store)

    results = searcher.search(query=query, k=k)

    assert isinstance(results, list)
    assert len(results) == k

    # Extract scores and validate range.
    scores = np.array([score for _, score in results], dtype=float)
    assert np.all(scores <= 1.0 + 1e-7)
    assert np.all(scores >= -1.0 - 1e-7)

    # The top result must match the maximum cosine similarity score across all vectors.
    all_scores = cosine_similarity(query, store.all_vectors())
    max_score = float(all_scores.max())
    top_score = float(results[0][1])

    assert np.isclose(top_score, max_score)

