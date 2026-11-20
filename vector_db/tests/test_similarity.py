"""Edge-case regression tests for cosine top-k retrieval."""

import numpy as np

from vector_db.core.similarity import top_k_indices


def test_top_k_empty_matrix_returns_empty_indices():
    query = np.array([1.0, 0.0, 0.0])
    vectors = np.empty((0, 3))

    indices = top_k_indices(query, vectors, k=5)

    assert indices.shape == (0,)
    assert indices.dtype.kind in "iu"


def test_top_k_larger_than_collection_returns_every_index():
    query = np.array([1.0, 0.0])
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])

    indices = top_k_indices(query, vectors, k=10)

    assert indices.tolist() == [0, 1]


def test_top_k_nonpositive_k_returns_empty_indices():
    query = np.array([1.0, 0.0])
    vectors = np.array([[1.0, 0.0]])

    assert top_k_indices(query, vectors, k=0).size == 0
    assert top_k_indices(query, vectors, k=-1).size == 0


def test_cosine_similarity_handles_zero_vectors_without_nan():
    """Zero-length vectors have zero similarity, not NaN or infinity."""
    from vector_db.core.similarity import cosine_similarity

    query = np.array([1.0, 0.0])
    vectors = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])

    scores = cosine_similarity(query, vectors)

    assert np.all(np.isfinite(scores))
    np.testing.assert_allclose(scores, [0.0, 1.0, -1.0])


def test_cosine_similarity_zero_query_returns_zero_scores():
    from vector_db.core.similarity import cosine_similarity

    scores = cosine_similarity(
        np.array([0.0, 0.0]),
        np.array([[1.0, 0.0], [0.0, 0.0]]),
    )

    np.testing.assert_array_equal(scores, [0.0, 0.0])


def test_top_k_full_collection_preserves_input_order_for_ties():
    """Equal-score vectors retain insertion order when requesting all rows."""
    query = np.array([1.0, 0.0])
    vectors = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, -1.0],
        [-1.0, 0.0],
    ])

    indices = top_k_indices(query, vectors, k=len(vectors))

    assert indices.tolist() == [1, 0, 2, 3]


def test_top_k_oversized_request_preserves_order_for_ties():
    query = np.array([1.0, 0.0])
    vectors = np.array([[0.0, 1.0], [0.0, -1.0]])

    assert top_k_indices(query, vectors, k=20).tolist() == [0, 1]


def test_top_k_subset_tie_at_cutoff_prefers_earlier_indices():
    """A tied cutoff must choose the first matching vectors consistently."""
    query = np.array([1.0, 0.0])
    vectors = np.array([
        [0.0, 1.0],
        [0.0, -1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ])

    for _ in range(3):
        assert top_k_indices(query, vectors, k=3).tolist() == [2, 0, 1]


def test_top_k_subset_without_cutoff_ties_remains_score_sorted():
    query = np.array([1.0, 0.0])
    vectors = np.array([
        [-1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    assert top_k_indices(query, vectors, k=2).tolist() == [2, 1]
