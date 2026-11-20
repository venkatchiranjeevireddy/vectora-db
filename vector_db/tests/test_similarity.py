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
