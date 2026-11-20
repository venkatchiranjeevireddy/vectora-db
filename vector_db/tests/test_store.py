"""Regression tests for the in-memory vector store."""

import numpy as np
import pytest

from vector_db.core.store import VectorStore


@pytest.mark.parametrize("dim", [0, -1, -128])
def test_store_rejects_nonpositive_dimensions(dim):
    with pytest.raises(ValueError, match="positive integer"):
        VectorStore(dim=dim)


def test_store_accepts_positive_dimension_and_stores_vector():
    store = VectorStore(dim=3)
    vector = np.array([1.0, 2.0, 3.0])

    store.add("example", vector)

    assert len(store) == 1
    np.testing.assert_array_equal(store.get("example"), vector)
    assert store.all_vectors().shape == (1, 3)


def test_empty_store_preserves_vector_dimension():
    store = VectorStore(dim=3)

    assert store.all_vectors().shape == (0, 3)
    assert store.all_ids() == []


def test_single_insert_copies_caller_owned_vector():
    """Changing an input array must not change a previously stored vector."""
    store = VectorStore(dim=2)
    original = np.array([1.0, 2.0])
    store.add("item", original)

    original[0] = 99.0

    np.testing.assert_array_equal(store.get("item"), [1.0, 2.0])
