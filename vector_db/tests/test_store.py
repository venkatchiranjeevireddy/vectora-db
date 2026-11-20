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


def test_batch_insert_copies_caller_owned_matrix():
    """Editing the original batch must not change stored vectors."""
    store = VectorStore(dim=2)
    original = np.array([[1.0, 2.0], [3.0, 4.0]])
    store.add_batch(["first", "second"], original)

    original[:, :] = -1.0

    np.testing.assert_array_equal(store.get("first"), [1.0, 2.0])
    np.testing.assert_array_equal(store.get("second"), [3.0, 4.0])
    assert store.all_ids() == ["first", "second"]


def test_get_returns_copy_that_cannot_mutate_stored_vector():
    store = VectorStore(dim=2)
    store.add("item", np.array([1.0, 2.0]))

    retrieved = store.get("item")
    retrieved[0] = 99.0

    np.testing.assert_array_equal(store.get("item"), [1.0, 2.0])


def test_all_vectors_returns_independent_snapshot():
    store = VectorStore(dim=2)
    store.add_batch(["first", "second"], np.array([[1.0, 2.0], [3.0, 4.0]]))

    snapshot = store.all_vectors()
    snapshot[:, :] = -10.0

    np.testing.assert_array_equal(store.get("first"), [1.0, 2.0])
    np.testing.assert_array_equal(store.get("second"), [3.0, 4.0])


@pytest.mark.parametrize(
    "ids, vectors",
    [
        (["only-one"], np.array([[1.0, 2.0], [3.0, 4.0]])),
        (["first"], np.array([[1.0, 2.0, 3.0]])),
        (["first"], np.array([1.0, 2.0])),
    ],
)
def test_invalid_batch_does_not_partially_update_store(ids, vectors):
    """Validation errors must leave existing indexed vectors untouched."""
    store = VectorStore(dim=2)
    store.add("existing", np.array([7.0, 8.0]))

    with pytest.raises(ValueError):
        store.add_batch(ids, vectors)

    assert store.all_ids() == ["existing"]
    np.testing.assert_array_equal(store.get("existing"), [7.0, 8.0])


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_single_insert_rejects_nonfinite_values_without_mutation(invalid_value):
    store = VectorStore(dim=2)
    store.add("existing", np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="finite values"):
        store.add("invalid", np.array([invalid_value, 3.0]))

    assert store.all_ids() == ["existing"]
    np.testing.assert_array_equal(store.get("existing"), [1.0, 2.0])


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_batch_insert_rejects_nonfinite_values_atomically(invalid_value):
    store = VectorStore(dim=2)
    store.add("existing", np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="finite values"):
        store.add_batch(
            ["valid", "invalid"],
            np.array([[3.0, 4.0], [invalid_value, 5.0]]),
        )

    assert store.all_ids() == ["existing"]
    np.testing.assert_array_equal(store.get("existing"), [1.0, 2.0])
