import numpy as np


def cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single query vector and many vectors.

    query: 1D numpy array with shape (D,)
    vectors: 2D numpy array with shape (N, D)
    returns: 1D numpy array with shape (N,) containing similarity scores
    """

    # Convert inputs to NumPy arrays (without copying if already arrays).
    query = np.asarray(query)
    vectors = np.asarray(vectors)

    # Enforce expected dimensionality for clearer errors.
    if query.ndim != 1:
        raise ValueError(f"`query` must be 1D with shape (D,), got shape {query.shape}")
    if vectors.ndim != 2:
        raise ValueError(f"`vectors` must be 2D with shape (N, D), got shape {vectors.shape}")

    # Enforce matching feature dimensions (D in both query and vectors).
    if vectors.shape[1] != query.shape[0]:
        raise ValueError(
            f"Feature dimension mismatch: query has D={query.shape[0]} but vectors has D={vectors.shape[1]}"
        )

    # Compute dot products for all vectors at once: (N, D) @ (D,) -> (N,).
    dots = vectors @ query

    # Compute the L2 norm (magnitude) of the query vector: scalar.
    query_norm = np.linalg.norm(query)

    # Compute L2 norms of all vectors along axis=1: (N,).
    vector_norms = np.linalg.norm(vectors, axis=1)

    # Compute the cosine denominator for each vector: ||q|| * ||v_i|| -> (N,).
    denom = query_norm * vector_norms

    # Prepare output scores (default to 0 so we never divide by zero).
    scores = np.zeros(vectors.shape[0], dtype=float)

    # Only divide where the denominator is non-zero to avoid division-by-zero.
    valid = denom != 0

    # Fill only the valid entries; invalid entries remain 0.
    scores[valid] = dots[valid] / denom[valid]

    # Return the (N,) similarity scores.
    return scores


def top_k_indices(query: np.ndarray, vectors: np.ndarray, k: int) -> np.ndarray:
    """
    Return the indices of the top-k vectors by cosine similarity to `query`.

    Uses `np.argpartition` (not full sort) to select candidates, then sorts those
    candidates by score descending.
    """

    # Convert query/vectors to NumPy arrays for consistent behavior.
    query = np.asarray(query)
    vectors = np.asarray(vectors)

    # Coerce k to an integer (supports inputs like NumPy integer types).
    k = int(k)

    # Compute cosine similarity scores for all vectors first: (N,).
    scores = cosine_similarity(query, vectors)

    # If k is non-positive, return an empty index array.
    if k <= 0:
        return np.array([], dtype=int)

    # Determine how many vectors we actually have (N).
    n = scores.shape[0]

    # Clamp k so it never exceeds the number of available vectors.
    k = min(k, n)

    # Select top-k candidates using argpartition (no full sorting of all N elements).
    # -k means "partition such that the k largest elements are moved to the last k slots".
    candidate_indices = np.argpartition(scores, -k)[-k:]

    # Now sort the candidate indices by their scores descending.
    # np.argsort sorts ascending, so we reverse the order with [::-1].
    sorted_topk_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

    # Return indices of the top-k vectors, sorted by score.
    return sorted_topk_indices


if __name__ == "__main__":
    # -----------------------------
    # Test 1: same vector -> 1.0
    # -----------------------------

    # Define a deterministic query vector.
    query = np.array([1.0, 2.0, 3.0], dtype=float)

    # Make `vectors` contain the exact same vector (shape becomes (1, D)).
    vectors = query[None, :]

    # Compute cosine similarity scores (should be exactly 1.0 for identical non-zero vectors).
    scores = cosine_similarity(query, vectors)

    # Print the scores for visual confirmation.
    print("Same vector scores:", scores)

    # Numerically confirm the score is close to 1.0.
    assert np.isclose(scores[0], 1.0), f"Expected ~1.0, got {scores[0]}"

    # -----------------------------
    # Test 2: orthogonal -> 0.0
    # -----------------------------

    # Choose a query vector aligned to the x-axis.
    query = np.array([1.0, 0.0, 0.0], dtype=float)

    # Create vectors that include both a parallel and an orthogonal direction.
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],  # parallel to query => cosine should be 1
            [0.0, 1.0, 0.0],  # orthogonal to query => cosine should be 0
        ],
        dtype=float,
    )

    # Compute cosine similarity scores.
    scores = cosine_similarity(query, vectors)

    # Print the scores for visual confirmation.
    print("Orthogonal scores:", scores)

    # Confirm parallel is ~1.0 and orthogonal is ~0.0.
    assert np.isclose(scores[0], 1.0), f"Expected ~1.0, got {scores[0]}"
    assert np.isclose(scores[1], 0.0), f"Expected ~0.0, got {scores[1]}"

    # -----------------------------------
    # Test 3: random vectors -> print
    # -----------------------------------

    # Create a reproducible random generator.
    rng = np.random.default_rng(42)

    # Generate a random query vector of dimension D=5.
    query = rng.normal(size=(5,))

    # Generate N=10 random vectors, each with dimension D=5.
    vectors = rng.normal(size=(10, 5))

    # Compute similarity scores and print them.
    scores = cosine_similarity(query, vectors)

    # Print random cosine similarity scores.
    print("Random scores:", scores)

