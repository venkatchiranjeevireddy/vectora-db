import numpy as np

from vector_db.core.similarity import cosine_similarity
from vector_db.core.store import VectorStore


class BruteForceSearch:
    """
    Brute-force nearest-neighbor search using cosine similarity.

    This implementation scores the query against every vector in the provided
    VectorStore, then selects the top-k results.
    """

    def __init__(self, store: VectorStore):
        """
        Args:
            store: A VectorStore instance containing vectors and IDs.
        """

        self.store = store

    def search(self, query: np.ndarray, k: int):
        """
        Search the store for the top-k vectors by cosine similarity.

        Args:
            query: 1D NumPy array with shape (D,).
            k: Number of results to return.

        Returns:
            A list of (id, score) tuples for the top-k matches, sorted
            by score descending.
        """

        # Normalize input types for consistent NumPy operations.
        query = np.asarray(query)
        k = int(k)

        # Validate query shape early to avoid confusing downstream errors.
        if query.ndim != 1:
            raise ValueError(f"`query` must be 1D with shape (D,), got shape {query.shape}")

        # If k is non-positive, there are no results to return.
        if k <= 0:
            return []

        # Load all vectors and IDs from the store (vectorized operations follow).
        vectors = self.store.all_vectors()  # shape: (N, D)
        ids = np.asarray(self.store.all_ids(), dtype=object)  # shape: (N,)

        # Compute cosine similarity scores for all vectors at once: (N,).
        scores = cosine_similarity(query, vectors)

        # Clamp k to available number of vectors to keep selection safe.
        n = scores.shape[0]
        k = min(k, n)
        if k == 0:
            return []

        # Use argpartition to select the candidate top-k indices without sorting all N.
        candidate_indices = np.argpartition(scores, -k)[-k:]

        # Sort only those candidates by score descending to get final ranking.
        ranked_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

        # Gather IDs and scores for the ranked top-k results.
        top_ids = ids[ranked_indices]
        top_scores = scores[ranked_indices]

        # Convert to the requested output format.
        return list(zip(top_ids.tolist(), top_scores.tolist()))


if __name__ == "__main__":
    # Create store dim=128
    dim = 128
    store = VectorStore(dim=dim)

    # Add 1000 random vectors with ids "v0" to "v999"
    rng = np.random.default_rng(0)
    n_vectors = 1000
    ids = [f"v{i}" for i in range(n_vectors)]
    vectors = rng.normal(size=(n_vectors, dim))
    store.add_batch(ids, vectors)

    # Create BruteForceSearch
    searcher = BruteForceSearch(store=store)

    # Search with random query k=5 and time it
    query = rng.normal(size=(dim,))
    k = 5

    # Time the search using NumPy-only datetime/timedelta arithmetic.
    start = np.datetime64("now")
    results = searcher.search(query=query, k=k)
    elapsed_ms = (np.datetime64("now") - start) / np.timedelta64(1, "ms")

    # Print top 5 results (id, score)
    print("Top 5 results (id, score):")
    print(results)

    # Print time taken in milliseconds
    print(f"Time taken: {elapsed_ms:.3f} ms")

