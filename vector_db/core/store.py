import numpy as np


class VectorStore:
    """
    Simple in-memory vector store mapping IDs to vectors.

    Notes:
    - This is intentionally minimal and uses only NumPy for array handling.
    - Vectors are stored as float arrays of shape (D,).
    """

    def __init__(self, dim: int):
        """
        Initialize an empty vector store.

        Args:
            dim: The dimensionality D of each vector.
        """

        self.dim = int(dim)
        # Internal storage: id -> vector (np.ndarray of shape (D,))
        self._data = {}

    def add(self, id, vector: np.ndarray) -> None:
        """
        Add a single vector to the store.

        Args:
            id: A string or int identifier for the vector.
            vector: A 1D NumPy array with shape (D,).
        """

        vec = np.asarray(vector)

        if vec.ndim != 1:
            raise ValueError(f"`vector` must be 1D with shape (D,), got shape {vec.shape}")
        if vec.shape[0] != self.dim:
            raise ValueError(f"Vector dimension mismatch: expected D={self.dim}, got D={vec.shape[0]}")

        # Store as float for consistent downstream math.
        self._data[id] = vec.astype(float, copy=False)

    def add_batch(self, ids, vectors: np.ndarray) -> None:
        """
        Add multiple vectors to the store in one call.

        Args:
            ids: A list of IDs of length N.
            vectors: A 2D NumPy array with shape (N, D).
        """

        vecs = np.asarray(vectors)

        if vecs.ndim != 2:
            raise ValueError(f"`vectors` must be 2D with shape (N, D), got shape {vecs.shape}")

        n = vecs.shape[0]
        if len(ids) != n:
            raise ValueError(f"`ids` length must match vectors rows: len(ids)={len(ids)} != N={n}")
        if vecs.shape[1] != self.dim:
            raise ValueError(
                f"Vectors dimension mismatch: expected D={self.dim}, got D={vecs.shape[1]}"
            )

        # Insert each vector; this is storage-level iteration (not numeric computation).
        vecs = vecs.astype(float, copy=False)
        for i, id_ in enumerate(ids):
            self._data[id_] = vecs[i]

    def get(self, id):
        """
        Retrieve a vector by ID.

        Args:
            id: The identifier used when the vector was added.

        Returns:
            The stored vector as a NumPy array of shape (D,).
        """

        if id not in self._data:
            raise KeyError(f"VectorStore has no vector with id={id!r}")
        return self._data[id]

    def all_vectors(self) -> np.ndarray:
        """
        Get all stored vectors.

        Returns:
            A 2D NumPy array with shape (N, D).
            If the store is empty, returns an array with shape (0, D).
        """

        if not self._data:
            return np.empty((0, self.dim), dtype=float)

        # Preserve insertion order of dict (Python 3.7+).
        vectors = list(self._data.values())
        return np.stack(vectors, axis=0)

    def all_ids(self) -> list:
        """
        Get all stored IDs.

        Returns:
            A list of IDs of length N.
        """

        return list(self._data.keys())

    def __len__(self) -> int:
        """
        Return the number of vectors stored.
        """

        return len(self._data)


if __name__ == "__main__":
    # Create store dim=4
    store = VectorStore(dim=4)

    # Add 3 vectors manually
    store.add("v1", np.array([1.0, 0.0, 0.0, 0.0]))
    store.add("v2", np.array([0.0, 1.0, 0.0, 0.0]))
    store.add("v3", np.array([0.0, 0.0, 1.0, 0.0]))

    # Add_batch 5 vectors
    ids_batch = ["b1", "b2", "b3", "b4", "b5"]
    vectors_batch = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    store.add_batch(ids_batch, vectors_batch)

    # Print len, get one vector, print all_ids
    print("len:", len(store))
    print("get('v2'):", store.get("v2"))
    print("all_ids:", store.all_ids())

