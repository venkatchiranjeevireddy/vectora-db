import pickle

import numpy as np  # type: ignore

from vector_db.hnsw.graph import HNSWGraph
from vector_db.hnsw.insert import insert
from vector_db.hnsw.search import hnsw_search


def save_index(graph, vectors, filepath: str) -> str:
    """
    Save an HNSW graph and vectors dict to disk using pickle.

    Args:
        graph: HNSWGraph instance.
        vectors: dict {id: numpy array}.
        filepath: Base path (without extension). Saved file will be filepath + ".pkl".

    Returns:
        The full saved filepath including the ".pkl" extension.
    """

    out_path = f"{filepath}.pkl"

    with open(out_path, "wb") as f:
        pickle.dump({"graph": graph, "vectors": vectors}, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Index saved to {out_path}")
    return out_path


def load_index(filepath: str):
    """
    Load an HNSW graph and vectors dict from disk using pickle.

    Args:
        filepath: Base path (without extension) or full path. If filepath does not end
            with ".pkl", this function will load filepath + ".pkl".

    Returns:
        (graph, vectors) tuple.
    """

    in_path = filepath if filepath.endswith(".pkl") else f"{filepath}.pkl"

    with open(in_path, "rb") as f:
        payload = pickle.load(f)

    graph = payload["graph"]
    vectors = payload["vectors"]

    print(f"Index loaded from {in_path}")
    return graph, vectors


if __name__ == "__main__":
    # Create HNSWGraph M=16, ef_construction=100
    graph = HNSWGraph(M=16, ef_construction=100)
    vectors = {}

    # Insert 100 random 128-dim vectors
    dim = 128
    n = 100
    rng = np.random.default_rng(0)
    for i in range(n):
        node_id = f"v{i}"
        vec = rng.normal(size=(dim,)).astype(float)
        insert(id=node_id, vector=vec, graph=graph, vectors=vectors)

    # Save to "vector_db/saved_index" (writes "vector_db/saved_index.pkl")
    save_path = save_index(graph=graph, vectors=vectors, filepath="vector_db/saved_index")

    # Reload it
    loaded_graph, loaded_vectors = load_index("vector_db/saved_index")

    # Run hnsw_search on reloaded index k=5
    query = rng.normal(size=(dim,)).astype(float)
    results = hnsw_search(query=query, k=5, graph=loaded_graph, vectors=loaded_vectors, ef_search=50)
    print("Search results:", results)

    print("Persistence working")

