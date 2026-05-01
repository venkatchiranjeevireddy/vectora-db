import numpy as np  # type: ignore


def _cosine_similarity_vector_to_matrix(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single vector and a matrix of vectors.

    This helper exists so neighbor selection can quantify:
    - similarity(query, candidate) (already provided as `candidate` score)
    - similarity(candidate, selected_neighbor) (computed here)

    Returns:
        1D array of cosine similarities with shape (N,) where N is mat.shape[0].
        If either the query vector or any row in mat has zero norm, similarity is 0.
    """

    # Normalize inputs to expected shapes.
    vec = np.asarray(vec)
    mat = np.asarray(mat)

    if vec.ndim != 1:
        raise ValueError(f"`vec` must be 1D with shape (D,), got {vec.shape}")
    if mat.ndim != 2:
        raise ValueError(f"`mat` must be 2D with shape (N, D), got {mat.shape}")
    if mat.shape[1] != vec.shape[0]:
        raise ValueError(
            f"Dimension mismatch: vec has D={vec.shape[0]}, but mat has D={mat.shape[1]}"
        )

    # Dot products for all candidates at once: (N, D) @ (D,) -> (N,).
    dots = mat @ vec

    # Norms for division; computed separately so we can mask zero denominators.
    vec_norm = np.linalg.norm(vec)
    mat_norms = np.linalg.norm(mat, axis=1)

    # Denominator for cosine similarity.
    denom = vec_norm * mat_norms

    # Default to 0 for any pair involving a zero vector (avoids division-by-zero).
    scores = np.zeros(mat.shape[0], dtype=float)
    valid = denom != 0
    scores[valid] = dots[valid] / denom[valid]

    return scores


def select_neighbors_heuristic(query, candidates, M, layer, graph, vectors):
    """
    Diversified neighbor selection heuristic for HNSW graph quality.

    Algorithm goal:
    - Pick neighbors that are both high-scoring (close to query) and diverse
      (not redundant with already-selected neighbors).

    Why this matters:
    - Simply taking the top-M by score often yields neighbors that cluster
      around the same direction in vector space, reducing the graph's ability
      to navigate effectively.

    Args:
        query: 1D NumPy array with shape (D,).
        candidates: List of (id, score) tuples where score ~= cosine(query, id).
        M: Max number of neighbors to return.
        layer: Current layer index (not used directly, but provided for API parity).
        graph: HNSWGraph instance (not used directly in this heuristic).
        vectors: Dict {id: numpy array} holding vectors for scoring/diversity checks.

    Returns:
        List of up to M neighbor IDs (strings/ints/etc.), selected in order.
    """

    # Convert query to a NumPy array for vector operations.
    query = np.asarray(query)
    M = int(M)

    if M <= 0:
        return []

    # Sort by descending similarity-to-query so we try the best candidates first.
    sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    # `result_ids` are the selected neighbors so far.
    result_ids = []

    # Keep selected neighbor vectors in a matrix so we can compute diversity scores vectorized.
    # Start with an empty (0, D) matrix.
    D = query.shape[0]
    selected_vecs = np.empty((0, D), dtype=float)

    # Iterate candidates from best to worst (this loop is the heuristic itself).
    for cand_id, cand_score in sorted_candidates:
        # Stop once we have enough neighbors.
        if len(result_ids) >= M:
            break

        # Always add the first candidate: it establishes an initial direction.
        if not result_ids:
            result_ids.append(cand_id)
            selected_vecs = np.vstack([selected_vecs, np.asarray(vectors[cand_id], dtype=float)])
            continue

        # Compute candidate-to-selected-neighbors similarity (diversity penalty).
        # We decide to keep the candidate only if it provides new information.
        cand_vec = np.asarray(vectors[cand_id], dtype=float)

        # Similarity(cand, each_selected_neighbor) -> (len(result_ids),)
        cand_to_selected = _cosine_similarity_vector_to_matrix(cand_vec, selected_vecs)

        # Diversity check (exact rule from the spec):
        # - compute score_cq = similarity(query, cand)   (already provided as cand_score)
        # - compute score_cr = similarity(cand, r) for each already-selected neighbor r
        # - keep c only if score_cq > score_cr for ALL r
        score_cq = float(cand_score)
        score_cr = cand_to_selected  # (len(result_ids),)

        # "score_cq > score_cr for ALL r" means: score_cq is strictly greater than
        # every candidate-to-selected similarity.
        keep = bool(np.all(score_cq > score_cr))

        # If not keeping, skip this candidate because it's too redundant.
        if keep:
            result_ids.append(cand_id)
            selected_vecs = np.vstack([selected_vecs, cand_vec.reshape(1, -1)])

    return result_ids


if __name__ == "__main__":
    # Create 20 random 128-dim vectors to act as our candidate pool.
    rng = np.random.default_rng(0)
    dim = 128
    vectors = {f"v{i}": rng.normal(size=(dim,)) for i in range(20)}

    # Random query vector; in HNSW this would be the embedding we search with.
    query = rng.normal(size=(dim,))

    # Build candidates as (id, cosine_similarity(query, id)).
    # We compute cosine(query, id) once so the heuristic can focus on diversity.
    query_norm = np.linalg.norm(query)
    candidate_ids = list(vectors.keys())
    mat = np.stack([vectors[i] for i in candidate_ids], axis=0)  # (20, D)
    dots = mat @ query  # (20,)
    mat_norms = np.linalg.norm(mat, axis=1)  # (20,)
    denom = query_norm * mat_norms

    scores = np.zeros(len(candidate_ids), dtype=float)
    valid = denom != 0
    scores[valid] = dots[valid] / denom[valid]

    candidates = list(zip(candidate_ids, scores.tolist()))

    # Run diversified selection with M=5.
    selected = select_neighbors_heuristic(
        query=query,
        candidates=candidates,
        M=5,
        layer=0,
        graph=None,
        vectors=vectors,
    )

    print("Selected neighbor ids:", selected)
    print("Diversified selection working")

