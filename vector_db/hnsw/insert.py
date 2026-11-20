import heapq

import numpy as np  # type: ignore

from vector_db.core.similarity import cosine_similarity
from vector_db.hnsw.graph import HNSWGraph
from vector_db.hnsw.neighbor_selection import select_neighbors_heuristic


def search_layer(
    query: np.ndarray,
    entry_point,
    ef: int,
    layer: int,
    graph: HNSWGraph,
    vectors,
):
    """
    Greedy/beam search within a single HNSW layer to find the top `ef` closest nodes.

    This is the "graph search" primitive used during insertion (to find candidate neighbors).

    Min-heap for candidates, max-heap for results:
    - Candidates heap stores (distance, node_id) and is a min-heap by distance.
      We define: distance = -cosine_similarity (so smaller distance => higher similarity).
    - Results heap stores (-distance, node_id) and behaves like a max-heap by distance.

    Args:
        query: 1D NumPy array with shape (D,).
        entry_point: Node ID to start searching from.
        ef: Candidate width / max number of results to keep.
        layer: Which graph layer to search.
        graph: HNSWGraph instance (provides neighbor lists).
        vectors: Dict mapping node_id -> 1D NumPy vector of shape (D,).

    Returns:
        List of (id, score) tuples, sorted by score descending, of length <= ef.
        Score is cosine similarity in [-1, 1] (when vectors are non-zero).
    """

    # Normalize and validate inputs.
    query = np.asarray(query)
    ef = int(ef)
    layer = int(layer)

    # If ef is not positive, there is nothing to search.
    if ef <= 0:
        return []

    # Keep track of visited nodes so we don't expand the same node repeatedly.
    visited = set()
    visited.add(entry_point)

    # Define distance = -score where score is cosine similarity.
    # This lets us use a min-heap (smallest distance) to expand the best (closest) candidate.
    entry_vec = vectors[entry_point]
    entry_score = float(cosine_similarity(query, entry_vec[None, :])[0])
    entry_dist = -entry_score

    # Min-heap for candidates: (distance, node_id)
    candidates = [(entry_dist, entry_point)]

    # "Max-heap" for results by distance: store (-distance, node_id).
    # Since heapq is a min-heap, pushing -distance makes the largest distance appear on top.
    results = [(-entry_dist, entry_point)]

    # Continue searching until there are no more candidates to expand.
    while candidates:
        # Expand the best candidate so far.
        dist, curr_id = heapq.heappop(candidates)

        # If we already have ef results, we can stop when the best remaining candidate
        # is worse than the current worst result.
        if len(results) >= ef:
            # Results store (-distance), so the worst (largest distance) is:
            #   worst_dist = -results[0][0]
            worst_dist = -results[0][0]

            # If the current candidate's distance is already larger than the worst result,
            # no future expansions can improve the results set.
            if dist > worst_dist:
                break

        # Explore the neighborhood of the current node at the given layer.
        # Guard: `curr_id` may not exist in this layer (e.g. due to differing
        # node levels). If so, skip expansion for this candidate.
        try:
            neighbor_list = graph.get_neighbors(curr_id, layer)
        except KeyError:
            continue

        for nb_id in neighbor_list:
            # Skip nodes we've already visited/expanded.
            if nb_id in visited:
                continue

            # Mark this neighbor as visited.
            visited.add(nb_id)

            # Compute cosine similarity score between query and this neighbor vector.
            nb_vec = vectors[nb_id]
            nb_score = float(cosine_similarity(query, nb_vec[None, :])[0])
            nb_dist = -nb_score

            # Determine whether this neighbor could be relevant based on current results.
            if len(results) < ef:
                # If we haven't filled the results set yet, always add into candidates.
                heapq.heappush(candidates, (nb_dist, nb_id))
                heapq.heappush(results, (-nb_dist, nb_id))

                # If results grew too large, remove the current worst.
                if len(results) > ef:
                    heapq.heappop(results)
            else:
                # If results is full, compare against the current worst result distance.
                worst_dist = -results[0][0]

                # Only consider this neighbor if it is closer (smaller distance) than the worst.
                if nb_dist < worst_dist:
                    # Add to candidates so it can potentially expand further.
                    heapq.heappush(candidates, (nb_dist, nb_id))

                    # Add to results and prune back down to ef.
                    heapq.heappush(results, (-nb_dist, nb_id))
                    if len(results) > ef:
                        heapq.heappop(results)

                # If nb_dist is not better than worst_dist, we don't add it.

    # Convert results heap entries to (id, score) tuples.
    # results entries are (-distance, id) = (score, id), because distance = -score.
    out = [(id_, score_) for (score_, id_) in results]

    # Sort by score descending so the first item is the best match.
    out.sort(key=lambda x: x[1], reverse=True)

    # Return up to ef results (heap may contain exactly ef unless graph is tiny).
    return out[:ef]


def insert(id, vector: np.ndarray, graph: HNSWGraph, vectors) -> None:
    """
    Insert a new vector into the HNSW graph by connecting it to neighbors across layers.

    This is a simplified HNSW insertion routine that:
    - assigns a level to the new node
    - searches from the current entry point down from the top layer to the node's level
    - at each layer, finds ef_construction candidates and connects up to M neighbors
    - updates the entry point if the new node has the highest level

    Args:
        id: New node ID.
        vector: 1D NumPy array with shape (D,).
        graph: HNSWGraph instance (layer structure and neighbor lists).
        vectors: Dict mapping existing node_id -> 1D NumPy vector. This function
            will also add the new node's vector to this dict.
    """

    # Convert the vector to a 1D NumPy array for consistency.
    vector = np.asarray(vector)

    # Assign a random level for this node using the graph's level distribution.
    level = int(graph.get_random_level())

    # If the graph is currently empty (no assigned levels), we only add the node as entry point.
    is_empty = len(graph.levels) == 0 or graph.entry_point is None
    if is_empty:
        # Store the vector so future searches can score it.
        vectors[id] = vector

        # Add the node to graph layers (neighbors start empty).
        graph.add_node(id, level)

        # The entry point is now the new node (handled by graph.add_node()).
        return

    # Compute the highest currently populated layer in the existing graph.
    # This avoids searching in empty "upper bound" layers.
    existing_top_layer = max(graph.levels.values())

    # Keep the existing entry point for neighbor discovery. Adding a
    # higher-level node can replace graph.entry_point before it has edges.
    previous_entry_point = graph.entry_point

    # Store the new vector in the vectors dict so searches can reference it.
    vectors[id] = vector

    # Add the node to graph layers from 0 up to its assigned level.
    # Its neighbor lists start empty and will be filled during insertion.
    graph.add_node(id, level)

    # Read back the actual assigned (possibly capped) level stored in the graph.
    # graph.add_node() can cap the provided level to graph.max_level.
    assigned_level = graph.levels[id]

    # Search from the previous entry point, which belongs to the existing
    # graph. The newly inserted node must not select itself as a neighbor.
    current_entry_point = previous_entry_point

    # 1) From the top layer down to node's level + 1: find the closest single node.
    # We do this greedily using search_layer with ef=1 at each higher layer.
    for layer in range(existing_top_layer, assigned_level, -1):
        # Guard: the current entry point might not exist in this higher layer.
        # If it doesn't, we cannot search neighbors from it for this layer.
        if current_entry_point not in graph.layers[layer]:
            continue

        # Search the layer with ef=1 => effectively "closest single node".
        top = search_layer(
            query=vector,
            entry_point=current_entry_point,
            ef=1,
            layer=layer,
            graph=graph,
            vectors=vectors,
        )

        # Update the entry point for the next (lower) layer if we found a node.
        if top:
            current_entry_point = top[0][0]

    # 2) From node's level down to 0: find ef_construction candidates and connect M neighbors.
    for layer in range(min(assigned_level, existing_top_layer), -1, -1):
        # The previous entry point exists in every lower layer. Never fall
        # back to the unconnected new node during neighbor discovery.
        if current_entry_point not in graph.layers[layer]:
            current_entry_point = previous_entry_point

        # Find ef_construction candidates in this layer.
        candidates = search_layer(
            query=vector,
            entry_point=current_entry_point,
            ef=graph.ef_construction,
            layer=layer,
            graph=graph,
            vectors=vectors,
        )

        # Diversified neighbor selection is critical for HNSW quality.
        # It avoids choosing a cluster of nearly-identical neighbors.
        candidates = [(node_id, score) for node_id, score in candidates if node_id != id]
        neighbors_to_link = select_neighbors_heuristic(
            query=vector,
            candidates=candidates,
            M=graph.M,
            layer=layer,
            graph=graph,
            vectors=vectors,
        )

        # Step through selected neighbors and create reciprocal connections.
        for nb_id in neighbors_to_link:
            # Add neighbor to the new node's list at this layer.
            new_neighbors = graph.get_neighbors(id, layer)
            if nb_id not in new_neighbors:
                new_neighbors.append(nb_id)
            graph.set_neighbors(id, layer, new_neighbors)

            # Add the new node to the neighbor's list at this layer (reciprocal link).
            try:
                nb_neighbors = graph.get_neighbors(nb_id, layer)
            except KeyError:
                # If nb_id doesn't exist at this layer, skip reciprocal linking.
                continue

            if id not in nb_neighbors:
                nb_neighbors.append(id)

            # Prune reciprocal neighbor list back using standard HNSW practice:
            # - layer 0 allows up to 2*M neighbors
            # - higher layers allow up to M neighbors
            max_conn = graph.M * 2 if layer == 0 else graph.M
            if len(nb_neighbors) > max_conn:
                nb_query = vectors[nb_id]

                # Build (candidate_id, score) pairs where score is cosine(nb_query, candidate).
                # This lets the heuristic diversify while keeping the list bounded.
                nb_candidates = []
                for cand_id in nb_neighbors:
                    cand_vec = vectors[cand_id]
                    cand_score = float(cosine_similarity(nb_query, cand_vec[None, :])[0])

                    # Ensure the reciprocal edge to the inserted node is likely
                    # retained after pruning (it was added for this reason).
                    if cand_id == id:
                        cand_score = 2.0

                    nb_candidates.append((cand_id, cand_score))

                nb_neighbors = select_neighbors_heuristic(
                    query=nb_query,
                    candidates=nb_candidates,
                    M=max_conn,
                    layer=layer,
                    graph=graph,
                    vectors=vectors,
                )

            graph.set_neighbors(nb_id, layer, nb_neighbors)

        # Move the entry point down for the next layer.
        # Using the best candidate helps guide the next search layer.
        if candidates:
            current_entry_point = candidates[0][0]

    # 3) Update entry point if the new node's level is higher than the current entry point level.
    # graph.add_node() already attempts to do this, but we keep this explicit step for correctness.
    if graph.entry_point is None or graph.levels[id] > graph.levels.get(graph.entry_point, -1):
        graph.entry_point = id


if __name__ == "__main__":
    # Create HNSWGraph with M=16, ef_construction=50
    graph = HNSWGraph(M=16, ef_construction=50)

    # Start with an empty vectors dict that will hold id -> vector.
    vectors = {}

    # Insert 20 random 128-dim vectors with ids "v0" to "v19"
    dim = 128
    rng = np.random.default_rng(0)
    for i in range(20):
        node_id = f"v{i}"
        vec = rng.normal(size=(dim,))
        insert(id=node_id, vector=vec, graph=graph, vectors=vectors)

    # After all inserts print the requested diagnostics.
    total_nodes = len(graph.levels)
    entry_point = graph.entry_point
    entry_point_level = graph.levels[entry_point] if entry_point is not None else None

    print(f"Total nodes inserted: {total_nodes}")
    print(f"Entry point: {entry_point}")
    print(f"Entry point level: {entry_point_level}")
    print(f"Neighbors of v0 at layer 0: {len(graph.get_neighbors('v0', 0))}")
    print(f"Neighbors of v1 at layer 0: {len(graph.get_neighbors('v1', 0))}")

