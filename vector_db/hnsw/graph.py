import math
import random
from typing import Dict, List, Optional, Any


class HNSWGraph:
    """
    Stores the layer structure of an HNSW graph (neighbors per layer).

    This class only manages graph topology data structures (layers and
    neighbor lists). It does not implement insertion/linking heuristics
    or search.
    """

    def __init__(self, M: int = 16, ef_construction: int = 200, max_level: Optional[int] = None):
        """
        Initialize the HNSW graph.

        Args:
            M: Max connections per node per layer (must be > 1).
            ef_construction: Search width during build (stored for later use).
            max_level: Maximum number of layers minus one. If None, computed
                automatically as an upper bound for the level distribution.
        """

        self.M = int(M)
        self.ef_construction = int(ef_construction)

        if self.ef_construction <= 0:
            raise ValueError(
                f"`ef_construction` must be a positive integer, got {self.ef_construction}"
            )

        if self.M <= 1:
            raise ValueError(f"`M` must be > 1, got {self.M}")

        # Compute max_level automatically if not provided.
        # We use the distribution implied by:
        #   level = floor(-ln(U) * (1 / ln(M))) where U ~ Uniform(0, 1).
        # From that, P(level >= L) <= M^{-L}. Choose L such that M^{-L} < 1e-9.
        if max_level is None:
            tail_prob = 1e-9
            # Solve M^{-L} < tail_prob => L > -ln(tail_prob)/ln(M)
            computed = math.ceil(-math.log(tail_prob) / math.log(self.M))
            self.max_level = max(0, int(computed))
        else:
            self.max_level = int(max_level)
            if self.max_level < 0:
                raise ValueError(f"`max_level` must be >= 0, got {self.max_level}")

        # layers[L] is a dict: node_id -> list_of_neighbor_ids
        self.layers: List[Dict[Any, List[Any]]] = [dict() for _ in range(self.max_level + 1)]

        # entry_point is the node id with the highest assigned level so far.
        self.entry_point: Optional[Any] = None

        # levels maps node_id -> assigned level (integer)
        self.levels: Dict[Any, int] = {}

    def get_random_level(self) -> int:
        """
        Assign a random level using the requested geometric distribution formula:

            floor(-ln(random()) * (1 / ln(M)))

        Returns:
            An integer level capped to [0, max_level].
        """

        # Implement the corrected formula exactly as requested by the user:
        #   level = int(-ln(random()) * (1 / ln(M)))
        #   return max(0, level)
        # Clamp the sampled level to the allocated graph layers. A sample
        # outside this range would otherwise disagree with add_node(), which
        # caps levels when creating adjacency lists.
        sample = max(random.random(), 1e-300)
        level = int(-math.log(sample) * (1.0 / math.log(self.M)))
        return min(self.max_level, max(0, level))

    def add_node(self, id: Any, level: int) -> None:
        """
        Add a node to the graph layers from 0 up to `level` (inclusive).

        Args:
            id: Node identifier (string/int/any hashable).
            level: Assigned node level. Nodes appear in layers [0..level].
        """

        lvl = int(level)
        if lvl < 0:
            raise ValueError(f"`level` must be >= 0, got {lvl}")
        lvl = min(lvl, self.max_level)

        # Initialize empty neighbor list for each layer the node belongs to.
        for layer in range(lvl + 1):
            self.layers[layer][id] = []

        # Record assigned level.
        self.levels[id] = lvl

        # Update entry point if this node has a higher level.
        if self.entry_point is None or lvl > self.levels[self.entry_point]:
            self.entry_point = id

    def get_neighbors(self, id: Any, layer: int) -> List[Any]:
        """
        Get the neighbor list for a given node at a given layer.

        Args:
            id: Node identifier.
            layer: Layer index.

        Returns:
            A list of neighbor IDs.
        """

        layer_idx = int(layer)
        if layer_idx < 0 or layer_idx > self.max_level:
            raise ValueError(f"`layer` out of range: {layer_idx} (max {self.max_level})")

        if id not in self.layers[layer_idx]:
            raise KeyError(f"Node id={id!r} not found in layer={layer_idx}")

        return self.layers[layer_idx][id]

    def set_neighbors(self, id: Any, layer: int, neighbors: List[Any]) -> None:
        """
        Set the neighbor list for a node at a given layer.

        Args:
            id: Node identifier.
            layer: Layer index.
            neighbors: List of neighbor IDs to store.
        """

        layer_idx = int(layer)
        if layer_idx < 0 or layer_idx > self.max_level:
            raise ValueError(f"`layer` out of range: {layer_idx} (max {self.max_level})")

        if id not in self.layers[layer_idx]:
            raise KeyError(f"Node id={id!r} not found in layer={layer_idx}")

        self.layers[layer_idx][id] = list(neighbors)


if __name__ == "__main__":
    # Create HNSWGraph M=16
    graph = HNSWGraph(M=16)

    # Add 20 nodes with randomly assigned levels.
    for i in range(20):
        node_id = f"v{i}"
        level = graph.get_random_level()
        graph.add_node(node_id, level)

    # Print each node's assigned level.
    print("Assigned levels:")
    for node_id in sorted(graph.levels.keys()):
        print(f"{node_id}: level={graph.levels[node_id]}")

    # Summarize the level distribution to verify the pyramid shape.
    level_values = list(graph.levels.values())
    level_0_count = sum(1 for lv in level_values if lv == 0)
    level_1_count = sum(1 for lv in level_values if lv == 1)
    level_2_plus_count = sum(1 for lv in level_values if lv >= 2)

    print(f"\nLevel 0 nodes: {level_0_count}")
    print(f"Level 1 nodes: {level_1_count}")
    print(f"Level 2+ nodes: {level_2_plus_count}")

