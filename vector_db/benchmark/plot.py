import os

import matplotlib

# Use a non-interactive backend for headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vector_db.benchmark.run import run_benchmark


def _extract_series(results):
    """
    Extract plotting series from run_benchmark output.

    Returns:
        ef_vals, recall_vals, hnsw_latency_vals, brute_latency_vals
    """

    ef_vals = [r["ef_search"] for r in results]
    recall_vals = [r["recall"] for r in results]
    hnsw_latency_vals = [r["hnsw_latency_ms"] for r in results]
    brute_latency_vals = [r["brute_latency_ms"] for r in results]
    return ef_vals, recall_vals, hnsw_latency_vals, brute_latency_vals


def plot_recall_vs_efsearch(results, out_path):
    """
    Plot recall@k vs ef_search with a horizontal target line at y=0.9.
    """

    ef_vals, recall_vals, _, _ = _extract_series(results)

    plt.figure(figsize=(8, 5))
    plt.plot(ef_vals, recall_vals, marker="o")
    plt.axhline(y=0.9, color="red", linestyle="--", linewidth=1.5, label="target")

    plt.title("Recall vs efSearch")
    plt.xlabel("ef_search")
    plt.ylabel("recall@k")
    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_latency_vs_efsearch(results, out_path):
    """
    Plot hnsw_ms and brute_ms latency vs ef_search.
    """

    ef_vals, _, hnsw_latency_vals, brute_latency_vals = _extract_series(results)

    plt.figure(figsize=(8, 5))
    plt.plot(ef_vals, hnsw_latency_vals, marker="o", label="hnsw_ms")
    plt.plot(ef_vals, brute_latency_vals, marker="o", label="brute_ms")

    plt.title("Latency vs efSearch")
    plt.xlabel("ef_search")
    plt.ylabel("latency (ms)")
    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_hnsw_vs_brute(results, out_path, ef_target=100):
    """
    Bar chart comparing hnsw vs brute latency at a specific ef_search value.
    """

    ef_vals, _, hnsw_latency_vals, brute_latency_vals = _extract_series(results)

    # Find the row where ef_search == ef_target.
    idx = None
    for i, v in enumerate(ef_vals):
        if v == ef_target:
            idx = i
            break

    if idx is None:
        # If ef_target isn't present, fall back to the closest ef_search.
        # (This keeps the script robust even if defaults change.)
        idx = min(range(len(ef_vals)), key=lambda i: abs(ef_vals[i] - ef_target))

    hnsw_ms = hnsw_latency_vals[idx]
    brute_ms = brute_latency_vals[idx]

    plt.figure(figsize=(6, 5))
    labels = ["hnsw", "brute"]
    values = [hnsw_ms, brute_ms]
    plt.bar(labels, values, color=["steelblue", "gray"])

    plt.title("HNSW vs Brute Force Latency")
    plt.xlabel(f"ef_search = {ef_vals[idx]}")
    plt.ylabel("latency (ms)")
    plt.grid(True, axis="y", linestyle=":", linewidth=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    results = run_benchmark()

    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)

    plot_recall_vs_efsearch(
        results=results, out_path=os.path.join(out_dir, "recall_vs_efsearch.png")
    )
    plot_latency_vs_efsearch(
        results=results, out_path=os.path.join(out_dir, "latency_vs_efsearch.png")
    )
    plot_hnsw_vs_brute(
        results=results,
        out_path=os.path.join(out_dir, "hnsw_vs_brute.png"),
        ef_target=100,
    )

    print("Plots saved")

