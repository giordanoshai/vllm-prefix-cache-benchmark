"""
Generate high-resolution dual-panel chart for Part 4 NVIDIA L4 (24GB, SM 8.9) Benchmark:
Speculative Decoding Concurrency Crossover Curve (`c = 1, 2, 4, 8, 16, 32` on `Qwen2.5-7B-Instruct-AWQ`).

Produces:
  `charts/chart_l4_speculative_decoding_crossover.png`
  - Left Panel: Decode Latency Crossover (`TPOT P50` & `TPOT P99` vs. Concurrency `c = 1..32`), highlighting
    the Roofline transition from Memory-Bandwidth-Bound (`-43.5%` TPOT at `c=1..4`) to Compute-Bound
    (`+50%` P50 / `+85.5%` P99 latency degradation at `c=32`).
  - Right Panel: System Output Throughput (`tok/s`) & Speedup Multiplier vs. Standard Autoregressive Decode,
    showing peak `+82.2%` speedup at `c=4` collapsing to `-14.1%` net throughput loss at `c=32` (`k=5`).
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_spec_crossover(json_path: str, out_dir: str) -> Path:
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    rmap = {r["short_id"]: r for r in payload["results"]}

    base_pts = rmap["baseline_autoreg"]["sweep"]
    k3_pts = rmap["spec_ngram_k3"]["sweep"]
    k5_pts = rmap["spec_ngram_k5"]["sweep"]

    concs = [p["concurrency"] for p in base_pts]
    x_idx = np.arange(len(concs))
    x_labels = [f"c={c}" for c in concs]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 6.5), dpi=160)
    fig.suptitle(
        "The Speculative Decoding Concurrency Crossover on NVIDIA L4 (24GB, SM 8.9, Qwen2.5-7B-AWQ)\n"
        "Why Prompt-Lookup Speculative Decoding Speeds Up c=1..8 by +82% — Then Reverses and Hurts Latency at c=16..32",
        fontsize=12.3,
        fontweight="bold",
        y=0.98,
    )

    # --- Left Panel: TPOT P50 (solid) & TPOT P99 (dashed) Crossover ---
    base_p50 = [p["tpot_p50_ms"] for p in base_pts]
    base_p99 = [p["tpot_p99_ms"] for p in base_pts]
    k3_p50 = [p["tpot_p50_ms"] for p in k3_pts]
    k3_p99 = [p["tpot_p99_ms"] for p in k3_pts]
    k5_p50 = [p["tpot_p50_ms"] for p in k5_pts]
    k5_p99 = [p["tpot_p99_ms"] for p in k5_pts]

    ax1.plot(x_idx, base_p50, marker="o", linewidth=2.5, color="#2980b9", label="Standard Decode TPOT P50 (1.00 tok/step)")
    ax1.plot(x_idx, base_p99, marker="o", linewidth=1.5, linestyle=":", color="#2980b9", alpha=0.75, label="Standard Decode TPOT P99")

    ax1.plot(x_idx, k3_p50, marker="s", linewidth=2.5, color="#27ae60", label="Speculative N-gram (k=3) TPOT P50 (~1.62 tok/step)")
    ax1.plot(x_idx, k3_p99, marker="s", linewidth=1.5, linestyle=":", color="#27ae60", alpha=0.75, label="Speculative N-gram (k=3) TPOT P99")

    ax1.plot(x_idx, k5_p50, marker="^", linewidth=2.5, color="#e67e22", label="Speculative N-gram (k=5) TPOT P50 (~1.70 tok/step)")
    ax1.plot(x_idx, k5_p99, marker="^", linewidth=1.6, linestyle="--", color="#c0392b", alpha=0.88, label="Speculative N-gram (k=5) TPOT P99")

    ax1.axhline(45.0, color="#c0392b", linestyle="-.", linewidth=1.3, alpha=0.8, label="Interactive TPOT SLA (45ms)")

    ax1.set_title("Per-Token Decode Latency Crossover (TPOT P50 & P99)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(x_labels, fontsize=9.8, fontweight="bold")
    ax1.set_ylabel("Time Per Output Token — TPOT (ms/tok) [Lower is Better]")
    ax1.set_ylim(10, 76)
    ax1.legend(loc="upper left", fontsize=8.0, framealpha=0.93)

    # Annotate key points on Left Panel
    for i in (0, 2, 4, 5):
        ax1.annotate(
            f"{base_p50[i]:.1f}ms",
            (x_idx[i], base_p50[i]),
            xytext=(0 if i != 4 else -18, 7 if i < 5 else -15),
            textcoords="offset points",
            ha="center",
            fontsize=8.2,
            fontweight="bold",
            color="#1f618d",
        )
        ax1.annotate(
            f"{k5_p50[i]:.1f}ms",
            (x_idx[i], k5_p50[i]),
            xytext=(0 if i != 4 else 18, -14 if i < 4 else (9 if i == 5 else -12)),
            textcoords="offset points",
            ha="center",
            fontsize=8.2,
            fontweight="bold",
            color="#a04000",
        )

    ax1.annotate(
        f"P99={k5_p99[5]:.1f}ms\n(+85.5% worse!)",
        (x_idx[5], k5_p99[5]),
        xytext=(-44, 4),
        textcoords="offset points",
        ha="center",
        fontsize=8.3,
        fontweight="bold",
        color="#922b21",
    )

    ax1.annotate(
        "Memory-BW Bound (c=1..8):\nSpeculative (k=5) cuts TPOT\nby 43.5% (31.0 -> 17.5ms)",
        xy=(1.0, 17.6),
        xytext=(1.15, 4.5),
        ha="center",
        fontsize=8.2,
        fontweight="bold",
        color="#145a32",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#d4efdf", edgecolor="#27ae60", alpha=0.95),
    )

    ax1.annotate(
        "Crossover Inflection (c* ≈ 16–22):\nAt c=32, k=5 verifies 32×6 = 192 tokens/step\n(Compute-Bound!), degrading TPOT P50 by +50%\n(49.3ms vs 32.9ms) and breaching 45ms SLA.",
        xy=(4.0, 32.5),
        xytext=(3.2, 54.0),
        ha="center",
        fontsize=8.0,
        fontweight="bold",
        color="#922b21",
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#fadbd8", edgecolor="#c0392b", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4),
    )
    ax1.set_ylim(0, 78)

    # --- Right Panel: Total System Throughput (tok/s) & Relative Speedup vs Standard Decode ---
    base_tp = [p["throughput_tok_per_s"] for p in base_pts]
    k3_tp = [p["throughput_tok_per_s"] for p in k3_pts]
    k5_tp = [p["throughput_tok_per_s"] for p in k5_pts]

    w = 0.26
    b_base = ax2.bar(x_idx - w, base_tp, w, label="Standard Autoregressive Decode (Baseline)", color="#2980b9", alpha=0.88)
    b_k3 = ax2.bar(x_idx, k3_tp, w, label="Speculative N-gram (k=3, ~1.61 tok/step)", color="#27ae60", alpha=0.90)
    b_k5 = ax2.bar(x_idx + w, k5_tp, w, label="Speculative N-gram (k=5, ~1.69 tok/step)", color="#e67e22", alpha=0.90)

    ax2.set_title("System Throughput (tok/s) & Speedup Collapse at High Concurrency", fontsize=11, fontweight="bold")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(x_labels, fontsize=9.8, fontweight="bold")
    ax2.set_ylabel("Total Output Throughput (tokens/sec) [Higher is Better]")
    ax2.set_ylim(0, 760)
    ax2.legend(loc="upper left", fontsize=8.4, framealpha=0.93)

    for b in b_base:
        h = b.get_height()
        ax2.annotate(
            f"{h:.0f}",
            (b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.9,
            fontweight="bold",
            color="#1b4f72",
        )
    for b, bt in zip(b_k3, base_tp):
        h = b.get_height()
        pct = 100.0 * (h - bt) / bt
        sign = "+" if pct >= 0 else ""
        ax2.annotate(
            f"{h:.0f}\n({sign}{pct:.0f}%)",
            (b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
            color="#1e8449",
        )
    for b, bt in zip(b_k5, base_tp):
        h = b.get_height()
        pct = 100.0 * (h - bt) / bt
        sign = "+" if pct >= 0 else ""
        clr = "#a04000" if pct >= 0 else "#c0392b"
        ax2.annotate(
            f"{h:.0f}\n({sign}{pct:.0f}%)",
            (b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
            color=clr,
        )

    ax2.annotate(
        "Peak Speedup at c=4..8:\n+82.2% (209 vs 115 tok/s at c=4)\n+67.1% (366 vs 219 tok/s at c=8)",
        xy=(2.26, 235),
        xytext=(1.55, 420),
        ha="center",
        fontsize=8.2,
        fontweight="bold",
        color="#145a32",
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#d4efdf", edgecolor="#27ae60", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.4),
    )

    ax2.annotate(
        "At c=32, k=5 throughput drops -14.1%\nbelow Standard Decode (511 vs 594 tok/s)\neven though acceptance stays 1.67 tok/step!\n(72% wasted verification FLOPs saturate L4)",
        xy=(5.26, 560),
        xytext=(3.2, 655),
        ha="center",
        fontsize=8.1,
        fontweight="bold",
        color="#922b21",
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#fadbd8", edgecolor="#c0392b", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = out / "chart_l4_speculative_decoding_crossover.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved chart: {out_path}")
    return out_path


if __name__ == "__main__":
    import sys

    j_path = sys.argv[1] if len(sys.argv) > 1 else "research-vllm-benchmark/results_l4_spec_decoding.json"
    out_d = sys.argv[2] if len(sys.argv) > 2 else "research-vllm-benchmark/charts"
    plot_spec_crossover(j_path, out_d)
