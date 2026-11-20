"""
Generate high-resolution comparison chart for Part 3 L4 (24GB, Ada SM 8.9) Benchmark:
1. W8A8 Dynamic FP8 Linear Quantization (`--quantization fp8`) vs. AWQ 4-bit (`W4A16`) &
   Zero-Oracle SQLite Self-Healing Agent Loop (`0-Shot` -> `+1 Retry` -> `+2 Retries`).
2. The 24GB L4 Concurrency Cliff: Free KV Cache Capacity (`K tokens`) vs. TTFT P50 & TPOT P99 (`c=16`)
   across 7B-AWQ, 7B-FP8 (W8A8), 14B-AWQ, and 32B-AWQ.

Produces:
  `charts/chart_l4_self_healing_and_w8a8.png`
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = [
    ("Qwen2.5-7B-AWQ (W4A16 + FP16 KV)", "7B-AWQ\n(W4A16 + FP16 KV)"),
    ("Qwen2.5-7B-FP8 (W8A8 Linear + FP16 KV)", "7B-FP8 W8A8\n(Linear FP8 + FP16 KV)"),
    ("Qwen2.5-14B-AWQ (W4A16 + FP16 KV)", "14B-AWQ\n(W4A16 + FP16 KV)"),
    ("Qwen2.5-32B-AWQ (W4A16 + FP16 KV, Tight 24GB L4)", "32B-AWQ\n(Tight 24GB L4, util=0.92)"),
]


def plot_part3_suite(suite_json: str, out_dir: str) -> Path:
    payload = json.loads(Path(suite_json).read_text(encoding="utf-8"))
    rmap = {r["label"]: r for r in payload["results"]}

    rows = [rmap[k] for k, _ in MODEL_ORDER]
    x_labels = [short for _, short in MODEL_ORDER]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 6.5), dpi=160)
    fig.suptitle(
        "Single NVIDIA L4 (24GB, SM 8.9) Production Trade-Offs (c=16, 50-Task SQLite Benchmark)\n"
        "Zero-Oracle Self-Healing SQL Loop & W8A8 FP8 Rescue vs. The 32B-AWQ KV Cache Cliff",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )

    x = np.arange(len(x_labels))
    w = 0.25

    # --- Left Panel: 0-Shot vs Self-Healing Execution Accuracy & Final Syntax Validity ---
    acc_0 = [r["turn0_0shot"]["exec_accuracy_pct"] for r in rows]
    acc_1 = [r["turn1_1retry"]["exec_accuracy_pct"] for r in rows]
    syn_2 = [r["turn2_2retries"]["syntax_valid_pct"] for r in rows]

    b0 = ax1.bar(x - w, acc_0, w, label="0-Shot Exec Accuracy (%)", color="#3498db", alpha=0.88)
    b1 = ax1.bar(x, acc_1, w, label="+1 Self-Healing Retry Exec Accuracy (%)", color="#2ecc71", alpha=0.92)
    b2 = ax1.bar(x + w, syn_2, w, label="Post-Healing Valid SQL Syntax (%)", color="#7f8c8d", alpha=0.82)

    ax1.set_title("Zero-Oracle Self-Healing Accuracy & W8A8 FP8 Rescue", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=9.2)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_ylim(0, 130)
    ax1.legend(loc="upper left", fontsize=8.4, framealpha=0.92)

    for b in b0:
        h = b.get_height()
        ax1.annotate(
            f"{h:.0f}%",
            (b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
            color="#1f618d",
        )
    for b, a0 in zip(b1, acc_0):
        h = b.get_height()
        gain = h - a0
        ax1.annotate(
            f"{h:.0f}%\n(+{gain:.0f}%)",
            (b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.4,
            fontweight="bold",
            color="#1e8449",
        )
    for b in b2:
        h = b.get_height()
        ax1.annotate(
            f"{h:.0f}%",
            (b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.4,
            fontweight="bold",
            color="#4d5656",
        )

    ax1.annotate(
        "W8A8 Linear FP8 (--quantization fp8, FP16 KV)\nhits 68% -> 74% accuracy (2.64x KV pool vs BF16),\nwhereas uncalibrated FP8 KV collapsed to 0%!",
        xy=(1.0, 84),
        xytext=(1.85, 111),
        ha="center",
        fontsize=8.2,
        fontweight="bold",
        color="#145a32",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#d4efdf", edgecolor="#27ae60", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.4),
    )

    # --- Right Panel: Latency (TTFT P50 & TPOT P99) vs Free KV Cache Pool on 24GB L4 ---
    ttft_0 = [r["turn0_0shot"]["turn0_ttft_p50_ms"] for r in rows]
    ttft_retry = [r["turn1_1retry"]["retry_ttft_p50_ms"] for r in rows]
    tpot_p99 = [r["turn0_0shot"]["tpot_p99_ms"] for r in rows]
    kv_k = [(r.get("kv_cache_tokens") or 0) / 1000.0 for r in rows]
    max_conc = [r.get("max_conc_4k") or 0.0 for r in rows]

    w2 = 0.26
    b_t0 = ax2.bar(x - w2, ttft_0, w2, label="Turn-0 TTFT P50 (ms)", color="#e67e22", alpha=0.88)
    b_tr = ax2.bar(x, ttft_retry, w2, label="Retry TTFT P50 (ms) [Prefix Cached]", color="#f1c40f", alpha=0.90)
    b_tp = ax2.bar(x + w2, tpot_p99, w2, label="Decode TPOT P99 (ms) [SLA < 45ms]", color="#c0392b", alpha=0.88)

    ax2.axhline(45.0, color="#c0392b", linestyle="--", linewidth=1.3, label="Interactive TPOT SLA (45ms)")
    ax2.set_title("24GB L4 Concurrency Cliff: Latency vs. Available KV Pool (c=16)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=9.2)
    ax2.set_ylabel("Latency (ms)")
    ax2.set_ylim(0, max(ttft_retry) * 1.25)
    ax2.legend(loc="upper left", fontsize=8.4)

    for b in b_t0:
        h = b.get_height()
        ax2.annotate(
            f"{h:.0f}ms",
            (b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color="#a04000",
        )
    for b in b_tr:
        h = b.get_height()
        ax2.annotate(
            f"{h:.0f}ms",
            (b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color="#7d6608",
        )
    for b, kv_val, mc in zip(b_tp, kv_k, max_conc):
        h = b.get_height()
        ax2.annotate(
            f"{h:.1f}ms\n({kv_val:.1f}K KV / {mc:.1f}x)",
            (b.get_x() + b.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color="#922b21",
        )

    ax2.annotate(
        "32B-AWQ leaves only 4.6K KV tokens (1.12x @ 4k) on 24GB L4!\nAt c=16, queuing & preemption spike TTFT 7.2x (814ms)\nand push TPOT P99 to 120.9ms (2.7x over 45ms SLA).",
        xy=(3.0, 814),
        xytext=(1.75, 720),
        ha="center",
        fontsize=8.3,
        fontweight="bold",
        color="#922b21",
        bbox=dict(boxstyle="round,pad=0.36", facecolor="#fadbd8", edgecolor="#c0392b", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = out / "chart_l4_self_healing_and_w8a8.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved chart: {out_path}")
    return out_path


if __name__ == "__main__":
    import sys

    s_json = sys.argv[1] if len(sys.argv) > 1 else "research-vllm-benchmark/results_l4_part3_suite.json"
    out_d = sys.argv[2] if len(sys.argv) > 2 else "research-vllm-benchmark/charts"
    plot_part3_suite(s_json, out_d)
