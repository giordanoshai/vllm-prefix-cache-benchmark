"""
Generate high-resolution comparison chart for vLLM Quantization vs. SQL Execution Accuracy.

Produces:
`charts/chart_accuracy_vs_quantization.png`
  - Left Panel: SQLite Execution Accuracy (%) & Syntax Validity (%) across tested quantization & KV modes
  - Right Panel: Output Throughput (tok/s) & Allocated KV Cache Pool Size (K Tokens)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


PREFERRED_ORDER = [
    ("BF16 + FP16 KV", "BF16\n+ FP16 KV\n(Baseline)"),
    ("AWQ 4-bit + FP16 KV", "AWQ 4-bit\n+ FP16 KV"),
    ("BF16 + FP8 KV (FlashInfer Default)", "BF16 + FP8 KV\n(FlashInfer)"),
    ("BF16 + FP8 KV (TRITON_ATTN)", "BF16 + FP8 KV\n(Triton Attn)"),
    ("BF16 + FP8_PER_TOKEN_HEAD KV", "BF16 + FP8\nPer-Token-Head"),
    ("AWQ 4-bit + FP8 KV (FlashInfer Default)", "AWQ 4-bit\n+ FP8 KV\n(FlashInfer)"),
]


def plot_ablation(json_path: str, out_dir: str) -> Path:
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    by_label = {r["label"]: r for r in payload.get("results", []) if "exec_accuracy_pct" in r}

    rows = []
    labels = []
    for raw_label, short_label in PREFERRED_ORDER:
        if raw_label in by_label:
            rows.append(by_label[raw_label])
            labels.append(short_label)

    if not rows:
        raise ValueError(f"No valid benchmark entries found in {json_path}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn_v0_8-darkgrid" in plt.style.available else "default")

    exec_acc = [r["exec_accuracy_pct"] for r in rows]
    syntax_pct = [r["syntax_valid_pct"] for r in rows]
    tputs = [r["throughput_tok_per_s"] for r in rows]
    kv_tokens_k = [(r.get("kv_cache_tokens") or 0) / 1000.0 for r in rows]

    gpu_name = payload.get("gpu", "Colab GPU")
    num_tasks = payload.get("num_tasks", 50)
    conc = payload.get("concurrency", 16)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2), dpi=160)
    fig.suptitle(
        f"Quantization Speed vs. Output Quality: {num_tasks}-Task SQLite Execution Benchmark (c={conc})\n"
        f"vLLM 0.34.0 on {gpu_name} (SM 12.0) — Fastest FP8 KV Setup Returned HTTP 200 With 0% Valid SQL",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )

    x = np.arange(len(labels))
    w = 0.36

    # Left panel: Execution Accuracy vs Syntax Validity
    b_syn = ax1.bar(x - w / 2, syntax_pct, w, label="SQL Syntax Valid (%)", color="#7f8c8d", alpha=0.85)
    acc_colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in exec_acc]
    b_acc = ax1.bar(x + w / 2, exec_acc, w, label="Exact SQLite Execution Match (%)", color=acc_colors, alpha=0.92)
    ax1.set_title("Output Quality: SQL Syntax & Execution Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.8)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_ylim(0, 108)
    ax1.legend(loc="upper right", fontsize=9)

    for b in b_syn:
        h = b.get_height()
        ax1.annotate(f"{h:.0f}%", (b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.3, color="#555555", fontweight="bold")
    for b in b_acc:
        h = b.get_height()
        ax1.annotate(f"{h:.0f}%", (b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.8, fontweight="bold",
                     color="#1e8449" if h > 0 else "#c0392b")

    # Callout on FP8 silent corruption
    if len(x) >= 4:
        ax1.annotate(
            "Silent Corruption across ALL FP8 KV modes\n(FlashInfer, Triton, & Dynamic Per-Token-Head)",
            xy=(3.5, 4),
            xytext=(3.5, 42),
            ha="center",
            fontsize=8.8,
            fontweight="bold",
            color="#922b21",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fadbd8", edgecolor="#c0392b", alpha=0.92),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4),
        )

    # Right panel: Output Throughput (bars) + KV Cache Token Pool (line on twin axis)
    bar_colors = ["#2980b9" if v > 0 else "#e67e22" for v in exec_acc]
    b_tput = ax2.bar(x, tputs, width=0.50, label="Output Throughput (tok/s)", color=bar_colors, alpha=0.86)
    ax2.set_title("Inference Throughput (tok/s) vs. KV Cache Capacity", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.8)
    ax2.set_ylabel("Output Throughput (tokens/sec)", color="#1f618d")
    ax2.set_ylim(0, max(tputs) * 1.32)

    for b in b_tput:
        h = b.get_height()
        ax2.annotate(f"{h:,.0f}\ntok/s", (b.get_x() + b.get_width() / 2, 160),
                     ha="center", va="center", fontsize=8.4, fontweight="bold", color="white")

    if any(v > 0 for v in kv_tokens_k):
        ax2_r = ax2.twinx()
        ax2_r.plot(x, kv_tokens_k, marker="D", markersize=6.5, linewidth=2.2, color="#8e44ad", label="KV Cache Pool (K Tokens)")
        ax2_r.set_ylabel("KV Cache Capacity (Thousands of Tokens)", color="#8e44ad")
        ax2_r.set_ylim(0, max(kv_tokens_k) * 1.25)
        for xi, kv_k in zip(x, kv_tokens_k):
            ax2_r.annotate(
                f"{kv_k:,.0f}K",
                (xi, kv_k),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color="#5b2c6f",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#f4ecf7", edgecolor="#8e44ad", lw=0.7, alpha=0.95),
            )
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_r.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8.8)
    else:
        ax2.legend(loc="upper left", fontsize=8.8)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = out / "chart_accuracy_vs_quantization.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved chart: {out_path}")
    return out_path


if __name__ == "__main__":
    import sys
    json_p = sys.argv[1] if len(sys.argv) > 1 else "research-vllm-benchmark/results_sql_accuracy_ablation.json"
    out_d = sys.argv[2] if len(sys.argv) > 2 else "research-vllm-benchmark/charts"
    plot_ablation(json_p, out_d)
