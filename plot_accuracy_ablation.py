"""
Generate high-resolution comparison chart for vLLM Quantization vs. SQL Execution Accuracy
across NVIDIA L4 (SM 8.9) and NVIDIA RTX PRO 6000 Blackwell (SM 12.0).

Produces:
`charts/chart_accuracy_vs_quantization.png`
  - Left Panel: 50-Task SQLite Execution Accuracy & Syntax Validity across 4 core setups (reproduced identically on both L4 and Blackwell)
  - Right Panel: Throughput Speedup & TPOT P99 SLA Comparison (L4 vs Blackwell)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


CORE_ORDER = [
    ("BF16 + FP16 KV", "BF16\n+ FP16 KV"),
    ("AWQ 4-bit + FP16 KV", "AWQ 4-bit\n+ FP16 KV"),
    ("BF16 + FP8 KV", "BF16\n+ FP8 KV"),
    ("AWQ 4-bit + FP8 KV", "AWQ 4-bit\n+ FP8 KV"),
]


def _lookup_map(json_path: Path) -> dict[str, dict]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    out = {}
    for r in payload.get("results", []):
        lbl = r.get("label", "")
        if lbl == "BF16 + FP8 KV (FlashInfer Default)":
            lbl = "BF16 + FP8 KV"
        elif lbl == "AWQ 4-bit + FP8 KV (FlashInfer Default)":
            lbl = "AWQ 4-bit + FP8 KV"
        out[lbl] = r
    return out


def plot_ablation(blackwell_json: str, l4_json: str, out_dir: str) -> Path:
    bw_map = _lookup_map(Path(blackwell_json))
    l4_map = _lookup_map(Path(l4_json))

    labels = [short for _, short in CORE_ORDER]
    bw_rows = [bw_map[raw] for raw, _ in CORE_ORDER]
    l4_rows = [l4_map[raw] for raw, _ in CORE_ORDER]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2), dpi=160)
    fig.suptitle(
        "Quantization Speed vs. Output Quality: 50-Task SQLite Benchmark (Qwen2.5-7B-Instruct, c=16)\n"
        "Reproduced Identically on Both NVIDIA L4 (SM 8.9) & RTX PRO 6000 Blackwell (SM 12.0)",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )

    x = np.arange(len(labels))
    w = 0.34

    # Left Panel: Output Quality (identical across both GPUs!)
    syn_l4 = [r["syntax_valid_pct"] for r in l4_rows]
    acc_l4 = [r["exec_accuracy_pct"] for r in l4_rows]

    b_syn = ax1.bar(x - w / 2, syn_l4, w, label="SQL Syntax Valid (%) [L4 & Blackwell]", color="#7f8c8d", alpha=0.86)
    acc_colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in acc_l4]
    b_acc = ax1.bar(x + w / 2, acc_l4, w, label="Exact SQLite Execution Match (%) [L4 & Blackwell]", color=acc_colors, alpha=0.92)

    ax1.set_title("Output Quality: Syntax & Execution Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_ylim(0, 108)
    ax1.legend(loc="upper right", fontsize=8.8)

    for b in b_syn:
        h = b.get_height()
        ax1.annotate(f"{h:.0f}%", (b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, color="#444444", fontweight="bold")
    for b in b_acc:
        h = b.get_height()
        ax1.annotate(f"{h:.0f}%", (b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                     color="#1e8449" if h > 0 else "#c0392b")

    ax1.annotate(
        "Uncalibrated FP8 KV collapsed to 0% valid SQL\non BOTH L4 (SM 8.9) and Blackwell (SM 12.0)\n(also failed under Triton Attn & fp8_per_token_head)",
        xy=(2.5, 4),
        xytext=(2.5, 44),
        ha="center",
        fontsize=8.8,
        fontweight="bold",
        color="#922b21",
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#fadbd8", edgecolor="#c0392b", alpha=0.94),
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4),
    )

    # Right Panel: Throughput on L4 vs Blackwell (dual subplot or normalized / paired bars with clear labels)
    # Using two y-axes so L4 (205-350 tok/s) and Blackwell (942-1448 tok/s) are both legible!
    tput_l4 = [r["throughput_tok_per_s"] for r in l4_rows]
    tput_bw = [r["throughput_tok_per_s"] for r in bw_rows]
    kv_l4_k = [(r.get("kv_cache_tokens") or 0) / 1000.0 for r in l4_rows]

    b_bw = ax2.bar(x - w / 2, tput_bw, w, label="Blackwell SM 12.0 Throughput (tok/s)", color="#2980b9", alpha=0.88)
    ax2.set_ylabel("Blackwell Output Throughput (tok/s)", color="#1f618d")
    ax2.set_ylim(0, max(tput_bw) * 1.28)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_title("Throughput Speedup & L4 KV Capacity (tok/s)", fontsize=11, fontweight="bold")

    ax2_r = ax2.twinx()
    b_l4 = ax2_r.bar(x + w / 2, tput_l4, w, label="NVIDIA L4 SM 8.9 Throughput (tok/s)", color="#f39c12", alpha=0.90)
    ax2_r.set_ylabel("NVIDIA L4 Output Throughput (tok/s)", color="#b9770e")
    ax2_r.set_ylim(0, max(tput_l4) * 1.28)

    for b in b_bw:
        h = b.get_height()
        ax2.annotate(f"{h:,.0f}", (b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#1b4f72")
    for b, kv_k in zip(b_l4, kv_l4_k):
        h = b.get_height()
        ax2_r.annotate(f"{h:.0f}\n({kv_k:.0f}K KV)", (b.get_x() + b.get_width() / 2, h),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#9c640c")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8.8)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = out / "chart_accuracy_vs_quantization.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved chart: {out_path}")
    return out_path


if __name__ == "__main__":
    import sys
    bw_p = sys.argv[1] if len(sys.argv) > 1 else "research-vllm-benchmark/results_sql_accuracy_ablation.json"
    l4_p = sys.argv[2] if len(sys.argv) > 2 else "research-vllm-benchmark/results_sql_accuracy_l4.json"
    out_d = sys.argv[3] if len(sys.argv) > 3 else "research-vllm-benchmark/charts"
    plot_ablation(bw_p, l4_p, out_d)
