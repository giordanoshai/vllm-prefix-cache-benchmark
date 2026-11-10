"""
Generate publication-ready benchmark charts for X (Twitter) and LinkedIn (`plot_results.py`).

Produces three high-resolution charts:
1. `chart_prefix_caching_ttft.png`: TTFT P99 & Throughput (0% Cache Hit vs 100% Prefix Cache Hit on 8GB RTX 3070)
2. `chart_concurrency_inflection.png`: Prefill-Decode Interference (TPOT P99 & E2E P99 across c=1..32)
3. `chart_3070_vs_4090_fp8_scaling.png`: Cross-GPU Scaling (RTX 3070 8GB vs RTX 4090 24GB FP16 vs FP8 KV Cache up to c=64)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def generate_charts(shared_json_path: str, unique_json_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(shared_json_path, "r", encoding="utf-8") as f:
        shared_data = json.load(f)
    with open(unique_json_path, "r", encoding="utf-8") as f:
        unique_data = json.load(f)

    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    # -------------------------------------------------------------------------
    # Chart 1: Prefix Caching Impact on TTFT P99 & Throughput across Concurrency
    # -------------------------------------------------------------------------
    common_concs = [
        d["concurrency"]
        for d in unique_data
        if any(s["concurrency"] == d["concurrency"] for s in shared_data)
    ]
    u_by_c = {d["concurrency"]: d for d in unique_data}
    s_by_c = {d["concurrency"]: d for d in shared_data}

    x = np.arange(len(common_concs))
    width = 0.36

    u_ttft_p99 = [u_by_c[c]["ttft_p99_ms"] for c in common_concs]
    s_ttft_p99 = [s_by_c[c]["ttft_p99_ms"] for c in common_concs]
    u_tput = [u_by_c[c]["throughput_tok_per_s"] for c in common_concs]
    s_tput = [s_by_c[c]["throughput_tok_per_s"] for c in common_concs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=160)
    fig.suptitle(
        "vLLM Automatic Prefix Caching Benchmark (~1,800-Token Enterprise Schema Prefix)\n"
        "Model: Qwen2.5-1.5B-Instruct | GPU: NVIDIA RTX 3070 (8GB VRAM, 1.82 GiB KV Cache Pool)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    bars1 = ax1.bar(x - width / 2, u_ttft_p99, width, label="Unique Prefix (0% Cache Hit)", color="#e74c3c", alpha=0.88)
    bars2 = ax1.bar(x + width / 2, s_ttft_p99, width, label="Shared Prefix (Prefix Cache Hit)", color="#2ecc71", alpha=0.88)
    ax1.set_title("TTFT P99 Latency (ms) — Lower is Better", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"c={c}" for c in common_concs])
    ax1.set_xlabel("Concurrent Requests")
    ax1.set_ylabel("TTFT P99 (ms)")
    ax1.legend()

    for b in bars1:
        ax1.annotate(f"{b.get_height():.0f}ms", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for b in bars2:
        ax1.annotate(f"{b.get_height():.0f}ms", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    bars3 = ax2.bar(x - width / 2, u_tput, width, label="Unique Prefix (0% Cache Hit)", color="#e67e22", alpha=0.88)
    bars4 = ax2.bar(x + width / 2, s_tput, width, label="Shared Prefix (Prefix Cache Hit)", color="#3498db", alpha=0.88)
    ax2.set_title("System Output Throughput (tokens/sec) — Higher is Better", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"c={c}" for c in common_concs])
    ax2.set_xlabel("Concurrent Requests")
    ax2.set_ylabel("Output Throughput (tokens/s)")
    ax2.legend()

    for b in bars3:
        ax2.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", fontsize=8.5)
    for b in bars4:
        ax2.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    chart1_path = out / "chart_prefix_caching_ttft.png"
    plt.savefig(chart1_path)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Chart 2: Prefill-Decode Interference & Tail Latency Inflection Curve
    # -------------------------------------------------------------------------
    u_tpot_p99 = [u_by_c[c]["tpot_p99_ms"] for c in common_concs]
    s_tpot_p99 = [s_by_c[c]["tpot_p99_ms"] for c in common_concs]
    u_e2e_p99 = [u_by_c[c]["e2e_p99_ms"] for c in common_concs]
    s_e2e_p99 = [s_by_c[c]["e2e_p99_ms"] for c in common_concs]

    fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=160)
    fig2.suptitle(
        "The Hidden Cost of Uncached Prefills: Prefill-Decode Interference & Queue Saturation\n"
        "Why Decode Speed (TPOT) Degrades 2.8x Under High Concurrency Without Prefix Caching",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )

    ax_l.plot(common_concs, u_tpot_p99, marker="o", linewidth=2.5, color="#e74c3c", label="Unique Prefix (0% Cache Hit) — TPOT P99")
    ax_l.plot(common_concs, s_tpot_p99, marker="s", linewidth=2.5, color="#2ecc71", label="Shared Prefix (Cache Hit) — TPOT P99")
    ax_l.axhline(45.0, color="#7f8c8d", linestyle="--", linewidth=1.2, label="Interactive SLA Limit (45ms/tok)")
    ax_l.set_title("Decode Interference: TPOT P99 (ms/token)", fontsize=11, fontweight="bold")
    ax_l.set_xlabel("Concurrency Level (Active Requests)")
    ax_l.set_ylabel("TPOT P99 (ms per output token)")
    ax_l.set_xticks(common_concs)
    ax_l.legend()
    for c_val, y_u, y_s in zip(common_concs, u_tpot_p99, s_tpot_p99):
        ax_l.annotate(f"{y_u:.1f}ms", (c_val, y_u), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, color="#c0392b", fontweight="bold")
        ax_l.annotate(f"{y_s:.1f}ms", (c_val, y_s), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8.5, color="#27ae60", fontweight="bold")

    ax_r.plot(common_concs, u_e2e_p99, marker="o", linewidth=2.5, color="#8e44ad", label="Unique Prefix (0% Cache Hit) — E2E P99")
    ax_r.plot(common_concs, s_e2e_p99, marker="s", linewidth=2.5, color="#2980b9", label="Shared Prefix (Cache Hit) — E2E P99")
    ax_r.set_title("End-to-End Request Latency P99 (ms)", fontsize=11, fontweight="bold")
    ax_r.set_xlabel("Concurrency Level (Active Requests)")
    ax_r.set_ylabel("E2E P99 Latency (ms)")
    ax_r.set_xticks(common_concs)
    ax_r.legend()
    for c_val, y_u, y_s in zip(common_concs, u_e2e_p99, s_e2e_p99):
        ax_r.annotate(f"{y_u:.0f}ms", (c_val, y_u), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, color="#8e44ad", fontweight="bold")
        ax_r.annotate(f"{y_s:.0f}ms", (c_val, y_s), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8.5, color="#2980b9", fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    chart2_path = out / "chart_concurrency_inflection.png"
    plt.savefig(chart2_path)
    plt.close(fig2)

    # -------------------------------------------------------------------------
    # Chart 3: RTX 3070 (8GB) vs RTX 4090 (24GB FP16 vs FP8 KV Cache) Scaling
    # -------------------------------------------------------------------------
    base_dir = Path(shared_json_path).parent
    p4090_u = base_dir / "results_4090_unique.json"
    p4090_s = base_dir / "results_4090_shared.json"
    p4090_fp8 = base_dir / "results_4090_fp8_shared.json"

    if p4090_u.exists() and p4090_s.exists() and p4090_fp8.exists():
        d4090_u = {d["concurrency"]: d for d in json.loads(p4090_u.read_text(encoding="utf-8"))}
        d4090_s = {d["concurrency"]: d for d in json.loads(p4090_s.read_text(encoding="utf-8"))}
        d4090_fp8 = {d["concurrency"]: d for d in json.loads(p4090_fp8.read_text(encoding="utf-8"))}

        c4090 = sorted(d4090_s.keys())
        c3070 = sorted(s_by_c.keys())

        fig3, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=160)
        fig3.suptitle(
            "Cross-GPU Production Scaling: RTX 3070 (8GB) vs RTX 4090 (24GB FP16 & FP8 KV Cache)\n"
            "Throughput Scaling up to c=64 & The Dequantization Tradeoff of FP8 KV Cache",
            fontsize=12.5,
            fontweight="bold",
            y=0.98,
        )

        # Left Panel: Output Throughput across GPUs & Modes
        ax_a.plot(c4090, [d4090_s[c]["throughput_tok_per_s"] for c in c4090], marker="s", linewidth=2.6, color="#2ecc71", label="RTX 4090 (FP16 KV + Shared Prefix)")
        ax_a.plot(c4090, [d4090_fp8[c]["throughput_tok_per_s"] for c in c4090], marker="^", linewidth=2.3, linestyle="--", color="#3498db", label="RTX 4090 (FP8 KV + Shared Prefix)")
        ax_a.plot(c4090, [d4090_u[c]["throughput_tok_per_s"] for c in c4090], marker="o", linewidth=2.2, color="#e74c3c", label="RTX 4090 (FP16 KV + Unique Prefix 0% Hit)")
        ax_a.plot(c3070, [s_by_c[c]["throughput_tok_per_s"] for c in c3070], marker="d", linewidth=2.0, linestyle=":", color="#9b59b6", label="RTX 3070 8GB (FP16 KV + Shared Prefix)")

        ax_a.set_title("Output Throughput Scaling (tokens/sec)", fontsize=11, fontweight="bold")
        ax_a.set_xlabel("Concurrency Level (c=1 to c=64)")
        ax_a.set_ylabel("System Output Throughput (tokens/s)")
        ax_a.set_xticks(c4090)
        ax_a.legend(fontsize=8.8)

        for c_val in [16, 32, 64]:
            if c_val in d4090_s:
                v_s = d4090_s[c_val]["throughput_tok_per_s"]
                ax_a.annotate(f"{v_s:.0f}", (c_val, v_s), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8.5, fontweight="bold", color="#27ae60")
            if c_val in d4090_fp8:
                v_f = d4090_fp8[c_val]["throughput_tok_per_s"]
                ax_a.annotate(f"{v_f:.0f}", (c_val, v_f), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8.5, fontweight="bold", color="#2980b9")

        # Right Panel: KV Cache Capacity (Tokens) vs Max 4k Concurrency
        configs = [
            "RTX 3070 (8GB)\nFP16 KV (1.82 GiB)",
            "RTX 4090 (24GB)\nFP16 KV (14.5 GiB)",
            "RTX 4090 (24GB)\nFP8 KV (14.5 GiB)",
        ]
        kv_tokens_k = [68.18, 542.91, 1085.82]  # in thousands of tokens
        max_conc_4k = [16.64, 132.55, 265.09]

        x_pos = np.arange(len(configs))
        bars_kv = ax_b.bar(x_pos, kv_tokens_k, width=0.5, color=["#9b59b6", "#2ecc71", "#3498db"], alpha=0.88)
        ax_b.set_title("Allocated KV Cache Pool Size (Thousands of Tokens)", fontsize=11, fontweight="bold")
        ax_b.set_xticks(x_pos)
        ax_b.set_xticklabels(configs, fontsize=9.5)
        ax_b.set_ylabel("Total KV Cache Capacity (K Tokens)")
        ax_b.set_ylim(0, 1280)

        for bar, k_val, m_c in zip(bars_kv, kv_tokens_k, max_conc_4k):
            ax_b.annotate(
                f"{k_val:,.1f}K tokens\n({m_c:.1f}x @ 4k ctx)",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout(rect=[0, 0, 1, 0.91])
        chart3_path = out / "chart_3070_vs_4090_fp8_scaling.png"
        plt.savefig(chart3_path)
        plt.close(fig3)
        print(f"Generated charts:\n- {chart1_path}\n- {chart2_path}\n- {chart3_path}")
    else:
        print(f"Generated charts:\n- {chart1_path}\n- {chart2_path}")


if __name__ == "__main__":
    import sys
    generate_charts(sys.argv[1], sys.argv[2], sys.argv[3])
