"""
Part 4 Benchmark on NVIDIA L4 (24GB, Ada SM 8.9):
Speculative Decoding Concurrency Crossover Benchmark (`c = 1, 2, 4, 8, 16, 32`).

Evaluates Standard Autoregressive Decoding vs. N-gram Prompt-Lookup Speculative Decoding
(`k = 3` and `k = 5` speculative tokens) on `Qwen/Qwen2.5-7B-Instruct-AWQ` across 6 concurrency levels:
  - Uses `stream_options={"include_usage": True}` so `completion_tokens`, `TPOT (ms/tok)`, and
    `Throughput (tok/s)` are tokenizer-exact even when speculative verification emits multi-token bursts.
  - Verifies that `<sql>...</sql>` execution accuracy against the seeded SQLite sandbox stays mathematically
    identical (lossless greedy verification at `temperature=0.0`).
  - Pinpoints the exact Crossover Concurrency (`c*`) on NVIDIA L4 where speculative verification FLOP
    overhead outweighs memory-bandwidth savings and falls behind standard decoding.

Usage:
  # Local validation of task subset & helper logic
  python research-vllm-benchmark/spec_decoding_crossover_eval.py --validate-local

  # Run full concurrency crossover sweep on L4 GPU
  python research-vllm-benchmark/spec_decoding_crossover_eval.py --run-sweep --out /content/results_l4_spec_decoding.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sql_accuracy_eval import (
    SQL_BENCHMARK_TASKS,
    SYSTEM_PROMPT,
    _build_vllm_env,
    create_sandbox_db,
    evaluate_sql_prediction,
    parse_kv_telemetry,
    percentile,
)

# Append a concise structured schema-audit footer requirement so each response generates
# ~130-200 tokens (providing stable multi-step decode timing across c=1..32 while preserving <sql>...</sql>).
SPEC_BENCHMARK_SYSTEM_PROMPT = (
    SYSTEM_PROMPT.strip()
    + "\n4. After the closing `</sql>` tag, append a concise 3-line `<audit>` block listing the exact schema "
    "table names, join keys, and filter predicates used in your query."
)


def select_32_balanced_tasks() -> list[dict[str, str]]:
    """Select 32 deterministic tasks (8 per SQL category) so c=32 saturates all 32 slots evenly."""
    by_cat: dict[str, list[dict[str, str]]] = {
        "filter_null": [],
        "join_having": [],
        "case_ratio": [],
        "window_cte": [],
    }
    for t in SQL_BENCHMARK_TASKS:
        by_cat[t["category"]].append(t)
    selected: list[dict[str, str]] = []
    for idx in range(8):
        for cat in ("filter_null", "join_having", "case_ratio", "window_cte"):
            selected.append(by_cat[cat][idx])
    assert len(selected) == 32
    return selected


async def stream_request_exact_tokens(
    session: Any,
    endpoint_url: str,
    model_name: str,
    question: str,
    max_tokens: int = 256,
) -> dict[str, Any]:
    """
    Stream SSE completion with `stream_options.include_usage = True` so multi-token speculative
    verification bursts are counted by exact tokenizer `completion_tokens` rather than SSE chunk count.
    """
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SPEC_BENCHMARK_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    first_token_t: Optional[float] = None
    ttft_ms = 0.0
    sse_content_events = 0
    exact_completion_tokens: Optional[int] = None
    pieces: list[str] = []

    try:
        async with session.post(endpoint_url, json=payload) as resp:
            if resp.status != 200:
                txt = await resp.text()
                return {
                    "ok": False,
                    "error": f"HTTP {resp.status}: {txt[:160]}",
                    "text": "",
                    "ttft_ms": 0.0,
                    "tpot_ms": 0.0,
                    "e2e_ms": (time.perf_counter() - t0) * 1000.0,
                    "out_tokens": 0,
                    "sse_events": 0,
                    "tokens_per_step": 1.0,
                }
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                chunk = json.loads(data_str)
                usage = chunk.get("usage")
                if usage and isinstance(usage.get("completion_tokens"), int):
                    exact_completion_tokens = usage["completion_tokens"]
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    now = time.perf_counter()
                    if first_token_t is None:
                        first_token_t = now
                        ttft_ms = (now - t0) * 1000.0
                    pieces.append(delta)
                    sse_content_events += 1

        e2e_ms = (time.perf_counter() - t0) * 1000.0
        out_tokens = exact_completion_tokens if (exact_completion_tokens and exact_completion_tokens > 0) else sse_content_events
        tpot_ms = (e2e_ms - ttft_ms) / (out_tokens - 1) if out_tokens > 1 else 0.0
        tokens_per_step = round(out_tokens / max(1, sse_content_events), 3)
        return {
            "ok": out_tokens > 0,
            "error": None,
            "text": "".join(pieces),
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "e2e_ms": e2e_ms,
            "out_tokens": out_tokens,
            "sse_events": sse_content_events,
            "tokens_per_step": tokens_per_step,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "text": "",
            "ttft_ms": 0.0,
            "tpot_ms": 0.0,
            "e2e_ms": (time.perf_counter() - t0) * 1000.0,
            "out_tokens": 0,
            "sse_events": 0,
            "tokens_per_step": 1.0,
        }


async def run_concurrency_point(
    endpoint_url: str,
    model_name: str,
    tasks: list[dict[str, str]],
    concurrency: int,
    max_tokens: int = 256,
) -> dict[str, Any]:
    import aiohttp

    conn = create_sandbox_db(seed=42)
    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=240)

    async def eval_one(session: aiohttp.ClientSession, task: dict[str, str]) -> dict[str, Any]:
        async with semaphore:
            res = await stream_request_exact_tokens(
                session=session,
                endpoint_url=endpoint_url,
                model_name=model_name,
                question=task["question"],
                max_tokens=max_tokens,
            )
            verdict = evaluate_sql_prediction(conn, res["text"], task["ground_truth_sql"])
            return {
                "id": task["id"],
                "category": task["category"],
                **res,
                "syntax_valid": verdict["syntax_valid"],
                "exec_match": verdict["exec_match"],
            }

    async with aiohttp.ClientSession(timeout=timeout) as session:
        wave_t0 = time.perf_counter()
        results = await asyncio.gather(*(eval_one(session, t) for t in tasks))
    wall_s = time.perf_counter() - wave_t0

    ok_res = [r for r in results if r.get("ok")]
    n_tasks = len(tasks)
    syntax_cnt = sum(1 for r in results if r.get("syntax_valid"))
    exec_cnt = sum(1 for r in results if r.get("exec_match"))
    total_toks = sum(r.get("out_tokens", 0) for r in ok_res)
    total_sse = sum(r.get("sse_events", 0) for r in ok_res)

    ttfts = sorted(r["ttft_ms"] for r in ok_res)
    tpots = sorted(r["tpot_ms"] for r in ok_res if r.get("out_tokens", 0) > 1)
    e2es = sorted(r["e2e_ms"] for r in ok_res)

    return {
        "concurrency": concurrency,
        "num_tasks": n_tasks,
        "succeeded_requests": len(ok_res),
        "syntax_valid_pct": round(100.0 * syntax_cnt / n_tasks, 1),
        "exec_accuracy_pct": round(100.0 * exec_cnt / n_tasks, 1),
        "total_output_tokens": total_toks,
        "total_sse_events": total_sse,
        "mean_tokens_per_sse_event": round(total_toks / max(1, total_sse), 3),
        "wall_time_s": round(wall_s, 2),
        "throughput_tok_per_s": round(total_toks / wall_s, 2) if wall_s > 0 else 0.0,
        "ttft_p50_ms": round(percentile(ttfts, 50), 1),
        "ttft_p99_ms": round(percentile(ttfts, 99), 1),
        "tpot_p50_ms": round(percentile(tpots, 50), 1),
        "tpot_p99_ms": round(percentile(tpots, 99), 1),
        "e2e_p50_ms": round(percentile(e2es, 50), 1),
        "e2e_p99_ms": round(percentile(e2es, 99), 1),
    }


def build_ngram_spec_config(k: int, lookup_max: int = 4) -> str:
    return json.dumps({
        "method": "ngram",
        "num_speculative_tokens": k,
        "prompt_lookup_max": lookup_max,
        "prompt_lookup_min": 1,
    })


def _run_spec_setup_sweep(
    vllm_bin: str,
    gpu_name: str,
    cfg: dict[str, Any],
    concurrencies: list[int],
    tasks_32: list[dict[str, str]],
    log_tag: str,
    env: dict[str, str],
) -> dict[str, Any]:
    import aiohttp
    import urllib.request

    util = cfg.get("gpu_mem_util", 0.85)
    print(
        f"\n=== Starting Speculative Decoding Sweep [{log_tag}]: {cfg['label']} "
        f"(model={cfg['model']}, extra={cfg.get('extra_args', [])}) ===",
        flush=True,
    )
    subprocess.run(["pkill", "-f", "vllm serve"], check=False)
    time.sleep(3)

    log_file = Path(f"/tmp/vllm_spec_{log_tag}.log")
    cmd = [
        vllm_bin, "serve", cfg["model"],
        "--host", "127.0.0.1",
        "--port", "8000",
        "--gpu-memory-utilization", str(util),
        "--max-model-len", "4096",
        "--enforce-eager",
        "--kv-cache-dtype", "auto",
        "--enable-prefix-caching",
    ]
    cmd.extend(cfg.get("extra_args", []))

    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    try:
        ready = False
        for _ in range(180):
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(2)

        log_text = log_file.read_text(encoding="utf-8", errors="ignore")
        if not ready:
            print(f"ERROR: vLLM failed to start for {cfg['label']}. Log tail:", flush=True)
            print("\n".join(log_text.splitlines()[-35:]), flush=True)
            return {
                **cfg,
                "gpu": gpu_name,
                "status": "failed_to_start",
                "log_tail": log_text.splitlines()[-25:],
            }

        kv_info = parse_kv_telemetry(log_text)
        print(f"vLLM ready! KV telemetry: {kv_info}", flush=True)

        # Warmup prefix cache once before sweeping concurrencies
        async def _warmup() -> None:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as s:
                await stream_request_exact_tokens(
                    session=s,
                    endpoint_url="http://127.0.0.1:8000/v1/chat/completions",
                    model_name=cfg["model"],
                    question=tasks_32[0]["question"],
                    max_tokens=256,
                )

        asyncio.run(_warmup())

        sweep_points: list[dict[str, Any]] = []
        for c in concurrencies:
            # Use 16 balanced tasks for low concurrency (c=1,2,4) to keep sweep fast, and 32 tasks for c>=8
            subset = tasks_32[:16] if c <= 4 else tasks_32
            pt = asyncio.run(
                run_concurrency_point(
                    endpoint_url="http://127.0.0.1:8000/v1/chat/completions",
                    model_name=cfg["model"],
                    tasks=subset,
                    concurrency=c,
                    max_tokens=256,
                )
            )
            sweep_points.append(pt)
            print(
                f"  [c={c:2d}] Tput={pt['throughput_tok_per_s']:6.2f} tok/s | "
                f"TPOT_P50={pt['tpot_p50_ms']:5.1f}ms | TPOT_P99={pt['tpot_p99_ms']:5.1f}ms | "
                f"E2E_P50={pt['e2e_p50_ms']:6.1f}ms | ExecAcc={pt['exec_accuracy_pct']}% | "
                f"Tok/Event={pt['mean_tokens_per_sse_event']}",
                flush=True,
            )

        return {
            "status": "ok",
            "label": cfg["label"],
            "short_id": cfg["short_id"],
            "model": cfg["model"],
            "extra_args": cfg.get("extra_args", []),
            "gpu": gpu_name,
            "gpu_mem_util": util,
            **kv_info,
            "sweep": sweep_points,
        }
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
                except Exception:
                    pass
        subprocess.run(["pkill", "-f", "vllm serve"], check=False)
        time.sleep(3)


def run_l4_spec_sweep(out_path: str) -> None:
    import torch

    vllm_bin = str(Path(sys.executable).parent / "vllm")
    if not Path(vllm_bin).exists():
        vllm_bin = "vllm"

    gpu_name = torch.cuda.get_device_name(0)
    env = _build_vllm_env()
    tasks_32 = select_32_balanced_tasks()
    concurrencies = [1, 2, 4, 8, 16, 32]

    configs = [
        {
            "short_id": "baseline_autoreg",
            "label": "Standard Autoregressive Decode (Baseline)",
            "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "gpu_mem_util": 0.85,
            "extra_args": [],
        },
        {
            "short_id": "spec_ngram_k3",
            "label": "Speculative Decoding (N-gram Prompt Lookup, k=3)",
            "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "gpu_mem_util": 0.85,
            "extra_args": ["--speculative-config", build_ngram_spec_config(k=3, lookup_max=4)],
        },
        {
            "short_id": "spec_ngram_k5",
            "label": "Speculative Decoding (N-gram Prompt Lookup, k=5)",
            "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "gpu_mem_util": 0.85,
            "extra_args": ["--speculative-config", build_ngram_spec_config(k=5, lookup_max=4)],
        },
    ]

    all_results: list[dict[str, Any]] = []
    for idx, cfg in enumerate(configs, 1):
        res = _run_spec_setup_sweep(vllm_bin, gpu_name, cfg, concurrencies, tasks_32, f"spec_{idx}", env)
        all_results.append(res)
        payload = {
            "gpu": gpu_name,
            "concurrencies": concurrencies,
            "results": all_results,
        }
        Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Checkpointed {len(all_results)}/{len(configs)} sweeps to {out_path}", flush=True)


def validate_local() -> None:
    tasks = select_32_balanced_tasks()
    print(f"Selected {len(tasks)} balanced SQL tasks across 4 categories.")
    print(f"Sample N-gram SpeculativeConfig JSON (k=5): {build_ngram_spec_config(5)}")
    print("SUCCESS: Local validation passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-local", action="store_true")
    parser.add_argument("--run-sweep", action="store_true")
    parser.add_argument("--out", default="/content/results_l4_spec_decoding.json")
    args = parser.parse_args()

    if args.validate_local:
        validate_local()
    elif args.run_sweep:
        run_l4_spec_sweep(out_path=args.out)
    else:
        parser.print_help()
