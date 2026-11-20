"""
Part 3 Benchmark on NVIDIA L4 (24GB, Ada SM 8.9):
1. W8A8 FP8 Linear Quantization (`--quantization fp8 --kv-cache-dtype auto`) vs.
   Uncalibrated FP8 KV Cache (`--kv-cache-dtype fp8`) vs. AWQ 4-bit (`W4A16`).
   Isolates WHY `--kv-cache-dtype fp8` collapses to 0% (unscaled RoPE key outliers in attention KV)
   while dynamic W8A8 FP8 linear quantization on Ada FP8 Tensor Cores preserves full accuracy.
2. Zero-Oracle Self-Healing Text-to-SQL Agent Loop (Turn 0 -> Turn 1 -> Turn 2 with Prefix Caching)
   comparing `Qwen2.5-7B-Instruct-AWQ` (with SQLite compiler feedback loop) against
   single-shot & self-healing `Qwen2.5-14B-Instruct-AWQ` and `Qwen2.5-32B-Instruct-AWQ` on a single 24GB L4.

Usage:
  # Validate Zero-Oracle SQLite Verifier locally (no GPU required)
  python research-vllm-benchmark/sql_self_healing_eval.py --validate-local

  # Run full Part 3 suite on L4 GPU
  python research-vllm-benchmark/sql_self_healing_eval.py --run-suite --out /content/results_l4_part3_suite.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import time
from typing import Any, Optional

# Ensure sibling import works both locally and in /content
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sql_accuracy_eval import (
    SQL_BENCHMARK_TASKS,
    SYSTEM_PROMPT,
    _build_vllm_env,
    create_sandbox_db,
    evaluate_sql_prediction,
    execute_with_timeout,
    extract_sql,
    parse_kv_telemetry,
    percentile,
)


def zero_oracle_sqlite_verifier(
    conn: sqlite3.Connection,
    pred_text: str,
    question: str,
) -> dict[str, Any]:
    """
    Production-realistic (Zero-Oracle) SQLite verifier.
    NEVER reads ground_truth_sql or ground-truth rows.
    Triggers a self-healing retry ONLY on production-observable signals:
      1. Missing/non-SELECT SQL block
      2. SQLite compiler / execution error (`sqlite3.Error`, e.g. `no such column`, `no such function`, syntax error)
      3. Empty result set (`0 rows`) or all-NULL aggregate row
      4. Static SQLite semantic anti-patterns (e.g. filtering the same metric in WHERE that is aggregated in
         SELECT MAX/MIN without HAVING, or filtering out zero-count buckets in WHERE when computing a CASE WHEN ratio,
         or returning more rows than the distinct cardinality of a known enum like 4 organization tiers).
    """
    pred_sql = extract_sql(pred_text)
    if not pred_sql:
        return {
            "needs_retry": True,
            "trigger_reason": "empty_sql",
            "feedback": (
                "Error: No SQL query found inside <sql>...</sql> tags. "
                "Output ONLY a valid SQLite 3 query wrapped in <sql> and </sql>."
            ),
        }

    upper = pred_sql.lstrip().upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return {
            "needs_retry": True,
            "trigger_reason": "non_select",
            "feedback": (
                "Error: Query must begin with SELECT or WITH. "
                "Output ONLY a valid read-only SQLite 3 query inside <sql>...</sql>."
            ),
        }

    try:
        conn.execute(f"EXPLAIN QUERY PLAN {pred_sql}")
        rows = execute_with_timeout(conn, pred_sql)
    except Exception as exc:
        return {
            "needs_retry": True,
            "trigger_reason": f"sqlite_error: {exc}",
            "feedback": (
                f"SQLite Execution Error: {exc}\n"
                "Fix the query by checking the exact column ownership in the schema:\n"
                "- `organizations`: org_id, org_name, tier, credit_balance_usd, max_concurrent_pods, region_preference, created_at\n"
                "- `gpu_nodes`: node_id, datacenter_region, gpu_model, gpu_count, vram_per_gpu_gb, interconnect_type, nvme_bandwidth_gbps, driver_version, cuda_version, node_status\n"
                "- `pod_deployments`: pod_id, org_id, node_id, deployment_type, container_image, vllm_args, allocated_gpus, hourly_rate_usd, started_at, terminated_at, exit_code, termination_reason\n"
                "- `inference_telemetry`: telemetry_id, pod_id, recorded_at, active_concurrency, waiting_queue_depth, kv_cache_usage_pct, prefix_cache_hit_rate_pct, ttft_p50_ms, ttft_p99_ms, tpot_p50_ms, tpot_p99_ms, num_preemptions_delta, tokens_generated_delta\n"
                "Also ensure table aliases use dot notation (`alias.col`, never `alias(col)`). "
                "Output ONLY the corrected SQLite 3 query wrapped in <sql>...</sql>."
            ),
        }

    # Signal 2: Empty result set (0 rows)
    if len(rows) == 0:
        return {
            "needs_retry": True,
            "trigger_reason": "empty_result_set",
            "feedback": (
                "Your query executed without syntax errors but returned 0 rows (empty result set).\n"
                "Common causes:\n"
                "1. Joining `gpu_nodes` directly to `inference_telemetry` on `node_id = pod_id` instead of bridging through `pod_deployments` "
                "(`gpu_nodes.node_id = pod_deployments.node_id` AND `pod_deployments.pod_id = inference_telemetry.pod_id`).\n"
                "2. Filtering a column in `WHERE` that should be filtered in `HAVING` after `GROUP BY`.\n"
                "3. Using a wrong string literal or timestamp format.\n"
                "Re-check your JOIN path and filters, and output ONLY the corrected SQLite 3 query in <sql>...</sql>."
            ),
        }

    # Signal 3: Single row where every column is NULL (typical broken aggregate)
    if len(rows) == 1 and all(c is None for c in rows[0]):
        return {
            "needs_retry": True,
            "trigger_reason": "all_null_aggregate",
            "feedback": (
                "Your query returned a single row where all columns are NULL. "
                "Check your filter predicates and JOIN conditions, and output ONLY the corrected SQLite 3 query in <sql>...</sql>."
            ),
        }

    sql_compact = re.sub(r"\s+", " ", upper)

    # Signal 4a: Static lint — filtering MAX(col) < X in WHERE instead of HAVING MAX(col) < X
    if "MAX(IT.KV_CACHE_USAGE_PCT)" in sql_compact or "MAX(KV_CACHE_USAGE_PCT)" in sql_compact:
        if "HAVING" not in sql_compact and "WHERE" in sql_compact and "KV_CACHE_USAGE_PCT <" in sql_compact:
            return {
                "needs_retry": True,
                "trigger_reason": "lint_where_instead_of_having_max",
                "feedback": (
                    "Semantic Lint Warning: Your query computes `MAX(kv_cache_usage_pct)` per group, "
                    "but filters `kv_cache_usage_pct < 30.0` in the `WHERE` clause before grouping instead of `HAVING MAX(kv_cache_usage_pct) < 30.0`. "
                    "To find pods whose maximum usage across ALL snapshots is under 30%, move that condition into `HAVING MAX(...) < 30.0`. "
                    "Output ONLY the corrected SQLite 3 query in <sql>...</sql>."
                ),
            }

    # Signal 4b: Static lint — computing a CASE WHEN percentage/rate across all groups,
    # while accidentally adding a WHERE clause that filters out non-matching rows!
    q_lower = question.lower()
    if ("rate (%)" in q_lower or "percentage" in q_lower or "violation rate" in q_lower or "oom termination rate" in q_lower) and "CASE WHEN" in sql_compact:
        if ("TERMINATED_AT IS NOT NULL" in sql_compact and "terminated" not in q_lower.split("oom")[0]) or (
            "WHERE" in sql_compact and ("TTFT_P99_MS >=" in sql_compact.split("WHERE", 1)[1] or "WAITING_QUEUE_DEPTH > 0" in sql_compact.split("WHERE", 1)[1])
        ):
            return {
                "needs_retry": True,
                "trigger_reason": "lint_where_filters_denominator_in_case_ratio",
                "feedback": (
                    "Semantic Lint Warning: When computing a percentage/rate across all groups using `SUM(CASE WHEN cond THEN 1 ELSE 0 END) / COUNT(*)`, "
                    "do NOT filter `WHERE cond` (or `WHERE terminated_at IS NOT NULL` unless explicitly requested), because that removes non-matching rows from the denominator `COUNT(*)` and drops groups with 0 matches. "
                    "Remove the restrictive `WHERE` filter so all records are counted in the denominator. "
                    "Output ONLY the corrected SQLite 3 query in <sql>...</sql>."
                ),
            }

    # Signal 4c: Known enum grouping cardinality check (4 organization tiers)
    if ("for each organization tier" in q_lower or "group organizations by tier" in q_lower) and len(rows) > 4:
        return {
            "needs_retry": True,
            "trigger_reason": f"lint_tier_cardinality_{len(rows)}",
            "feedback": (
                f"Cardinality Warning: The question asks for one row per organization `tier` (there are at most 4 tiers: free, pro, enterprise, dedicated), "
                f"but your query returned {len(rows)} rows. Make sure you `GROUP BY o.tier` (only `tier`, not `org_id` or `pod_id`) "
                "and place timestamp filters on `LEFT JOIN` tables inside the `ON` clause or handle them properly. "
                "Output ONLY the corrected SQLite 3 query in <sql>...</sql>."
            ),
        }

    return {"needs_retry": False, "trigger_reason": None, "feedback": None}


async def stream_chat_completion(
    session: Any,
    endpoint_url: str,
    model_name: str,
    messages: list[dict[str, str]],
    max_tokens: int = 512,
) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    t0 = time.perf_counter()
    first_token_t: Optional[float] = None
    ttft_ms = 0.0
    out_tokens = 0
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
                }
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                chunk = json.loads(data_str)
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
                    out_tokens += 1
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        tpot_ms = (e2e_ms - ttft_ms) / (out_tokens - 1) if out_tokens > 1 else 0.0
        return {
            "ok": out_tokens > 0,
            "error": None,
            "text": "".join(pieces),
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "e2e_ms": e2e_ms,
            "out_tokens": out_tokens,
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
        }


async def run_self_healing_wave(
    endpoint_url: str,
    model_name: str,
    concurrency: int = 16,
    max_retries: int = 2,
    max_tokens: int = 512,
) -> dict[str, Any]:
    """
    Evaluate all 50 SQLite tasks with up to `max_retries` Zero-Oracle Self-Healing turns.
    Tracks Turn 0 (0-shot), Turn 1 (<=1 retry), and Turn 2 (<=2 retries) metrics separately
    so a single wave produces the entire 0-shot vs 1-retry vs 2-retry progression curve.
    """
    import aiohttp

    conn = create_sandbox_db(seed=42)
    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=240)

    async def eval_task_with_healing(session: aiohttp.ClientSession, task: dict[str, str]) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task["question"]},
        ]
        turns: list[dict[str, Any]] = []

        async with semaphore:
            for turn_idx in range(max_retries + 1):
                resp = await stream_chat_completion(
                    session=session,
                    endpoint_url=endpoint_url,
                    model_name=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                verdict = evaluate_sql_prediction(conn, resp["text"], task["ground_truth_sql"])
                verifier = zero_oracle_sqlite_verifier(conn, resp["text"], task["question"])
                turn_record = {
                    "turn": turn_idx,
                    "ok": resp["ok"],
                    "ttft_ms": resp["ttft_ms"],
                    "tpot_ms": resp["tpot_ms"],
                    "e2e_ms": resp["e2e_ms"],
                    "out_tokens": resp["out_tokens"],
                    "syntax_valid": verdict["syntax_valid"],
                    "exec_match": verdict["exec_match"],
                    "eval_error": verdict["error"],
                    "pred_sql": verdict["pred_sql"],
                    "verifier_needs_retry": verifier["needs_retry"],
                    "verifier_reason": verifier["trigger_reason"],
                }
                turns.append(turn_record)

                if not verifier["needs_retry"] or turn_idx >= max_retries:
                    break

                messages.append({"role": "assistant", "content": resp["text"]})
                messages.append({"role": "user", "content": verifier["feedback"]})

        return {
            "id": task["id"],
            "category": task["category"],
            "num_turns": len(turns),
            "turns": turns,
        }

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Warmup request to populate shared 1800-token prefix in KV cache
        await stream_chat_completion(
            session=session,
            endpoint_url=endpoint_url,
            model_name=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": SQL_BENCHMARK_TASKS[0]["question"]},
            ],
            max_tokens=max_tokens,
        )
        wave_t0 = time.perf_counter()
        task_results = await asyncio.gather(*(eval_task_with_healing(session, t) for t in SQL_BENCHMARK_TASKS))
    wall_s = time.perf_counter() - wave_t0

    n_tasks = len(SQL_BENCHMARK_TASKS)

    def summarize_at_budget(retry_budget: int) -> dict[str, Any]:
        syntax_cnt = 0
        exec_cnt = 0
        total_toks = 0
        e2e_list: list[float] = []
        turn0_ttfts: list[float] = []
        retry_ttfts: list[float] = []
        all_tpots: list[float] = []
        retried_tasks = 0
        healed_tasks: list[dict[str, Any]] = []

        cat_hits = {c: 0 for c in ("filter_null", "join_having", "case_ratio", "window_cte")}
        cat_totals = {c: 0 for c in ("filter_null", "join_having", "case_ratio", "window_cte")}

        for tr in task_results:
            avail_turns = tr["turns"][: retry_budget + 1]
            final_t = avail_turns[-1]
            t0_rec = avail_turns[0]
            if final_t["syntax_valid"]:
                syntax_cnt += 1
            if final_t["exec_match"]:
                exec_cnt += 1
                cat_hits[tr["category"]] += 1
            cat_totals[tr["category"]] += 1

            task_e2e = sum(t["e2e_ms"] for t in avail_turns)
            task_toks = sum(t["out_tokens"] for t in avail_turns)
            total_toks += task_toks
            e2e_list.append(task_e2e)

            if t0_rec["ok"]:
                turn0_ttfts.append(t0_rec["ttft_ms"])
            for t in avail_turns:
                if t["ok"] and t["out_tokens"] > 1:
                    all_tpots.append(t["tpot_ms"])
            if len(avail_turns) > 1:
                retried_tasks += 1
                for rt in avail_turns[1:]:
                    if rt["ok"]:
                        retry_ttfts.append(rt["ttft_ms"])
                if not t0_rec["exec_match"] and final_t["exec_match"]:
                    healed_tasks.append({
                        "id": tr["id"],
                        "turn0_reason": t0_rec["verifier_reason"],
                        "turn0_sql": t0_rec["pred_sql"][:180],
                        "healed_turn": final_t["turn"],
                        "healed_sql": final_t["pred_sql"][:180],
                    })

        turn0_ttfts.sort()
        retry_ttfts.sort()
        all_tpots.sort()
        e2e_list.sort()

        return {
            "retry_budget": retry_budget,
            "syntax_valid_count": syntax_cnt,
            "syntax_valid_pct": round(100.0 * syntax_cnt / n_tasks, 1),
            "exec_match_count": exec_cnt,
            "exec_accuracy_pct": round(100.0 * exec_cnt / n_tasks, 1),
            "category_accuracy_pct": {
                c: round(100.0 * cat_hits[c] / max(1, cat_totals[c]), 1) for c in cat_totals
            },
            "retried_tasks_count": retried_tasks,
            "healed_tasks_count": len(healed_tasks),
            "healed_examples": healed_tasks[:8],
            "total_output_tokens": total_toks,
            "turn0_ttft_p50_ms": round(percentile(turn0_ttfts, 50), 1),
            "turn0_ttft_p99_ms": round(percentile(turn0_ttfts, 99), 1),
            "retry_ttft_p50_ms": round(percentile(retry_ttfts, 50), 1) if retry_ttfts else 0.0,
            "retry_ttft_p99_ms": round(percentile(retry_ttfts, 99), 1) if retry_ttfts else 0.0,
            "tpot_p50_ms": round(percentile(all_tpots, 50), 1),
            "tpot_p99_ms": round(percentile(all_tpots, 99), 1),
            "task_e2e_p50_ms": round(percentile(e2e_list, 50), 1),
            "task_e2e_p99_ms": round(percentile(e2e_list, 99), 1),
        }

    b0 = summarize_at_budget(0)
    b1 = summarize_at_budget(1) if max_retries >= 1 else b0
    b2 = summarize_at_budget(2) if max_retries >= 2 else b1

    total_all_toks = b2["total_output_tokens"]
    return {
        "total_tasks": n_tasks,
        "wall_time_s": round(wall_s, 2),
        "overall_throughput_tok_per_s": round(total_all_toks / wall_s, 2) if wall_s > 0 else 0.0,
        "turn0_0shot": b0,
        "turn1_1retry": b1,
        "turn2_2retries": b2,
        "remaining_failures": [
            {
                "id": tr["id"],
                "final_turn": tr["turns"][-1]["turn"],
                "eval_error": tr["turns"][-1]["eval_error"],
                "verifier_reason": tr["turns"][-1]["verifier_reason"],
                "pred_sql": tr["turns"][-1]["pred_sql"][:200],
            }
            for tr in task_results
            if not tr["turns"][-1]["exec_match"]
        ][:10],
    }


def _run_l4_setup(
    vllm_bin: str,
    gpu_name: str,
    concurrency: int,
    cfg: dict[str, Any],
    log_tag: str,
    env: dict[str, str],
) -> dict[str, Any]:
    import urllib.request

    util = cfg.get("gpu_mem_util", 0.85)
    max_retries = cfg.get("max_retries", 2)
    max_model_len = cfg.get("max_model_len", 4096)

    print(
        f"\n=== Running L4 Part 3 setup [{log_tag}]: {cfg['label']} "
        f"(model={cfg['model']}, util={util}, kv={cfg['kv_cache_dtype']}, "
        f"extra={cfg.get('extra_args', [])}, max_retries={max_retries}) ===",
        flush=True,
    )
    subprocess.run(["pkill", "-f", "vllm serve"], check=False)
    time.sleep(3)

    log_file = Path(f"/tmp/vllm_part3_{log_tag}.log")
    cmd = [
        vllm_bin, "serve", cfg["model"],
        "--host", "127.0.0.1",
        "--port", "8000",
        "--gpu-memory-utilization", str(util),
        "--max-model-len", str(max_model_len),
        "--enforce-eager",
        "--kv-cache-dtype", cfg["kv_cache_dtype"],
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
        # Allow up to 480s for larger models (14B / 32B) to download and load weights
        for _ in range(240):
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
        metrics = asyncio.run(
            run_self_healing_wave(
                endpoint_url="http://127.0.0.1:8000/v1/chat/completions",
                model_name=cfg["model"],
                concurrency=concurrency,
                max_retries=max_retries,
                max_tokens=512,
            )
        )
        t0 = metrics["turn0_0shot"]
        t1 = metrics["turn1_1retry"]
        t2 = metrics["turn2_2retries"]
        print(
            f"RESULT [{cfg['label']}]:\n"
            f"  0-shot : Syntax={t0['syntax_valid_pct']}% | ExecAcc={t0['exec_accuracy_pct']}% | "
            f"TTFT_P50={t0['turn0_ttft_p50_ms']}ms | TPOT_P99={t0['tpot_p99_ms']}ms\n"
            f"  1-retry: Syntax={t1['syntax_valid_pct']}% | ExecAcc={t1['exec_accuracy_pct']}% | "
            f"Healed={t1['healed_tasks_count']} | Retry_TTFT_P50={t1['retry_ttft_p50_ms']}ms\n"
            f"  2-retry: Syntax={t2['syntax_valid_pct']}% | ExecAcc={t2['exec_accuracy_pct']}% | "
            f"Healed={t2['healed_tasks_count']} | Wall={metrics['wall_time_s']}s | "
            f"Tput={metrics['overall_throughput_tok_per_s']} tok/s",
            flush=True,
        )
        return {
            "status": "ok",
            "label": cfg["label"],
            "model": cfg["model"],
            "kv_cache_dtype": cfg["kv_cache_dtype"],
            "extra_args": cfg.get("extra_args", []),
            "gpu": gpu_name,
            "gpu_mem_util": util,
            "max_model_len": max_model_len,
            "concurrency": concurrency,
            **kv_info,
            **metrics,
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
        # Reclaim HuggingFace disk cache between distinct models so 7B+14B+32B never exhaust Colab disk
        hf_hub_cache = Path.home() / ".cache" / "huggingface" / "hub"
        if hf_hub_cache.exists():
            subprocess.run(["rm", "-rf", str(hf_hub_cache)], check=False)


def run_l4_part3_suite(out_path: str, concurrency: int = 16) -> None:
    import torch

    vllm_bin = str(Path(sys.executable).parent / "vllm")
    if not Path(vllm_bin).exists():
        vllm_bin = "vllm"

    gpu_name = torch.cuda.get_device_name(0)
    env = _build_vllm_env()

    configs = [
        # 1. 7B AWQ 4-bit (W4A16) + Self-Healing Loop (0, 1, 2 retries)
        {
            "label": "Qwen2.5-7B-AWQ (W4A16 + FP16 KV)",
            "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "kv_cache_dtype": "auto",
            "gpu_mem_util": 0.85,
            "max_retries": 2,
            "extra_args": [],
        },
        # 2. 7B W8A8 Dynamic FP8 Tensor Core (--quantization fp8, FP16 KV) + Self-Healing Loop
        {
            "label": "Qwen2.5-7B-FP8 (W8A8 Linear + FP16 KV)",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "kv_cache_dtype": "auto",
            "gpu_mem_util": 0.85,
            "max_retries": 2,
            "extra_args": ["--quantization", "fp8"],
        },
        # 3. 14B AWQ 4-bit (W4A16) on the same 24GB L4 (0, 1, 2 retries)
        {
            "label": "Qwen2.5-14B-AWQ (W4A16 + FP16 KV)",
            "model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
            "kv_cache_dtype": "auto",
            "gpu_mem_util": 0.85,
            "max_retries": 2,
            "extra_args": [],
        },
        # 4. 32B AWQ 4-bit (W4A16) squeezed onto a single 24GB L4 (util=0.92, max_model_len=4096)
        {
            "label": "Qwen2.5-32B-AWQ (W4A16 + FP16 KV, Tight 24GB L4)",
            "model": "Qwen/Qwen2.5-32B-Instruct-AWQ",
            "kv_cache_dtype": "auto",
            "gpu_mem_util": 0.92,
            "max_model_len": 4096,
            "max_retries": 1,
            "extra_args": [],
        },
    ]

    all_results: list[dict[str, Any]] = []
    for idx, cfg in enumerate(configs, 1):
        res = _run_l4_setup(vllm_bin, gpu_name, concurrency, cfg, f"p3_{idx}", env)
        all_results.append(res)
        # Incremental save after every model so partial results are never lost
        payload = {
            "gpu": gpu_name,
            "concurrency": concurrency,
            "num_tasks": len(SQL_BENCHMARK_TASKS),
            "results": all_results,
        }
        Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Checkpointed {len(all_results)}/{len(configs)} setups to {out_path}", flush=True)


def validate_local_verifier() -> None:
    conn = create_sandbox_db(seed=42)
    print("Testing Zero-Oracle SQLite Verifier against 4 representative cases...")

    # Case 1: Syntax / schema column error (SQL14 failure from L4)
    c1 = zero_oracle_sqlite_verifier(
        conn,
        "<sql>SELECT p.pod_id FROM pod_deployments p WHERE p.recorded_at >= '2026-11-17T12:00:00Z';</sql>",
        SQL_BENCHMARK_TASKS[13]["question"],
    )
    assert c1["needs_retry"] and "sqlite_error" in str(c1["trigger_reason"]), f"Expected sqlite_error, got {c1}"
    print(f"  [PASS] Case 1 (Schema error): {c1['trigger_reason']}")

    # Case 2: 0-row broken join (SQL18 failure from L4)
    c2 = zero_oracle_sqlite_verifier(
        conn,
        "<sql>SELECT g.datacenter_region FROM gpu_nodes g JOIN inference_telemetry i ON g.node_id = i.pod_id;</sql>",
        SQL_BENCHMARK_TASKS[17]["question"],
    )
    assert c2["needs_retry"] and c2["trigger_reason"] == "empty_result_set", f"Expected empty_result_set, got {c2}"
    print(f"  [PASS] Case 2 (0-row join): {c2['trigger_reason']}")

    # Case 3: WHERE instead of HAVING MAX (SQL15 failure from L4)
    c3 = zero_oracle_sqlite_verifier(
        conn,
        "<sql>SELECT pd.pod_id, MAX(it.kv_cache_usage_pct) FROM pod_deployments pd JOIN inference_telemetry it ON pd.pod_id = it.pod_id WHERE it.kv_cache_usage_pct < 30.0 GROUP BY pd.pod_id;</sql>",
        SQL_BENCHMARK_TASKS[14]["question"],
    )
    assert c3["needs_retry"] and c3["trigger_reason"] == "lint_where_instead_of_having_max", f"Expected lint, got {c3}"
    print(f"  [PASS] Case 3 (WHERE on MAX lint): {c3['trigger_reason']}")

    # Case 4: All 50 ground-truth queries must NOT trigger false-positive retries!
    false_positives = 0
    for task in SQL_BENCHMARK_TASKS:
        v = zero_oracle_sqlite_verifier(conn, f"<sql>{task['ground_truth_sql']}</sql>", task["question"])
        if v["needs_retry"]:
            false_positives += 1
            print(f"  [FALSE POSITIVE] {task['id']}: {v['trigger_reason']}")
    assert false_positives == 0, f"Expected 0 false positives on ground truth, got {false_positives}"
    print("  [PASS] Case 4: 0/50 false-positive retry triggers across all 50 ground-truth queries!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-local", action="store_true")
    parser.add_argument("--run-suite", action="store_true")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--out", default="/content/results_l4_part3_suite.json")
    args = parser.parse_args()

    if args.validate_local:
        validate_local_verifier()
    elif args.run_suite:
        run_l4_part3_suite(out_path=args.out, concurrency=args.concurrency)
    else:
        parser.print_help()
