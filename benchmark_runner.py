"""
Asynchronous vLLM Stress-Testing & Latency Profiler (`benchmark_runner.py`).

Measures production-critical FDE metrics under controlled concurrency (`asyncio.Semaphore`):
- TTFT (Time To First Token, ms): Prefill + queuing latency
- TPOT (Time Per Output Token, ms): Decode step speed
- E2E Latency (End-to-End, ms): Total request duration
- P50 / P95 / P99 Tail Latency percentiles
- System Throughput (Output tokens / sec)
"""

import argparse
import asyncio
from dataclasses import dataclass, field
import json
import statistics
import time
from typing import Optional
import aiohttp

from workload_generator import PromptRequest, build_workload


@dataclass
class RequestMetrics:
    request_id: str
    success: bool
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    e2e_ms: float = 0.0
    output_tokens: int = 0
    error_msg: Optional[str] = None
    inter_token_latencies_ms: list[float] = field(default_factory=list)


async def measure_single_request(
    session: aiohttp.ClientSession,
    endpoint_url: str,
    model_name: str,
    req: PromptRequest,
    semaphore: asyncio.Semaphore,
) -> RequestMetrics:
    """
    Send a single streaming Chat Completions request (`stream=True`) and record
    exact TTFT, TPOT, and E2E latency using `time.perf_counter()`.
    """
    payload = {
        "model": model_name,
        "messages": req.messages,
        "max_tokens": req.max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    async with semaphore:
        # =====================================================================
        # TODO (由你手写核心的 ~15 行异步流式计时逻辑):
        #
        # 目标：
        # 1. 记录请求发出的起始时间 `start_time = time.perf_counter()`
        # 2. 初始化 `first_token_time: Optional[float] = None`、
        #    `last_token_time: Optional[float] = None`、`output_tokens = 0`
        # 3. 在 `async with session.post(endpoint_url, json=payload) as resp:` 中：
        #    遍历流式返回的每一行 `async for raw_line in resp.content:`
        #    - 解析 SSE 格式（以 `data: ` 开头，遇到 `data: [DONE]` 则 `break`）
        #    - 解析 JSON 后取出 `delta_content = data["choices"][0]["delta"].get("content")`
        #    - 当 `delta_content` 非空（即真的收到了生成的 token 文本）时：
        #      a) 用 `now = time.perf_counter()` 获取当前时间戳
        #      b) 如果是收到的第 1 个非空 token (`first_token_time is None`)：
        #         记录 `first_token_time = now`，并计算 `ttft_ms = (now - start_time) * 1000.0`
        #      c) 如果不是第 1 个 token：
        #         把 `(now - last_token_time) * 1000.0` 追加进 `itl_list`
        #      d) 更新 `last_token_time = now`，并让 `output_tokens += 1`
        # 4. 循环结束后：
        #    - 计算总耗时 `e2e_ms = (time.perf_counter() - start_time) * 1000.0`
        #    - 计算平均每词延迟 `tpot_ms`（若 `output_tokens > 1`，等于 `(e2e_ms - ttft_ms) / (output_tokens - 1)`）
        #    - 返回 `RequestMetrics(...)`
        # =====================================================================
        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        last_token_time: Optional[float] = None
        output_tokens = 0
        ttft_ms = 0.0
        itl_list: list[float] = []

        try:
            async with session.post(endpoint_url, json=payload) as resp:
                if resp.status != 200:
                    err_text = await resp.text()
                    return RequestMetrics(
                        request_id=req.request_id,
                        success=False,
                        error_msg=f"HTTP {resp.status}: {err_text[:200]}",
                    )

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break

                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta_content = choices[0].get("delta", {}).get("content")

                    # 只有真正收到非空文本 token 时才触发计时（跳过首个 role-only 空包）
                    if delta_content:
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                            ttft_ms = (now - start_time) * 1000.0
                        elif last_token_time is not None:
                            itl_list.append((now - last_token_time) * 1000.0)
                        last_token_time = now
                        output_tokens += 1

            e2e_ms = (time.perf_counter() - start_time) * 1000.0
            tpot_ms = (
                (e2e_ms - ttft_ms) / (output_tokens - 1)
                if output_tokens > 1
                else 0.0
            )
            return RequestMetrics(
                request_id=req.request_id,
                success=True,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
                e2e_ms=e2e_ms,
                output_tokens=output_tokens,
                inter_token_latencies_ms=itl_list,
            )
        except Exception as exc:
            return RequestMetrics(
                request_id=req.request_id,
                success=False,
                error_msg=str(exc),
            )


def percentile(sorted_values: list[float], pct: float) -> float:
    """Compute percentile (e.g. 50, 95, 99) from a pre-sorted list of floats."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


async def run_benchmark_wave(
    endpoint_url: str,
    model_name: str,
    concurrency: int,
    num_requests: int,
    shared_prefix: bool,
    max_tokens: int,
) -> dict:
    """Run one wave of concurrent requests and aggregate P50/P95/P99 metrics."""
    workload = build_workload(
        num_requests=num_requests,
        shared_prefix=shared_prefix,
        max_tokens=max_tokens,
    )
    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=300)

    wall_start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            measure_single_request(session, endpoint_url, model_name, req, semaphore)
            for req in workload
        ]
        results: list[RequestMetrics] = await asyncio.gather(*tasks)
    wall_duration_s = time.perf_counter() - wall_start

    ok_results = [r for r in results if r.success and r.output_tokens > 0]
    failed_count = len(results) - len(ok_results)

    ttfts = sorted(r.ttft_ms for r in ok_results)
    tpots = sorted(r.tpot_ms for r in ok_results if r.output_tokens > 1)
    e2es = sorted(r.e2e_ms for r in ok_results)
    total_tokens = sum(r.output_tokens for r in ok_results)

    summary = {
        "concurrency": concurrency,
        "prompt_mode": "shared_prefix" if shared_prefix else "unique_prefix",
        "total_requests": num_requests,
        "succeeded": len(ok_results),
        "failed": failed_count,
        "wall_time_s": round(wall_duration_s, 2),
        "throughput_tok_per_s": round(total_tokens / wall_duration_s, 2) if wall_duration_s > 0 else 0.0,
        "ttft_p50_ms": round(percentile(ttfts, 50), 1),
        "ttft_p95_ms": round(percentile(ttfts, 95), 1),
        "ttft_p99_ms": round(percentile(ttfts, 99), 1),
        "tpot_p50_ms": round(percentile(tpots, 50), 1),
        "tpot_p95_ms": round(percentile(tpots, 95), 1),
        "tpot_p99_ms": round(percentile(tpots, 99), 1),
        "e2e_p99_ms": round(percentile(e2es, 99), 1),
    }
    return summary


def print_markdown_table(summaries: list[dict]) -> None:
    """Print clean GitHub/LinkedIn/X-ready Markdown table of results."""
    headers = [
        "Concurrency",
        "Mode",
        "Throughput (tok/s)",
        "TTFT P50 (ms)",
        "TTFT P99 (ms)",
        "TPOT P50 (ms)",
        "TPOT P99 (ms)",
        "E2E P99 (ms)",
    ]
    print("\n| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---:"] * len(headers)) + "|")
    for s in summaries:
        row = [
            str(s["concurrency"]),
            s["prompt_mode"],
            f"**{s['throughput_tok_per_s']}**",
            str(s["ttft_p50_ms"]),
            f"**{s['ttft_p99_ms']}**",
            str(s["tpot_p50_ms"]),
            str(s["tpot_p99_ms"]),
            str(s["e2e_p99_ms"]),
        ]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Production Stress-Testing Tool")
    parser.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--sweep",
        type=str,
        default="",
        help="Comma-separated concurrency levels to sweep, e.g. '1,2,4,8,16,32'",
    )
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--no-shared-prefix", action="store_true")
    parser.add_argument("--save-json", type=str, default="", help="Path to save results JSON")
    args = parser.parse_args()

    concurrency_levels = (
        [int(x.strip()) for x in args.sweep.split(",") if x.strip()]
        if args.sweep
        else [args.concurrency]
    )

    all_summaries: list[dict] = []
    for c in concurrency_levels:
        # Ensure at least 2xconcurrency requests so the queue stays saturated
        n_reqs = max(args.requests, c * 2)
        print(f"Running wave: concurrency={c}, total_requests={n_reqs}...")
        res = asyncio.run(
            run_benchmark_wave(
                endpoint_url=args.url,
                model_name=args.model,
                concurrency=c,
                num_requests=n_reqs,
                shared_prefix=not args.no_shared_prefix,
                max_tokens=args.max_tokens,
            )
        )
        all_summaries.append(res)

    print_markdown_table(all_summaries)
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nSaved benchmark results to {args.save_json}")
