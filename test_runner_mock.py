"""
Local Mock SSE Server & Self-Test (`test_runner_mock.py`).

Simulates a vLLM OpenAI-compatible streaming endpoint:
- Simulates a 120ms Prefill delay before the first token (TTFT ~ 120ms)
- Streams 10 tokens with a 20ms delay between each token (TPOT ~ 20ms)
Run `python research-vllm-benchmark/test_runner_mock.py` to verify your
`measure_single_request()` implementation without needing a live GPU!
"""

import asyncio
import json
from aiohttp import web

from benchmark_runner import run_benchmark_wave, print_markdown_table


async def fake_vllm_chat_completions(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={"Content-Type": "text/event-stream"},
    )
    await response.prepare(request)

    # 1. Simulate Prefill delay (120ms -> expected TTFT ~ 120ms)
    await asyncio.sleep(0.120)

    # 2. Send first chunk with role only (empty content, just like real OpenAI/vLLM!)
    role_chunk = {"choices": [{"delta": {"role": "assistant", "content": ""}}]}
    await response.write(f"data: {json.dumps(role_chunk)}\n\n".encode("utf-8"))

    # 3. Stream 10 real tokens with 20ms Decode interval (expected TPOT ~ 20ms)
    for i in range(10):
        if i > 0:
            await asyncio.sleep(0.020)
        token_chunk = {"choices": [{"delta": {"content": f" tok{i}"}}]}
        await response.write(f"data: {json.dumps(token_chunk)}\n\n".encode("utf-8"))

    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


async def main() -> None:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", fake_vllm_chat_completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 18080)
    await site.start()

    print("Mock vLLM server running at http://127.0.0.1:18080/v1/chat/completions")
    print("Expected metrics: TTFT ~ 120ms, TPOT ~ 20ms, Output Tokens = 10")
    try:
        summary = await run_benchmark_wave(
            endpoint_url="http://127.0.0.1:18080/v1/chat/completions",
            model_name="mock-vllm",
            concurrency=4,
            num_requests=8,
            shared_prefix=True,
            max_tokens=16,
        )
        print_markdown_table([summary])
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
