# Ready-to-Publish Social Media Posts (X Thread & LinkedIn Article)

---

## Option 1: LinkedIn Long-Form Post (With 3 High-Res Charts)

**Why Uncached Prefills Silently Break Your Decode SLA (and the Hidden Tradeoff of FP8 KV Cache): Stress-Testing vLLM Across 8GB & 24GB GPUs**

When enterprise teams deploy RAG or Text-to-SQL agents on self-hosted GPUs, two questions always come up during architecture reviews:
1. *"Why does our token streaming speed (TPOT) feel choppy at 30+ concurrent users when only the input prompt is long?"*
2. *"Should we always turn on `--kv-cache-dtype fp8` on Ada Lovelace / Hopper GPUs?"*

To get empirical answers, I wrote a custom asynchronous SSE profiler (`Python asyncio` + `aiohttp`) and stress-tested `vLLM` across concurrency levels (`c=1` to `c=64`) using a ~1,800-token production PostgreSQL schema prompt on both a constrained **RTX 3070 (8GB)** and an **RTX 4090 (24GB)**.

### 1. The Misconception: "Prefix Caching Only Speeds Up TTFT"
Most engineers assume Automatic Prefix Caching (`--enable-prefix-caching`) only improves Time-To-First-Token (Prefill). Our stress test proved that **missing the prefix cache also degrades Decode speed (TPOT P99) by up to 2.8x**:
- **On RTX 3070 (8GB, `1.82 GiB` KV Cache Pool) @ `c=32`**:
  - **TTFT P99** dropped **10.8x** (`3,495.0 ms` → `321.9 ms`)
  - **TPOT P99 (Decode speed)** improved **2.8x** (`47.9 ms/tok` → `17.2 ms/tok`)
  - **Output Throughput** jumped **2.83x** (`632.9 tok/s` → `1,789.4 tok/s`)
- **On RTX 4090 (24GB, `14.5 GiB` KV Cache Pool) @ `c=64`**:
  - **TTFT P99** dropped **11.8x** (`1,778.0 ms` → `150.7 ms`)
  - **TPOT P99** improved **2.2x** (`28.9 ms/tok` → `13.1 ms/tok`)
  - **Output Throughput** jumped **2.29x** (`2,011.1 tok/s` → **`4,613.9 tok/s`**)

**Why?** *Prefill-Decode Interference.* When 32–64 concurrent requests miss the prefix cache, vLLM's unified scheduler must interleave compute-bound Prefill GEMM chunks alongside memory-bandwidth-bound Decode steps. Bypassing prefill via shared physical PagedAttention KV blocks frees the GPU to run pure batched decode—pushing a single RTX 4090 to **4,614 output tokens/sec (`~$0.023 per 1M output tokens` at RunPod spot rates)**.

### 2. The Counter-Intuitive Tradeoff of `--kv-cache-dtype fp8`
On the RTX 4090 (`14.5 GiB` available for KV Cache), enabling `--kv-cache-dtype fp8` **exactly doubled KV cache capacity**:
- **FP16 KV Cache**: `542,912 tokens` (`132.55x` max concurrency @ 4k context)
- **FP8 KV Cache**: **`1,085,824 tokens`** (`265.09x` max concurrency @ 4k context — over **1 Million KV tokens** on a single 24GB GPU!)

**However**, when our `c=64` workload already fit comfortably inside the FP16 KV pool, **FP16 KV Cache delivered 33.8% higher throughput (`4,613.9 tok/s` vs `3,449.1 tok/s`) and faster TPOT (`13.1 ms` vs `19.4 ms`) than FP8 KV Cache** due to runtime FP8 quantization/dequantization scaling overhead in the attention kernel.

**Takeaway for Forward Deployed & AI Infra Engineers**: Don't blindly enable `--kv-cache-dtype fp8` when VRAM utilization is low. Turn it on specifically when your workload is VRAM-capacity-bound (high preemption rate, `32k+` long context, or squeezing a `14B/32B` model onto a single card).

### 3. Why 90% of Production Prompts Accidentally Get a 0% Cache Hit Rate
To trigger the 0% cache hit collapse in our baseline wave, I didn't disable `--enable-prefix-caching` on the server—I simply prepended `[Session-Nonce: <uuid>]` on Line 1 of the system prompt. Because vLLM hashes KV blocks sequentially in 16-token blocks, placing a dynamic `current_timestamp`, `request_id`, or `user_id` at the top of your system prompt invalidates 100% of downstream KV blocks. Always keep static DDLs and tool definitions strictly at token `0`, and append dynamic metadata at the end.

Full benchmark harness (`asyncio` SSE profiler, workload generator, and plotting scripts) is open-sourced on GitHub:
https://github.com/giordanoshai/vllm-prefix-cache-benchmark

#AIInfrastructure #vLLM #LLMInference #GPU #RunPod #ForwardDeployedEngineer #PerformanceEngineering #Python

---

## Option 1B: LinkedIn Short & Punchy Version (Yimin Signature Style — 12 Lines)

```text
Most people think vLLM Prefix Caching only speeds up the first token (TTFT).
I tested it on my 8GB RTX 3070 and 24GB RTX 4090 (1,800-token SQL schema prompt, up to 64 concurrent requests).

When I broke the prefix cache by putting a dynamic session ID on Line 1:
• TTFT P99 jumped 11.8x on the 4090 (151ms → 1,778ms) and 10.8x on the 3070 (322ms → 3,495ms).
• Per-token streaming speed (TPOT P99) also slowed down up to 2.8x (17.2ms → 47.9ms) because uncached prefill chunks steal GPU cycles from active decode batches.

I also tested --kv-cache-dtype fp8 on the 4090:
• It doubled the KV cache pool from 543K to 1.08M tokens (265x concurrency @ 4k context).
• But at c=64 (when VRAM was not full), FP16 KV cache was still 34% faster in throughput (4,614 vs 3,449 tok/s) due to FP8 dequantization overhead.

Two lessons:
1. Keep static schemas and tools at token 0; put timestamps and user IDs at the very end.
2. Turn on FP8 KV cache when you are VRAM-bound, not by default.

Code, raw JSON logs & plots: https://github.com/giordanoshai/vllm-prefix-cache-benchmark

#vLLM #GPU #LLMInference #AIInfrastructure #Python
```

---

## Option 2: X (Twitter) Technical Thread (7 Tweets)

**Tweet 1/7 (Hook + Attach `chart_3070_vs_4090_fp8_scaling.png` & `chart_prefix_caching_ttft.png`)**
Most engineers think vLLM Prefix Caching only improves Time-To-First-Token (TTFT), and that `--kv-cache-dtype fp8` is always a free speedup.

I built an async SSE profiler and stress-tested a 1,800-token SQL/RAG workload (`c=1` to `c=64`) across an 8GB RTX 3070 and a 24GB RTX 4090.

3 counter-intuitive findings from the telemetry logs 🧵👇

**Tweet 2/7 (Finding 1: Missing Prefix Cache Kills Decode TPOT by 2.8x)**
When Prefix Caching broke (0% hit rate via a Line-1 dynamic nonce):
- On RTX 3070 (`c=32`): TTFT P99 exploded **10.8x** (`322ms` → `3,495ms`), and Decode TPOT P99 degraded **2.8x** (`17.2ms` → `47.9ms/tok`, breaching 45ms SLA).
- On RTX 4090 (`c=64`): TTFT P99 jumped **11.8x** (`151ms` → `1,778ms`), and output throughput dropped from **4,614 tok/s** to **2,011 tok/s**.

Why does a prefill cache miss slow down per-token generation?
**Prefill-Decode Interference.** Interleaving 64 uncached 1,800-token prefill GEMMs steals GPU cycles from active decode batches.

**Tweet 3/7 (Finding 2: The FP8 KV Cache Tradeoff — 1.08M Tokens vs Dequant Overhead)**
On the 24GB RTX 4090 (`14.5 GiB` KV pool), enabling `--kv-cache-dtype fp8` doubled capacity:
- FP16 KV pool: `542,912 tokens` (`132.6x` @ 4k ctx)
- FP8 KV pool: **`1,085,824 tokens`** (`265.1x` @ 4k ctx — >1M tokens on one 24GB card!)

BUT at `c=64` (when not VRAM-starved), **FP16 KV was 33.8% faster in throughput (`4,614 tok/s` vs `3,449 tok/s`)** and had lower TPOT P99 (`13.1ms` vs `19.4ms`) because of FP8 attention dequantization overhead.

Rule: Use FP8 KV cache when you are VRAM-capacity-bound (long context / preemptions), not when KV usage is <50%.

**Tweet 4/7 (Finding 3: The #1 Prompt Bug That Zeroes Out Prefix Cache Hit Rate)**
How did I trigger the 0% cache hit wave without changing a single vLLM server flag?

I put `[Session-Nonce: <uuid>]` on Line 1 of `messages[0]`.
vLLM hashes KV cache blocks sequentially (16 tokens/block). Putting a dynamic timestamp or `user_id` at the top of your system prompt invalidates 100% of downstream KV blocks.
Keep static DDLs/tool schemas at token `0`; push dynamic variables to the bottom.

**Tweet 5/7 (Bonus Production Bug: vLLM V1 FlashInfer Sampler Crash Without `nvcc`)**
While spinning up `vllm 0.33.0` on a minimal driver-only node (no `/usr/local/cuda/bin/nvcc`), EngineCore crashed during `_dummy_sampler_run()` when `flashinfer-python` tried to JIT-compile `top_k_mask_logits` via Ninja.

Quick fix in minimal containers: pre-install `flashinfer-cubin` or set `VLLM_USE_FLASHINFER_SAMPLER=0` to use native PyTorch/Triton ops.

**Tweet 6/7 (Unit Economics on Cloud GPUs)**
At `c=64` with Prefix Caching hitting (`4,613.94 output tok/s` = `16.61M output tokens/hr`), a single @runpod RTX 4090 (`~$0.39/hr` Spot) delivers:
**`$0.023 per 1M output tokens`** with `150.7ms` TTFT P99 and `13.1ms` TPOT P99.

**Tweet 7/7 (Open Source Repo)**
Full benchmark suite (`asyncio` SSE profiler that properly skips the initial empty `role: assistant` delta, workload generator, raw JSON telemetry, and plotting scripts) is open-sourced here:
https://github.com/giordanoshai/vllm-prefix-cache-benchmark (@vllm_project)

---

## Option 3: Part 2 Follow-Up Post (Answering the Commenter: Speed vs. SQLite Execution Quality on Blackwell SM 12.0)

### LinkedIn Post (Short & Punchy — Yimin Signature Style)
*(Attach `charts/chart_accuracy_vs_quantization.png`)*

```text
Last week, someone asked in the comments: "You tested FP8 KV cache throughput—did you check output quality too? On L4, AWQ + FP8 KV dropped accuracy from 83.6% to 79.6%."

So I built a 50-task Text-to-SQL benchmark backed by a real in-memory SQLite database and tested Qwen2.5-7B across 4 setups on an NVIDIA RTX PRO 6000 Blackwell (vLLM 0.34.0, c=16).

What I found was way worse than a 4% drop:
• BF16 + FP16 KV (baseline): 84% valid SQL, 62% exact execution match (1,049 tok/s).
• AWQ 4-bit + FP16 KV: 80% valid SQL, 60% exact match (1,372 tok/s, +31% speed, 1.8x KV capacity).
• AWQ 4-bit + FP8 KV: fastest throughput of all at 1,448 tok/s (717K KV tokens) — and 0% valid SQL.

With --kv-cache-dtype fp8 on Blackwell (SM 12.0), vLLM returned HTTP 200 and fast streams, while emitting garbled SQL like:
"SELECT org_name, , credit_balance_usd FROM schema WHERE tier = ' '-enterprise"

I didn't stop at the first run—I tested both FlashInfer and Triton Attention, turned off prefix caching, and even tried dynamic fp8_per_token_head scaling. All of them returned 0% valid SQL on SM 12.0, while fp8_e5m2 crashed at startup.

Lesson: Never benchmark inference configs with tok/s alone. Always run a real output/execution check before turning on FP8 KV cache on new GPU architectures.

50-task SQLite verifier + raw logs: https://github.com/giordanoshai/vllm-prefix-cache-benchmark

#vLLM #LLMInference #GPU #Blackwell #AIInfrastructure #Python
```

### Reply to the Original L4 Commenter (After Posting Part 2)

```text
Great question—I followed up and built a 50-task Text-to-SQL execution verifier (in-memory SQLite) to test BF16 vs AWQ 4-bit × FP16 vs FP8 KV cache on Qwen2.5-7B (vLLM 0.34.0 on an RTX PRO 6000 Blackwell).

On FP16 KV, AWQ 4-bit behaved close to what you saw: +31% throughput (1,049 → 1,372 tok/s) for a 2-point drop in exact execution accuracy (62% → 60%).

The real surprise was FP8 KV on Blackwell (SM 12.0): AWQ + FP8 KV hit the top speed (1,448 tok/s) while silently emitting 0% valid SQL—across FlashInfer, Triton Attention, and even dynamic fp8_per_token_head.

Just wrote up the full breakdown and verifier repo here if you'd like to compare notes!
```
