"""
Workload Generator for vLLM Production Stress-Testing.

Generates realistic enterprise RAG / Text-to-SQL prompts with:
1. A ~2,000-token shared system context (Database DDL schemas + business rules + examples)
   to benchmark vLLM Automatic Prefix Caching (APC) hit rates and TTFT reduction.
2. Unique user queries (or cache-busting prefixes when testing 0% cache hit scenarios).
"""

from dataclasses import dataclass
import random
import uuid


@dataclass
class PromptRequest:
    request_id: str
    messages: list[dict[str, str]]
    max_tokens: int
    prompt_mode: str  # "shared_prefix" or "unique_prefix"


# ~1,800+ tokens of realistic Enterprise Cloud & Financial Database Schema + Guidelines
SHARED_ENTERPRISE_PREFIX = """You are a Principal Cloud Infrastructure & Financial Data Analyst Assistant.
Below is the complete production database schema, SLA policies, and query execution rules for our multi-tenant GPU cloud platform.
Always analyze the schema carefully and provide a concise, technically accurate troubleshooting plan and SQL diagnostic query.

=== SECTION 1: PRODUCTION DATABASE SCHEMA (PostgreSQL 16) ===

CREATE TABLE organizations (
    org_id UUID PRIMARY KEY,
    org_name VARCHAR(255) NOT NULL,
    tier VARCHAR(32) CHECK (tier IN ('free', 'pro', 'enterprise', 'dedicated')),
    credit_balance_usd NUMERIC(12, 4) NOT NULL DEFAULT 0.0,
    max_concurrent_pods INTEGER NOT NULL DEFAULT 5,
    region_preference VARCHAR(64) DEFAULT 'ap-southeast-1',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE gpu_nodes (
    node_id VARCHAR(64) PRIMARY KEY,
    datacenter_region VARCHAR(64) NOT NULL,
    gpu_model VARCHAR(64) NOT NULL, -- e.g., 'NVIDIA-A100-SXM4-80GB', 'NVIDIA-RTX-4090-24GB'
    gpu_count INTEGER NOT NULL,
    vram_per_gpu_gb INTEGER NOT NULL,
    interconnect_type VARCHAR(32) NOT NULL, -- 'NVLink-4', 'PCIe-4.0-x16'
    nvme_bandwidth_gbps NUMERIC(6, 2) NOT NULL,
    driver_version VARCHAR(32) NOT NULL,
    cuda_version VARCHAR(32) NOT NULL,
    node_status VARCHAR(32) CHECK (node_status IN ('healthy', 'degraded', 'cordoned', 'maintenance'))
);

CREATE TABLE pod_deployments (
    pod_id UUID PRIMARY KEY,
    org_id UUID NOT NULL REFERENCES organizations(org_id),
    node_id VARCHAR(64) NOT NULL REFERENCES gpu_nodes(node_id),
    deployment_type VARCHAR(32) CHECK (deployment_type IN ('on_demand_pod', 'spot_pod', 'serverless_worker')),
    container_image VARCHAR(512) NOT NULL,
    vllm_args JSONB NOT NULL DEFAULT '{}'::jsonb,
    allocated_gpus INTEGER NOT NULL DEFAULT 1,
    hourly_rate_usd NUMERIC(8, 4) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    terminated_at TIMESTAMPTZ,
    exit_code INTEGER,
    termination_reason VARCHAR(128) -- e.g., 'OOMKilled', 'UserStop', 'SpotPreempted', 'HealthCheckFailed'
);

CREATE TABLE inference_telemetry (
    telemetry_id BIGSERIAL PRIMARY KEY,
    pod_id UUID NOT NULL REFERENCES pod_deployments(pod_id),
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    active_concurrency INTEGER NOT NULL,
    Waiting_queue_depth INTEGER NOT NULL DEFAULT 0,
    kv_cache_usage_pct NUMERIC(5, 2) NOT NULL,
    prefix_cache_hit_rate_pct NUMERIC(5, 2) NOT NULL,
    ttft_p50_ms NUMERIC(10, 2) NOT NULL,
    ttft_p99_ms NUMERIC(10, 2) NOT NULL,
    tpot_p50_ms NUMERIC(10, 2) NOT NULL,
    tpot_p99_ms NUMERIC(10, 2) NOT NULL,
    num_preemptions_delta INTEGER NOT NULL DEFAULT 0,
    tokens_generated_delta INTEGER NOT NULL DEFAULT 0
);

=== SECTION 2: OPERATIONAL RUNBOOK & SLA THRESHOLDS ===
1. Latency SLA:
   - Interactive Chat / Agent Workloads: TTFT P99 must remain < 800ms; TPOT P99 must remain < 45ms (>22 tokens/sec).
   - Batch Offline Processing: TTFT is unconstrained; maximize total throughput (tokens/sec/dollar).
2. Memory & KV Cache Preemption Policy:
   - When `kv_cache_usage_pct` exceeds 92% and `num_preemptions_delta > 0`, vLLM evicts running sequences and recomputes Prefill later.
   - Mitigation A: Enable `--kv-cache-dtype fp8` on Ada Lovelace (RTX 4090/L40S) or Hopper (H100) GPUs to cut KV cache footprint by 50%.
   - Mitigation B: Lower `--max-num-seqs` or cap `--max-model-len` to match actual P99 prompt + completion length rather than the default 32768/131072.
   - Mitigation C: Enable `--enable-prefix-caching` when `prefix_cache_hit_rate_pct` potential is > 40% (such as multi-turn agents or RAG with static system prompts).
3. Multi-GPU Tensor Parallelism (TP) Rules:
   - Over PCIe 4.0 without P2P/NVLink, `tp_size >= 4` incurs up to 35% AllReduce synchronization overhead during Decode steps.
   - Always verify `nvidia-smi topo -m` and test `NCCL_P2P_DISABLE=1` if cross-NUMA PCIe hangs occur during model initialization.
"""

USER_QUESTIONS = [
    "Write a SQL query to find the top 5 enterprise organizations experiencing the highest rate of KV cache preemptions in the last 24 hours, and explain how to tune their vLLM arguments.",
    "Find all GPU nodes in 'ap-southeast-1' where pods were terminated with exit_code 137 (OOMKilled) more than 3 times today, and give a root-cause checklist.",
    "Calculate the effective cost per 1M generated tokens for each GPU model over the past 7 days by joining pod_deployments and inference_telemetry.",
    "Identify serverless_worker deployments where TTFT P99 exceeded 1500ms despite active_concurrency being under 8, and diagnose why cold starts or long prefills are happening.",
    "Compare the average TPOT P99 between NVLink-4 nodes and PCIe-4.0-x16 nodes for pods running with allocated_gpus >= 2.",
    "Write a query to detect pods whose kv_cache_usage_pct stays below 30% at peak concurrency, indicating over-provisioned VRAM that could be downgraded to a cheaper GPU.",
]


def build_workload(
    num_requests: int = 32,
    shared_prefix: bool = True,
    max_tokens: int = 128,
    seed: int = 42,
) -> list[PromptRequest]:
    """
    Generate a list of PromptRequest objects for stress testing.

    Args:
        num_requests: Total number of requests to generate in this wave.
        shared_prefix: If True, all requests share the exact same 1,800-token prefix
                       (simulating RAG/Agent system prompt, ideal for Prefix Caching).
                       If False, prepends a unique UUID nonce to the system prompt so
                       vLLM cannot reuse cached KV blocks (0% cache hit baseline).
        max_tokens: Maximum output tokens to generate per request.
        seed: Random seed for reproducible benchmarks.
    """
    rng = random.Random(seed)
    workload: list[PromptRequest] = []

    for i in range(num_requests):
        req_id = f"req-{i:04d}-{uuid.uuid4().hex[:6]}"
        question = rng.choice(USER_QUESTIONS)
        # Add a tiny per-request filter parameter in the user turn so outputs aren't 100% identical
        user_content = f"[Ticket #{1000 + i}] {question}"

        if shared_prefix:
            sys_content = SHARED_ENTERPRISE_PREFIX
            mode = "shared_prefix"
        else:
            # Bust the prefix cache at token 0 by injecting a unique nonce at the very start
            sys_content = f"[Session-Nonce: {uuid.uuid4()}]\n" + SHARED_ENTERPRISE_PREFIX
            mode = "unique_prefix"

        workload.append(
            PromptRequest(
                request_id=req_id,
                messages=[
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=max_tokens,
                prompt_mode=mode,
            )
        )

    return workload
