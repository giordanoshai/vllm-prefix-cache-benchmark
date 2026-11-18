"""
Deterministic Text-to-SQL Execution Accuracy & Quantization Ablation Benchmark.

Evaluates 4 vLLM quantization setups on 50 multi-table SQLite tasks derived from
an enterprise GPU cloud & inference telemetry schema:
  1. BF16 Weights + FP16 KV Cache (`Qwen/Qwen2.5-7B-Instruct`, `--kv-cache-dtype auto`)
  2. BF16 Weights + FP8 KV Cache  (`Qwen/Qwen2.5-7B-Instruct`, `--kv-cache-dtype fp8`)
  3. AWQ 4-bit + FP16 KV Cache    (`Qwen/Qwen2.5-7B-Instruct-AWQ`, `--kv-cache-dtype auto`)
  4. AWQ 4-bit + FP8 KV Cache     (`Qwen/Qwen2.5-7B-Instruct-AWQ`, `--kv-cache-dtype fp8`)

Usage:
  # Local validation of DB seed and all 50 ground-truth queries (no GPU required)
  python research-vllm-benchmark/sql_accuracy_eval.py --validate-local

  # Full 4-way GPU ablation run (on Colab or local GPU server)
  python research-vllm-benchmark/sql_accuracy_eval.py --run-ablation --out /content/results_sql_accuracy_ablation.json
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import sqlite3
import subprocess
import sys
import time
from typing import Any, Optional

SQLITE_SCHEMA_DDL = """
CREATE TABLE organizations (
    org_id TEXT PRIMARY KEY,
    org_name TEXT NOT NULL,
    tier TEXT NOT NULL CHECK (tier IN ('free', 'pro', 'enterprise', 'dedicated')),
    credit_balance_usd REAL NOT NULL DEFAULT 0.0,
    max_concurrent_pods INTEGER NOT NULL DEFAULT 5,
    region_preference TEXT DEFAULT 'ap-southeast-1',
    created_at TEXT NOT NULL
);

CREATE TABLE gpu_nodes (
    node_id TEXT PRIMARY KEY,
    datacenter_region TEXT NOT NULL,
    gpu_model TEXT NOT NULL,
    gpu_count INTEGER NOT NULL,
    vram_per_gpu_gb INTEGER NOT NULL,
    interconnect_type TEXT NOT NULL,
    nvme_bandwidth_gbps REAL NOT NULL,
    driver_version TEXT NOT NULL,
    cuda_version TEXT NOT NULL,
    node_status TEXT NOT NULL CHECK (node_status IN ('healthy', 'degraded', 'cordoned', 'maintenance'))
);

CREATE TABLE pod_deployments (
    pod_id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL REFERENCES organizations(org_id),
    node_id TEXT NOT NULL REFERENCES gpu_nodes(node_id),
    deployment_type TEXT NOT NULL CHECK (deployment_type IN ('on_demand_pod', 'spot_pod', 'serverless_worker')),
    container_image TEXT NOT NULL,
    vllm_args TEXT NOT NULL DEFAULT '{}',
    allocated_gpus INTEGER NOT NULL DEFAULT 1,
    hourly_rate_usd REAL NOT NULL,
    started_at TEXT NOT NULL,
    terminated_at TEXT,
    exit_code INTEGER,
    termination_reason TEXT
);

CREATE TABLE inference_telemetry (
    telemetry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pod_id TEXT NOT NULL REFERENCES pod_deployments(pod_id),
    recorded_at TEXT NOT NULL,
    active_concurrency INTEGER NOT NULL,
    waiting_queue_depth INTEGER NOT NULL DEFAULT 0,
    kv_cache_usage_pct REAL NOT NULL,
    prefix_cache_hit_rate_pct REAL NOT NULL,
    ttft_p50_ms REAL NOT NULL,
    ttft_p99_ms REAL NOT NULL,
    tpot_p50_ms REAL NOT NULL,
    tpot_p99_ms REAL NOT NULL,
    num_preemptions_delta INTEGER NOT NULL DEFAULT 0,
    tokens_generated_delta INTEGER NOT NULL DEFAULT 0
);
"""

# ~1,600+ tokens shared prefix combining DDL, SQLite rules, and operational runbook
# so Prefix Caching is exercised identically to production SQL-Agent workloads.
SYSTEM_PROMPT = f"""You are a Principal Cloud Infrastructure & SQL Data Engineer.
Write an exact SQLite 3 query to answer the user's question against the production database schema below.

=== DATABASE SCHEMA (SQLite 3) ===
{SQLITE_SCHEMA_DDL}

=== OPERATIONAL RUNBOOK & DATA CONVENTIONS ===
1. Reference timestamp: Assume the current UTC timestamp is '2026-11-18T12:00:00Z'.
   - All timestamps are stored as ISO-8601 UTC strings ('YYYY-MM-DDTHH:MM:SSZ'), so string comparisons like `recorded_at >= '2026-11-17T12:00:00Z'` (last 24h) and `recorded_at >= '2026-11-11T12:00:00Z'` (last 7d) work directly.
   - "Today" means timestamps >= '2026-11-18T00:00:00Z'.
2. SLA Thresholds:
   - Interactive SLA: `ttft_p99_ms < 800.0` and `tpot_p99_ms < 45.0`.
   - Memory Pressure: `kv_cache_usage_pct > 92.0` and `num_preemptions_delta > 0` indicates sequence preemption.
   - Over-provisioned VRAM: `kv_cache_usage_pct < 30.0`.
   - OOM termination: `exit_code = 137` or `termination_reason = 'OOMKilled'`.
   - Still running pods have `terminated_at IS NULL` and `exit_code IS NULL`.
3. SQLite 3 Dialects & Division Rules:
   - Use SQLite 3 syntax ONLY (no PostgreSQL `::numeric`, `DATE_TRUNC`, `ILIKE`, or `NOW()`).
   - Integer division in SQLite truncates! Always multiply by `1.0` or `100.0` before dividing integers (e.g. `100.0 * SUM(...) / COUNT(*)`).
   - Output ONLY the SQL query wrapped in `<sql>` and `</sql>` tags. Do not include prose outside the tags. Return the exact columns and sort order specified in the question.
"""


def create_sandbox_db(seed: int = 42) -> sqlite3.Connection:
    """Create a deterministic in-memory SQLite database seeded with realistic telemetry."""
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:")
    conn.executescript(SQLITE_SCHEMA_DDL)

    tiers = ["free", "pro", "enterprise", "dedicated"]
    regions = ["ap-southeast-1", "us-east-1", "eu-west-1", "ap-northeast-2"]
    org_prefixes = [
        "Acme", "Vortex", "Kestrel", "Nova", "Helios",
        "Borealis", "Cinder", "Meridian", "Stratos", "Zenith",
        "Atlas", "Pulse", "Aether", "Cobalt", "Apex",
        "Horizon", "Quantum", "Nimbus", "Vertex", "Polaris",
    ]

    orgs = []
    for i in range(20):
        org_id = f"org-{i + 1:03d}"
        tier = tiers[i % len(tiers)]
        # Make several enterprise/dedicated orgs with varying balances
        balance = round(rng.uniform(25.0, 25000.0), 2)
        if i in (2, 7, 14):
            balance = round(rng.uniform(-50.0, 15.0), 2)
        max_pods = {"free": 2, "pro": 8, "enterprise": 32, "dedicated": 64}[tier]
        region = regions[i % len(regions)]
        created_at = f"2026-0{(i % 9) + 1}-{(i % 20) + 5:02d}T08:00:00Z"
        orgs.append((org_id, f"{org_prefixes[i]} AI", tier, balance, max_pods, region, created_at))

    conn.executemany("INSERT INTO organizations VALUES (?, ?, ?, ?, ?, ?, ?)", orgs)

    node_specs = [
        ("node-01", "ap-southeast-1", "NVIDIA-A100-SXM4-80GB", 8, 80, "NVLink-4", 7.2, "560.35", "12.6", "healthy"),
        ("node-02", "ap-southeast-1", "NVIDIA-RTX-4090-24GB", 8, 24, "PCIe-4.0-x16", 5.5, "560.35", "12.6", "healthy"),
        ("node-03", "ap-southeast-1", "NVIDIA-L4-24GB", 4, 24, "PCIe-4.0-x16", 4.1, "550.90", "12.4", "degraded"),
        ("node-04", "us-east-1", "NVIDIA-A100-SXM4-80GB", 8, 80, "NVLink-4", 7.4, "560.35", "12.6", "healthy"),
        ("node-05", "us-east-1", "NVIDIA-H100-SXM5-80GB", 8, 80, "NVLink-4", 12.0, "565.57", "12.6", "healthy"),
        ("node-06", "us-east-1", "NVIDIA-RTX-4090-24GB", 4, 24, "PCIe-4.0-x16", 5.0, "550.90", "12.4", "cordoned"),
        ("node-07", "eu-west-1", "NVIDIA-L4-24GB", 8, 24, "PCIe-4.0-x16", 4.5, "560.35", "12.6", "healthy"),
        ("node-08", "eu-west-1", "NVIDIA-A100-SXM4-80GB", 4, 80, "NVLink-4", 6.8, "560.35", "12.6", "healthy"),
        ("node-09", "ap-northeast-2", "NVIDIA-RTX-4090-24GB", 8, 24, "PCIe-4.0-x16", 5.6, "565.57", "12.6", "healthy"),
        ("node-10", "ap-northeast-2", "NVIDIA-H100-SXM5-80GB", 8, 80, "NVLink-4", 11.8, "565.57", "12.6", "maintenance"),
        ("node-11", "ap-southeast-1", "NVIDIA-A100-SXM4-80GB", 4, 80, "NVLink-4", 7.0, "560.35", "12.6", "healthy"),
        ("node-12", "us-east-1", "NVIDIA-L4-24GB", 8, 24, "PCIe-4.0-x16", 4.2, "550.90", "12.4", "healthy"),
    ]
    conn.executemany("INSERT INTO gpu_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", node_specs)

    dep_types = ["on_demand_pod", "spot_pod", "serverless_worker"]
    images = [
        "vllm/vllm-openai:v0.6.3",
        "vllm/vllm-openai:v0.7.2",
        "runpod/worker-v1-vllm:2.1.0",
        "custom/qwen-sql-agent:latest",
    ]
    rate_map = {
        "NVIDIA-RTX-4090-24GB": 0.44,
        "NVIDIA-L4-24GB": 0.36,
        "NVIDIA-A100-SXM4-80GB": 1.39,
        "NVIDIA-H100-SXM5-80GB": 2.49,
    }
    term_choices = [
        (None, None),
        (None, None),
        (0, "UserStop"),
        (137, "OOMKilled"),
        (143, "SpotPreempted"),
        (1, "HealthCheckFailed"),
    ]

    pods = []
    # Fixed timestamps anchored around 2026-11-18T12:00:00Z
    start_dates = [
        "2026-11-10T06:00:00Z",
        "2026-11-12T10:15:00Z",
        "2026-11-14T15:30:00Z",
        "2026-11-16T09:00:00Z",
        "2026-11-17T14:00:00Z",
        "2026-11-18T02:00:00Z",
        "2026-11-18T08:30:00Z",
    ]
    for i in range(100):
        pod_id = f"pod-{i + 1:03d}"
        org_id = orgs[i % len(orgs)][0]
        node = node_specs[i % len(node_specs)]
        node_id, _, gpu_model = node[0], node[1], node[2]
        dep_type = dep_types[i % len(dep_types)]
        image = images[i % len(images)]
        alloc_gpus = 1 if gpu_model in ("NVIDIA-RTX-4090-24GB", "NVIDIA-L4-24GB") and i % 3 != 0 else (2 if i % 4 != 0 else 4)
        alloc_gpus = min(alloc_gpus, node[3])
        hourly_rate = round(rate_map[gpu_model] * alloc_gpus * (0.65 if dep_type == "spot_pod" else 1.0), 4)
        vllm_args = json.dumps({
            "enable_prefix_caching": (i % 2 == 0),
            "kv_cache_dtype": "fp8" if i % 5 == 0 else "auto",
            "max_model_len": 4096 if i % 3 == 0 else 8192,
        })
        started_at = start_dates[i % len(start_dates)]

        # Ensure deterministic OOMs today in ap-southeast-1 and varied termination states
        if i in (0, 1, 12, 24, 36, 48):
            exit_code, term_reason = 137, "OOMKilled"
            terminated_at = "2026-11-18T05:30:00Z"
        else:
            exit_code, term_reason = term_choices[rng.randrange(len(term_choices))]
            if exit_code is None:
                terminated_at = None
            else:
                terminated_at = "2026-11-18T09:45:00Z" if started_at >= "2026-11-17" else "2026-11-15T18:00:00Z"

        pods.append((
            pod_id, org_id, node_id, dep_type, image, vllm_args,
            alloc_gpus, hourly_rate, started_at, terminated_at, exit_code, term_reason
        ))

    conn.executemany("INSERT INTO pod_deployments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", pods)

    telemetry = []
    rec_times = [
        "2026-11-11T10:00:00Z",  # > 7 days ago relative to 2026-11-18T12:00:00Z
        "2026-11-13T16:00:00Z",  # within 7 days
        "2026-11-15T20:00:00Z",  # within 7 days
        "2026-11-17T15:00:00Z",  # within 24h
        "2026-11-18T03:00:00Z",  # today & within 24h
        "2026-11-18T10:30:00Z",  # today & within 24h
    ]
    for i in range(400):
        pod_idx = i % len(pods)
        pod = pods[pod_idx]
        pod_id = pod[0]
        rec_at = rec_times[i % len(rec_times)]
        # Make every 10th pod genuinely over-provisioned (low KV cache across all snapshots)
        if pod_idx % 10 == 9:
            conc = [1, 4][i % 2]
            kv_pct = round(rng.uniform(10.0, 24.5), 2)
        else:
            conc = [1, 4, 8, 16, 32, 48][i % 6]
            kv_pct = round(min(99.5, 12.0 + conc * 1.8 + rng.uniform(-8.0, 14.0)), 2)
            if i % 11 == 0:
                kv_pct = round(rng.uniform(93.0, 99.2), 2)
            if i % 9 == 0:
                kv_pct = round(rng.uniform(10.0, 26.0), 2)

        prefix_hit = round(rng.uniform(60.0, 95.0) if "true" in pod[5] else rng.uniform(0.0, 15.0), 2)
        waiting = max(0, int((conc - 12) * 0.6)) if kv_pct > 75 else 0
        ttft_p50 = round(45.0 + conc * 6.5 + (200.0 if prefix_hit < 20 else 0.0) + rng.uniform(0, 80), 2)
        ttft_p99 = round(ttft_p50 * rng.uniform(1.6, 3.2), 2)
        if pod[3] == "serverless_worker" and conc < 8 and i % 4 == 0:
            ttft_p99 = round(rng.uniform(1600.0, 2900.0), 2)
        tpot_p50 = round(11.0 + conc * 0.45 + rng.uniform(0, 6.0), 2)
        tpot_p99 = round(tpot_p50 * rng.uniform(1.15, 1.55), 2)
        preemptions = rng.randint(1, 8) if kv_pct > 92.0 else 0
        tokens_gen = rng.randint(2500, 95000)

        telemetry.append((
            pod_id, rec_at, conc, waiting, kv_pct, prefix_hit,
            ttft_p50, ttft_p99, tpot_p50, tpot_p99, preemptions, tokens_gen
        ))

    conn.executemany(
        """INSERT INTO inference_telemetry (
            pod_id, recorded_at, active_concurrency, waiting_queue_depth,
            kv_cache_usage_pct, prefix_cache_hit_rate_pct,
            ttft_p50_ms, ttft_p99_ms, tpot_p50_ms, tpot_p99_ms,
            num_preemptions_delta, tokens_generated_delta
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        telemetry,
    )
    conn.commit()
    return conn


# 50 graded Text-to-SQL tasks covering Filtering/NULLs, Multi-table JOINs, CASE Ratios, and Window/CTEs
SQL_BENCHMARK_TASKS: list[dict[str, str]] = [
    # --- Category A: Filtering, NULL handling & Basic Aggregations (1-10) ---
    {
        "id": "SQL01",
        "category": "filter_null",
        "question": "List org_name and credit_balance_usd for all 'enterprise' organizations whose credit_balance_usd is less than 5000, ordered by credit_balance_usd ASC, org_name ASC.",
        "ground_truth_sql": """SELECT org_name, credit_balance_usd FROM organizations WHERE tier = 'enterprise' AND credit_balance_usd < 5000 ORDER BY credit_balance_usd ASC, org_name ASC;""",
    },
    {
        "id": "SQL02",
        "category": "filter_null",
        "question": "Find node_id and datacenter_region of all 'healthy' gpu_nodes that use 'NVLink-4' interconnect and have nvme_bandwidth_gbps >= 7.0, ordered by node_id ASC.",
        "ground_truth_sql": """SELECT node_id, datacenter_region FROM gpu_nodes WHERE node_status = 'healthy' AND interconnect_type = 'NVLink-4' AND nvme_bandwidth_gbps >= 7.0 ORDER BY node_id ASC;""",
    },
    {
        "id": "SQL03",
        "category": "filter_null",
        "question": "Count how many pod deployments are currently still running (terminated_at IS NULL) grouped by deployment_type. Return deployment_type and running_count, ordered by running_count DESC, deployment_type ASC.",
        "ground_truth_sql": """SELECT deployment_type, COUNT(*) AS running_count FROM pod_deployments WHERE terminated_at IS NULL GROUP BY deployment_type ORDER BY running_count DESC, deployment_type ASC;""",
    },
    {
        "id": "SQL04",
        "category": "filter_null",
        "question": "List pod_id and COALESCE(termination_reason, 'STILL_RUNNING') AS status_label for all pods belonging to org_id 'org-001', ordered by pod_id ASC.",
        "ground_truth_sql": """SELECT pod_id, COALESCE(termination_reason, 'STILL_RUNNING') AS status_label FROM pod_deployments WHERE org_id = 'org-001' ORDER BY pod_id ASC;""",
    },
    {
        "id": "SQL05",
        "category": "filter_null",
        "question": "Find distinct container_image values used by 'spot_pod' deployments that were terminated with exit_code = 137, ordered by container_image ASC.",
        "ground_truth_sql": """SELECT DISTINCT container_image FROM pod_deployments WHERE deployment_type = 'spot_pod' AND exit_code = 137 ORDER BY container_image ASC;""",
    },
    {
        "id": "SQL06",
        "category": "filter_null",
        "question": "Calculate the total VRAM capacity (gpu_count * vram_per_gpu_gb) for each datacenter_region across all gpu_nodes whose node_status != 'maintenance'. Return datacenter_region and total_vram_gb, ordered by total_vram_gb DESC, datacenter_region ASC.",
        "ground_truth_sql": """SELECT datacenter_region, SUM(gpu_count * vram_per_gpu_gb) AS total_vram_gb FROM gpu_nodes WHERE node_status != 'maintenance' GROUP BY datacenter_region ORDER BY total_vram_gb DESC, datacenter_region ASC;""",
    },
    {
        "id": "SQL07",
        "category": "filter_null",
        "question": "Find the top 5 telemetry records by highest ttft_p99_ms where waiting_queue_depth = 0. Return telemetry_id, pod_id, and ttft_p99_ms, ordered by ttft_p99_ms DESC, telemetry_id ASC LIMIT 5.",
        "ground_truth_sql": """SELECT telemetry_id, pod_id, ttft_p99_ms FROM inference_telemetry WHERE waiting_queue_depth = 0 ORDER BY ttft_p99_ms DESC, telemetry_id ASC LIMIT 5;""",
    },
    {
        "id": "SQL08",
        "category": "filter_null",
        "question": "For each tier in organizations, compute the minimum, average, and maximum credit_balance_usd. Return tier, min_balance, avg_balance, max_balance, ordered by tier ASC.",
        "ground_truth_sql": """SELECT tier, MIN(credit_balance_usd) AS min_balance, AVG(credit_balance_usd) AS avg_balance, MAX(credit_balance_usd) AS max_balance FROM organizations GROUP BY tier ORDER BY tier ASC;""",
    },
    {
        "id": "SQL09",
        "category": "filter_null",
        "question": "Find pod_id and hourly_rate_usd for all running pods (exit_code IS NULL) with allocated_gpus >= 2 and hourly_rate_usd > 2.0, ordered by hourly_rate_usd DESC, pod_id ASC.",
        "ground_truth_sql": """SELECT pod_id, hourly_rate_usd FROM pod_deployments WHERE exit_code IS NULL AND allocated_gpus >= 2 AND hourly_rate_usd > 2.0 ORDER BY hourly_rate_usd DESC, pod_id ASC;""",
    },
    {
        "id": "SQL10",
        "category": "filter_null",
        "question": "Count total telemetry snapshots and total tokens_generated_delta per active_concurrency level across all records from the last 24 hours (recorded_at >= '2026-11-17T12:00:00Z'). Return active_concurrency, snapshot_count, total_tokens, ordered by active_concurrency ASC.",
        "ground_truth_sql": """SELECT active_concurrency, COUNT(*) AS snapshot_count, SUM(tokens_generated_delta) AS total_tokens FROM inference_telemetry WHERE recorded_at >= '2026-11-17T12:00:00Z' GROUP BY active_concurrency ORDER BY active_concurrency ASC;""",
    },

    # --- Category B: Multi-table JOINs & GROUP BY HAVING (11-25) ---
    {
        "id": "SQL11",
        "category": "join_having",
        "question": "Find the top 5 organizations (any tier) with the highest total num_preemptions_delta in the last 24 hours (recorded_at >= '2026-11-17T12:00:00Z'). Return org_name and total_preemptions, ordered by total_preemptions DESC, org_name ASC LIMIT 5.",
        "ground_truth_sql": """SELECT o.org_name, SUM(t.num_preemptions_delta) AS total_preemptions FROM organizations o JOIN pod_deployments p ON o.org_id = p.org_id JOIN inference_telemetry t ON p.pod_id = t.pod_id WHERE t.recorded_at >= '2026-11-17T12:00:00Z' GROUP BY o.org_id, o.org_name ORDER BY total_preemptions DESC, o.org_name ASC LIMIT 5;""",
    },
    {
        "id": "SQL12",
        "category": "join_having",
        "question": "Find all GPU nodes in 'ap-southeast-1' where pods terminated today (terminated_at >= '2026-11-18T00:00:00Z') with exit_code = 137 at least 2 times. Return node_id and oom_count, ordered by oom_count DESC, node_id ASC.",
        "ground_truth_sql": """SELECT g.node_id, COUNT(*) AS oom_count FROM gpu_nodes g JOIN pod_deployments p ON g.node_id = p.node_id WHERE g.datacenter_region = 'ap-southeast-1' AND p.exit_code = 137 AND p.terminated_at >= '2026-11-18T00:00:00Z' GROUP BY g.node_id HAVING COUNT(*) >= 2 ORDER BY oom_count DESC, g.node_id ASC;""",
    },
    {
        "id": "SQL13",
        "category": "join_having",
        "question": "Compare the average tpot_p99_ms between each interconnect_type for pods running with allocated_gpus >= 2. Return interconnect_type and avg_tpot_p99_ms, ordered by avg_tpot_p99_ms ASC.",
        "ground_truth_sql": """SELECT g.interconnect_type, AVG(t.tpot_p99_ms) AS avg_tpot_p99_ms FROM gpu_nodes g JOIN pod_deployments p ON g.node_id = p.node_id JOIN inference_telemetry t ON p.pod_id = t.pod_id WHERE p.allocated_gpus >= 2 GROUP BY g.interconnect_type ORDER BY avg_tpot_p99_ms ASC;""",
    },
    {
        "id": "SQL14",
        "category": "join_having",
        "question": "Identify serverless_worker deployments that experienced ttft_p99_ms > 1500.0 while active_concurrency < 8. Return distinct p.pod_id and o.org_name, ordered by p.pod_id ASC.",
        "ground_truth_sql": """SELECT DISTINCT p.pod_id, o.org_name FROM pod_deployments p JOIN organizations o ON p.org_id = o.org_id JOIN inference_telemetry t ON p.pod_id = t.pod_id WHERE p.deployment_type = 'serverless_worker' AND t.ttft_p99_ms > 1500.0 AND t.active_concurrency < 8 ORDER BY p.pod_id ASC;""",
    },
    {
        "id": "SQL15",
        "category": "join_having",
        "question": "Find pods whose maximum kv_cache_usage_pct across all telemetry snapshots is strictly below 30.0 (indicating over-provisioned VRAM). Return pod_id, gpu_model, and max_kv_pct, ordered by max_kv_pct ASC, pod_id ASC.",
        "ground_truth_sql": """SELECT p.pod_id, g.gpu_model, MAX(t.kv_cache_usage_pct) AS max_kv_pct FROM pod_deployments p JOIN gpu_nodes g ON p.node_id = g.node_id JOIN inference_telemetry t ON p.pod_id = t.pod_id GROUP BY p.pod_id, g.gpu_model HAVING MAX(t.kv_cache_usage_pct) < 30.0 ORDER BY max_kv_pct ASC, p.pod_id ASC;""",
    },
    {
        "id": "SQL16",
        "category": "join_having",
        "question": "For each gpu_model, calculate total generated tokens (SUM(tokens_generated_delta)) over the last 7 days (recorded_at >= '2026-11-11T12:00:00Z') and the average pod hourly_rate_usd of pods that emitted those telemetry rows. Return gpu_model, total_tokens, avg_hourly_rate, ordered by total_tokens DESC.",
        "ground_truth_sql": """SELECT g.gpu_model, SUM(t.tokens_generated_delta) AS total_tokens, AVG(p.hourly_rate_usd) AS avg_hourly_rate FROM gpu_nodes g JOIN pod_deployments p ON g.node_id = p.node_id JOIN inference_telemetry t ON p.pod_id = t.pod_id WHERE t.recorded_at >= '2026-11-11T12:00:00Z' GROUP BY g.gpu_model ORDER BY total_tokens DESC;""",
    },
    {
        "id": "SQL17",
        "category": "join_having",
        "question": "List organizations whose total hourly spend across currently running pods (terminated_at IS NULL) exceeds 5.0 USD. Return org_name and active_hourly_spend, ordered by active_hourly_spend DESC, org_name ASC.",
        "ground_truth_sql": """SELECT o.org_name, SUM(p.hourly_rate_usd) AS active_hourly_spend FROM organizations o JOIN pod_deployments p ON o.org_id = p.org_id WHERE p.terminated_at IS NULL GROUP BY o.org_id, o.org_name HAVING SUM(p.hourly_rate_usd) > 5.0 ORDER BY active_hourly_spend DESC, o.org_name ASC;""",
    },
    {
        "id": "SQL18",
        "category": "join_having",
        "question": "Find all datacenter_region values that had at least 10 total KV cache preemptions (SUM(num_preemptions_delta) >= 10) in the last 7 days (recorded_at >= '2026-11-11T12:00:00Z'). Return datacenter_region and total_preemptions, ordered by total_preemptions DESC.",
        "ground_truth_sql": """SELECT g.datacenter_region, SUM(t.num_preemptions_delta) AS total_preemptions FROM gpu_nodes g JOIN pod_deployments p ON g.node_id = p.node_id JOIN inference_telemetry t ON p.pod_id = t.pod_id WHERE t.recorded_at >= '2026-11-11T12:00:00Z' GROUP BY g.datacenter_region HAVING SUM(t.num_preemptions_delta) >= 10 ORDER BY total_preemptions DESC;""",
    },
    {
        "id": "SQL19",
        "category": "join_having",
        "question": "For each container_image, compute the average prefix_cache_hit_rate_pct and average ttft_p50_ms across telemetry rows where active_concurrency >= 16. Return container_image, avg_hit_rate, avg_ttft_p50, ordered by avg_ttft_p50 ASC.",
        "ground_truth_sql": """SELECT p.container_image, AVG(t.prefix_cache_hit_rate_pct) AS avg_hit_rate, AVG(t.ttft_p50_ms) AS avg_ttft_p50 FROM pod_deployments p JOIN inference_telemetry t ON p.pod_id = t.pod_id WHERE t.active_concurrency >= 16 GROUP BY p.container_image ORDER BY avg_ttft_p50 ASC;""",
    },
    {
        "id": "SQL20",
        "category": "join_having",
        "question": "List each organization tier along with the count of distinct pods and total tokens generated in the last 24 hours (recorded_at >= '2026-11-17T12:00:00Z'). Return tier, active_pods, tokens_24h, ordered by tokens_24h DESC.",
        "ground_truth_sql": """SELECT o.tier, COUNT(DISTINCT p.pod_id) AS active_pods, SUM(t.tokens_generated_delta) AS tokens_24h FROM organizations o JOIN pod_deployments p ON o.org_id = p.org_id JOIN inference_telemetry t ON p.pod_id = t.pod_id WHERE t.recorded_at >= '2026-11-17T12:00:00Z' GROUP BY o.tier ORDER BY tokens_24h DESC;""",
    },
    {
        "id": "SQL21",
        "category": "join_having",
        "question": "Find all pods deployed on nodes whose node_status is 'degraded' or 'cordoned' that are still running (terminated_at IS NULL). Return p.pod_id, g.node_id, g.node_status, and o.org_name, ordered by p.pod_id ASC.",
        "ground_truth_sql": """SELECT p.pod_id, g.node_id, g.node_status, o.org_name FROM pod_deployments p JOIN gpu_nodes g ON p.node_id = g.node_id JOIN organizations o ON p.org_id = o.org_id WHERE g.node_status IN ('degraded', 'cordoned') AND p.terminated_at IS NULL ORDER BY p.pod_id ASC;""",
    },
    {
        "id": "SQL22",
        "category": "join_having",
        "question": "Find GPU models where average waiting_queue_depth exceeds 3.0 when active_concurrency >= 16. Return gpu_model and avg_wait_depth, ordered by avg_wait_depth DESC, gpu_model ASC.",
        "ground_truth_sql": """SELECT g.gpu_model, AVG(t.waiting_queue_depth) AS avg_wait_depth FROM gpu_nodes g JOIN pod_deployments p ON g.node_id = p.node_id JOIN inference_telemetry t ON p.pod_id = t.pod_id WHERE t.active_concurrency >= 16 GROUP BY g.gpu_model HAVING AVG(t.waiting_queue_depth) > 3.0 ORDER BY avg_wait_depth DESC, g.gpu_model ASC;""",
    },
    {
        "id": "SQL23",
        "category": "join_having",
        "question": "For organizations whose region_preference differs from the datacenter_region of the node their pod was placed on, count how many cross-region pods each organization has. Return org_name and cross_region_pods, ordered by cross_region_pods DESC, org_name ASC.",
        "ground_truth_sql": """SELECT o.org_name, COUNT(*) AS cross_region_pods FROM organizations o JOIN pod_deployments p ON o.org_id = p.org_id JOIN gpu_nodes g ON p.node_id = g.node_id WHERE o.region_preference != g.datacenter_region GROUP BY o.org_id, o.org_name ORDER BY cross_region_pods DESC, o.org_name ASC;""",
    },
    {
        "id": "SQL24",
        "category": "join_having",
        "question": "Find the top 5 pods by highest average ttft_p99_ms across all their telemetry records, excluding pods with fewer than 3 telemetry records. Return pod_id, record_count, and avg_ttft_p99, ordered by avg_ttft_p99 DESC, pod_id ASC LIMIT 5.",
        "ground_truth_sql": """SELECT pod_id, COUNT(*) AS record_count, AVG(ttft_p99_ms) AS avg_ttft_p99 FROM inference_telemetry GROUP BY pod_id HAVING COUNT(*) >= 3 ORDER BY avg_ttft_p99 DESC, pod_id ASC LIMIT 5;""",
    },
    {
        "id": "SQL25",
        "category": "join_having",
        "question": "List all gpu_nodes and the total number of allocated_gpus currently occupied by running pods (terminated_at IS NULL) on each node (include nodes with 0 occupied GPUs via LEFT JOIN). Return node_id, gpu_count, and occupied_gpus, ordered by occupied_gpus DESC, node_id ASC.",
        "ground_truth_sql": """SELECT g.node_id, g.gpu_count, COALESCE(SUM(p.allocated_gpus), 0) AS occupied_gpus FROM gpu_nodes g LEFT JOIN pod_deployments p ON g.node_id = p.node_id AND p.terminated_at IS NULL GROUP BY g.node_id, g.gpu_count ORDER BY occupied_gpus DESC, g.node_id ASC;""",
    },

    # --- Category C: Conditional Aggregation (CASE WHEN) & Percentages (26-35) ---
    {
        "id": "SQL26",
        "category": "case_ratio",
        "question": "For each deployment_type, calculate total pods and the percentage of pods terminated by OOMKilled (exit_code = 137). Return deployment_type, total_pods, and oom_pct (0 to 100 scale), ordered by oom_pct DESC, deployment_type ASC.",
        "ground_truth_sql": """SELECT deployment_type, COUNT(*) AS total_pods, 100.0 * SUM(CASE WHEN exit_code = 137 THEN 1 ELSE 0 END) / COUNT(*) AS oom_pct FROM pod_deployments GROUP BY deployment_type ORDER BY oom_pct DESC, deployment_type ASC;""",
    },
    {
        "id": "SQL27",
        "category": "case_ratio",
        "question": "For each active_concurrency level, compute the interactive SLA violation rate (%) where a violation is defined as ttft_p99_ms >= 800.0 OR tpot_p99_ms >= 45.0. Return active_concurrency, total_snapshots, and sla_violation_pct, ordered by active_concurrency ASC.",
        "ground_truth_sql": """SELECT active_concurrency, COUNT(*) AS total_snapshots, 100.0 * SUM(CASE WHEN ttft_p99_ms >= 800.0 OR tpot_p99_ms >= 45.0 THEN 1 ELSE 0 END) / COUNT(*) AS sla_violation_pct FROM inference_telemetry GROUP BY active_concurrency ORDER BY active_concurrency ASC;""",
    },
    {
        "id": "SQL28",
        "category": "case_ratio",
        "question": "Compare average ttft_p99_ms and average tpot_p99_ms between high-cache-hit telemetry (prefix_cache_hit_rate_pct >= 50.0) and low-cache-hit telemetry (prefix_cache_hit_rate_pct < 50.0) at active_concurrency >= 16. Return cache_bucket ('high_hit' or 'low_hit'), avg_ttft_p99, avg_tpot_p99, ordered by cache_bucket ASC.",
        "ground_truth_sql": """SELECT CASE WHEN prefix_cache_hit_rate_pct >= 50.0 THEN 'high_hit' ELSE 'low_hit' END AS cache_bucket, AVG(ttft_p99_ms) AS avg_ttft_p99, AVG(tpot_p99_ms) AS avg_tpot_p99 FROM inference_telemetry WHERE active_concurrency >= 16 GROUP BY cache_bucket ORDER BY cache_bucket ASC;""",
    },
    {
        "id": "SQL29",
        "category": "case_ratio",
        "question": "For each datacenter_region, count how many telemetry snapshots occurred under memory pressure (kv_cache_usage_pct > 92.0 AND num_preemptions_delta > 0) vs normal operation. Return datacenter_region, pressure_count, total_count, ordered by pressure_count DESC, datacenter_region ASC.",
        "ground_truth_sql": """SELECT g.datacenter_region, SUM(CASE WHEN t.kv_cache_usage_pct > 92.0 AND t.num_preemptions_delta > 0 THEN 1 ELSE 0 END) AS pressure_count, COUNT(*) AS total_count FROM gpu_nodes g JOIN pod_deployments p ON g.node_id = p.node_id JOIN inference_telemetry t ON p.pod_id = t.pod_id GROUP BY g.datacenter_region ORDER BY pressure_count DESC, g.datacenter_region ASC;""",
    },
    {
        "id": "SQL30",
        "category": "case_ratio",
        "question": "Group organizations by tier and calculate the ratio of spot_pod deployments to total deployments (as spot_pct from 0 to 100). Return tier, total_deployments, spot_pct, ordered by spot_pct DESC, tier ASC.",
        "ground_truth_sql": """SELECT o.tier, COUNT(*) AS total_deployments, 100.0 * SUM(CASE WHEN p.deployment_type = 'spot_pod' THEN 1 ELSE 0 END) / COUNT(*) AS spot_pct FROM organizations o JOIN pod_deployments p ON o.org_id = p.org_id GROUP BY o.tier ORDER BY spot_pct DESC, o.tier ASC;""",
    },
    {
        "id": "SQL31",
        "category": "case_ratio",
        "question": "Using json_extract(vllm_args, '$.enable_prefix_caching') on pod_deployments, compare the average prefix_cache_hit_rate_pct and average ttft_p50_ms between pods with prefix caching enabled (1) vs disabled (0). Return prefix_caching_enabled, avg_hit_rate, avg_ttft_p50, ordered by prefix_caching_enabled DESC.",
        "ground_truth_sql": """SELECT json_extract(p.vllm_args, '$.enable_prefix_caching') AS prefix_caching_enabled, AVG(t.prefix_cache_hit_rate_pct) AS avg_hit_rate, AVG(t.ttft_p50_ms) AS avg_ttft_p50 FROM pod_deployments p JOIN inference_telemetry t ON p.pod_id = t.pod_id GROUP BY prefix_caching_enabled ORDER BY prefix_caching_enabled DESC;""",
    },
    {
        "id": "SQL32",
        "category": "case_ratio",
        "question": "Using json_extract(vllm_args, '$.kv_cache_dtype') on pod_deployments, compute the average kv_cache_usage_pct and total num_preemptions_delta per kv_cache_dtype across all telemetry records. Return kv_cache_dtype, avg_kv_usage, total_preemptions, ordered by kv_cache_dtype ASC.",
        "ground_truth_sql": """SELECT json_extract(p.vllm_args, '$.kv_cache_dtype') AS kv_cache_dtype, AVG(t.kv_cache_usage_pct) AS avg_kv_usage, SUM(t.num_preemptions_delta) AS total_preemptions FROM pod_deployments p JOIN inference_telemetry t ON p.pod_id = t.pod_id GROUP BY kv_cache_dtype ORDER BY kv_cache_dtype ASC;""",
    },
    {
        "id": "SQL33",
        "category": "case_ratio",
        "question": "For each gpu_model, compute the count of telemetry snapshots where waiting_queue_depth > 0 and the percentage of total snapshots with queuing (queuing_pct). Return gpu_model, queued_snapshots, queuing_pct, ordered by queuing_pct DESC, gpu_model ASC.",
        "ground_truth_sql": """SELECT g.gpu_model, SUM(CASE WHEN t.waiting_queue_depth > 0 THEN 1 ELSE 0 END) AS queued_snapshots, 100.0 * SUM(CASE WHEN t.waiting_queue_depth > 0 THEN 1 ELSE 0 END) / COUNT(*) AS queuing_pct FROM gpu_nodes g JOIN pod_deployments p ON g.node_id = p.node_id JOIN inference_telemetry t ON p.pod_id = t.pod_id GROUP BY g.gpu_model ORDER BY queuing_pct DESC, g.gpu_model ASC;""",
    },
    {
        "id": "SQL34",
        "category": "case_ratio",
        "question": "Calculate for each pod deployment_type: count of pods still running, count exited cleanly (exit_code = 0), and count failed/preempted (exit_code > 0). Return deployment_type, running_cnt, clean_exit_cnt, error_exit_cnt, ordered by deployment_type ASC.",
        "ground_truth_sql": """SELECT deployment_type, SUM(CASE WHEN exit_code IS NULL THEN 1 ELSE 0 END) AS running_cnt, SUM(CASE WHEN exit_code = 0 THEN 1 ELSE 0 END) AS clean_exit_cnt, SUM(CASE WHEN exit_code > 0 THEN 1 ELSE 0 END) AS error_exit_cnt FROM pod_deployments GROUP BY deployment_type ORDER BY deployment_type ASC;""",
    },
    {
        "id": "SQL35",
        "category": "case_ratio",
        "question": "For each region_preference in organizations, find how many orgs have negative or low credit (< 100 USD) vs healthy credit (>= 100 USD). Return region_preference, low_credit_orgs, healthy_credit_orgs, ordered by region_preference ASC.",
        "ground_truth_sql": """SELECT region_preference, SUM(CASE WHEN credit_balance_usd < 100.0 THEN 1 ELSE 0 END) AS low_credit_orgs, SUM(CASE WHEN credit_balance_usd >= 100.0 THEN 1 ELSE 0 END) AS healthy_credit_orgs FROM organizations GROUP BY region_preference ORDER BY region_preference ASC;""",
    },

    # --- Category D: Window Functions & CTEs / Subqueries (36-50) ---
    {
        "id": "SQL36",
        "category": "window_cte",
        "question": "Rank organizations within each tier by their total tokens_generated_delta across all time using DENSE_RANK(). Return tier, org_name, total_tokens, and tier_rank for the top 2 ranks in each tier, ordered by tier ASC, tier_rank ASC, org_name ASC.",
        "ground_truth_sql": """WITH org_tokens AS (
    SELECT o.tier, o.org_name, SUM(t.tokens_generated_delta) AS total_tokens
    FROM organizations o
    JOIN pod_deployments p ON o.org_id = p.org_id
    JOIN inference_telemetry t ON p.pod_id = t.pod_id
    GROUP BY o.org_id, o.tier, o.org_name
), ranked AS (
    SELECT tier, org_name, total_tokens,
           DENSE_RANK() OVER (PARTITION BY tier ORDER BY total_tokens DESC) AS tier_rank
    FROM org_tokens
)
SELECT tier, org_name, total_tokens, tier_rank
FROM ranked
WHERE tier_rank <= 2
ORDER BY tier ASC, tier_rank ASC, org_name ASC;""",
    },
    {
        "id": "SQL37",
        "category": "window_cte",
        "question": "Find the single worst (highest ttft_p99_ms, tie-break telemetry_id ASC) telemetry snapshot for each gpu_model using ROW_NUMBER(). Return gpu_model, pod_id, ttft_p99_ms, and active_concurrency, ordered by ttft_p99_ms DESC.",
        "ground_truth_sql": """WITH ranked AS (
    SELECT g.gpu_model, t.pod_id, t.ttft_p99_ms, t.active_concurrency,
           ROW_NUMBER() OVER (PARTITION BY g.gpu_model ORDER BY t.ttft_p99_ms DESC, t.telemetry_id ASC) AS rn
    FROM inference_telemetry t
    JOIN pod_deployments p ON t.pod_id = p.pod_id
    JOIN gpu_nodes g ON p.node_id = g.node_id
)
SELECT gpu_model, pod_id, ttft_p99_ms, active_concurrency
FROM ranked
WHERE rn = 1
ORDER BY ttft_p99_ms DESC;""",
    },
    {
        "id": "SQL38",
        "category": "window_cte",
        "question": "Find all organizations whose total pod hourly spend (sum of hourly_rate_usd across all their pods) is strictly above the average total hourly spend across all 20 organizations. Return org_name and total_hourly_spend, ordered by total_hourly_spend DESC, org_name ASC.",
        "ground_truth_sql": """WITH org_spend AS (
    SELECT o.org_id, o.org_name, COALESCE(SUM(p.hourly_rate_usd), 0.0) AS total_hourly_spend
    FROM organizations o
    LEFT JOIN pod_deployments p ON o.org_id = p.org_id
    GROUP BY o.org_id, o.org_name
)
SELECT org_name, total_hourly_spend
FROM org_spend
WHERE total_hourly_spend > (SELECT AVG(total_hourly_spend) FROM org_spend)
ORDER BY total_hourly_spend DESC, org_name ASC;""",
    },
    {
        "id": "SQL39",
        "category": "window_cte",
        "question": "Compute each datacenter_region's share (%) of total tokens_generated_delta across the entire platform using a window function SUM() OVER (). Return datacenter_region, region_tokens, and pct_share, ordered by pct_share DESC.",
        "ground_truth_sql": """WITH region_totals AS (
    SELECT g.datacenter_region, SUM(t.tokens_generated_delta) AS region_tokens
    FROM gpu_nodes g
    JOIN pod_deployments p ON g.node_id = p.node_id
    JOIN inference_telemetry t ON p.pod_id = t.pod_id
    GROUP BY g.datacenter_region
)
SELECT datacenter_region, region_tokens,
       100.0 * region_tokens / SUM(region_tokens) OVER () AS pct_share
FROM region_totals
ORDER BY pct_share DESC;""",
    },
    {
        "id": "SQL40",
        "category": "window_cte",
        "question": "For each pod, calculate its average tpot_p99_ms and compare it against the average tpot_p99_ms of all pods on the same gpu_model. Return pod_id, gpu_model, pod_avg_tpot, and model_avg_tpot for the top 10 slowest pods relative to their GPU model (pod_avg_tpot - model_avg_tpot DESC, pod_id ASC).",
        "ground_truth_sql": """WITH pod_avgs AS (
    SELECT p.pod_id, g.gpu_model, AVG(t.tpot_p99_ms) AS pod_avg_tpot
    FROM pod_deployments p
    JOIN gpu_nodes g ON p.node_id = g.node_id
    JOIN inference_telemetry t ON p.pod_id = t.pod_id
    GROUP BY p.pod_id, g.gpu_model
), with_model_avg AS (
    SELECT pod_id, gpu_model, pod_avg_tpot,
           AVG(pod_avg_tpot) OVER (PARTITION BY gpu_model) AS model_avg_tpot
    FROM pod_avgs
)
SELECT pod_id, gpu_model, pod_avg_tpot, model_avg_tpot
FROM with_model_avg
ORDER BY (pod_avg_tpot - model_avg_tpot) DESC, pod_id ASC
LIMIT 10;""",
    },
    {
        "id": "SQL41",
        "category": "window_cte",
        "question": "Identify the most recent telemetry snapshot (highest recorded_at, tie-break telemetry_id DESC) for each currently running pod (terminated_at IS NULL) where kv_cache_usage_pct > 80.0. Return pod_id, recorded_at, kv_cache_usage_pct, ordered by kv_cache_usage_pct DESC, pod_id ASC.",
        "ground_truth_sql": """WITH latest AS (
    SELECT t.pod_id, t.recorded_at, t.kv_cache_usage_pct,
           ROW_NUMBER() OVER (PARTITION BY t.pod_id ORDER BY t.recorded_at DESC, t.telemetry_id DESC) AS rn
    FROM inference_telemetry t
    JOIN pod_deployments p ON t.pod_id = p.pod_id
    WHERE p.terminated_at IS NULL
)
SELECT pod_id, recorded_at, kv_cache_usage_pct
FROM latest
WHERE rn = 1 AND kv_cache_usage_pct > 80.0
ORDER BY kv_cache_usage_pct DESC, pod_id ASC;""",
    },
    {
        "id": "SQL42",
        "category": "window_cte",
        "question": "Using a CTE, compute for each gpu_model over the past 7 days (recorded_at >= '2026-11-11T12:00:00Z'): total tokens generated, sum of pod hourly_rate_usd per telemetry pod, and effective cost per 1M tokens defined as (SUM(p.hourly_rate_usd) * 1000000.0 / SUM(t.tokens_generated_delta)). Return gpu_model, total_tokens, and cost_index_per_1m, ordered by cost_index_per_1m ASC.",
        "ground_truth_sql": """SELECT g.gpu_model, SUM(t.tokens_generated_delta) AS total_tokens,
       (SUM(p.hourly_rate_usd) * 1000000.0 / SUM(t.tokens_generated_delta)) AS cost_index_per_1m
FROM gpu_nodes g
JOIN pod_deployments p ON g.node_id = p.node_id
JOIN inference_telemetry t ON p.pod_id = t.pod_id
WHERE t.recorded_at >= '2026-11-11T12:00:00Z'
GROUP BY g.gpu_model
ORDER BY cost_index_per_1m ASC;""",
    },
    {
        "id": "SQL43",
        "category": "window_cte",
        "question": "Find organizations that have experienced BOTH at least one OOMKilled pod (exit_code = 137) AND at least one SpotPreempted pod (exit_code = 143). Return org_id and org_name, ordered by org_id ASC.",
        "ground_truth_sql": """SELECT o.org_id, o.org_name
FROM organizations o
WHERE EXISTS (SELECT 1 FROM pod_deployments p1 WHERE p1.org_id = o.org_id AND p1.exit_code = 137)
  AND EXISTS (SELECT 1 FROM pod_deployments p2 WHERE p2.org_id = o.org_id AND p2.exit_code = 143)
ORDER BY o.org_id ASC;""",
    },
    {
        "id": "SQL44",
        "category": "window_cte",
        "question": "Find the top 3 GPU nodes with the highest cumulative tokens_generated_delta across all telemetry rows, along with their rank using RANK(). Return node_id, gpu_model, total_tokens, and node_rank, ordered by node_rank ASC, node_id ASC LIMIT 3.",
        "ground_truth_sql": """WITH node_tokens AS (
    SELECT g.node_id, g.gpu_model, SUM(t.tokens_generated_delta) AS total_tokens
    FROM gpu_nodes g
    JOIN pod_deployments p ON g.node_id = p.node_id
    JOIN inference_telemetry t ON p.pod_id = t.pod_id
    GROUP BY g.node_id, g.gpu_model
)
SELECT node_id, gpu_model, total_tokens,
       RANK() OVER (ORDER BY total_tokens DESC) AS node_rank
FROM node_tokens
ORDER BY node_rank ASC, node_id ASC
LIMIT 3;""",
    },
    {
        "id": "SQL45",
        "category": "window_cte",
        "question": "Calculate the cumulative running total of tokens_generated_delta across the distinct telemetry timestamps (recorded_at) in ascending chronological order. Return recorded_at, batch_tokens, and running_total_tokens, ordered by recorded_at ASC.",
        "ground_truth_sql": """WITH time_batches AS (
    SELECT recorded_at, SUM(tokens_generated_delta) AS batch_tokens
    FROM inference_telemetry
    GROUP BY recorded_at
)
SELECT recorded_at, batch_tokens,
       SUM(batch_tokens) OVER (ORDER BY recorded_at ASC) AS running_total_tokens
FROM time_batches
ORDER BY recorded_at ASC;""",
    },
    {
        "id": "SQL46",
        "category": "window_cte",
        "question": "Find the top 2 pods with the highest average prefix_cache_hit_rate_pct within each deployment_type (using ROW_NUMBER() with tie-break pod_id ASC). Return deployment_type, pod_id, and avg_hit_rate, ordered by deployment_type ASC, avg_hit_rate DESC, pod_id ASC.",
        "ground_truth_sql": """WITH pod_hits AS (
    SELECT p.deployment_type, p.pod_id, AVG(t.prefix_cache_hit_rate_pct) AS avg_hit_rate
    FROM pod_deployments p
    JOIN inference_telemetry t ON p.pod_id = t.pod_id
    GROUP BY p.deployment_type, p.pod_id
), ranked AS (
    SELECT deployment_type, pod_id, avg_hit_rate,
           ROW_NUMBER() OVER (PARTITION BY deployment_type ORDER BY avg_hit_rate DESC, pod_id ASC) AS rn
    FROM pod_hits
)
SELECT deployment_type, pod_id, avg_hit_rate
FROM ranked
WHERE rn <= 2
ORDER BY deployment_type ASC, avg_hit_rate DESC, pod_id ASC;""",
    },
    {
        "id": "SQL47",
        "category": "window_cte",
        "question": "Compute for each organization the difference between its max_concurrent_pods quota and its number of currently running pods (terminated_at IS NULL). Return org_name, max_concurrent_pods, running_pods, and remaining_quota for organizations that currently have at least 2 running pods, ordered by remaining_quota ASC, org_name ASC.",
        "ground_truth_sql": """WITH running AS (
    SELECT org_id, COUNT(*) AS running_pods
    FROM pod_deployments
    WHERE terminated_at IS NULL
    GROUP BY org_id
)
SELECT o.org_name, o.max_concurrent_pods, r.running_pods,
       (o.max_concurrent_pods - r.running_pods) AS remaining_quota
FROM organizations o
JOIN running r ON o.org_id = r.org_id
WHERE r.running_pods >= 2
ORDER BY remaining_quota ASC, o.org_name ASC;""",
    },
    {
        "id": "SQL48",
        "category": "window_cte",
        "question": "For pods that have BOTH at least one snapshot at active_concurrency = 1 AND at least one snapshot at active_concurrency >= 32, compare their average ttft_p50_ms at low concurrency (= 1) vs high concurrency (>= 32). Return pod_id, avg_ttft_c1, and avg_ttft_high, ordered by (avg_ttft_high - avg_ttft_c1) DESC, pod_id ASC.",
        "ground_truth_sql": """WITH pod_conc AS (
    SELECT pod_id,
           AVG(CASE WHEN active_concurrency = 1 THEN ttft_p50_ms END) AS avg_ttft_c1,
           AVG(CASE WHEN active_concurrency >= 32 THEN ttft_p50_ms END) AS avg_ttft_high
    FROM inference_telemetry
    GROUP BY pod_id
    HAVING COUNT(CASE WHEN active_concurrency = 1 THEN 1 END) > 0
       AND COUNT(CASE WHEN active_concurrency >= 32 THEN 1 END) > 0
)
SELECT pod_id, avg_ttft_c1, avg_ttft_high
FROM pod_conc
ORDER BY (avg_ttft_high - avg_ttft_c1) DESC, pod_id ASC;""",
    },
    {
        "id": "SQL49",
        "category": "window_cte",
        "question": "Find all pods whose total num_preemptions_delta in the last 24 hours (recorded_at >= '2026-11-17T12:00:00Z') is greater than 0, along with their organization tier and node gpu_model. Return pod_id, tier, gpu_model, and preemptions_24h, ordered by preemptions_24h DESC, pod_id ASC.",
        "ground_truth_sql": """SELECT p.pod_id, o.tier, g.gpu_model, SUM(t.num_preemptions_delta) AS preemptions_24h
FROM pod_deployments p
JOIN organizations o ON p.org_id = o.org_id
JOIN gpu_nodes g ON p.node_id = g.node_id
JOIN inference_telemetry t ON p.pod_id = t.pod_id
WHERE t.recorded_at >= '2026-11-17T12:00:00Z'
GROUP BY p.pod_id, o.tier, g.gpu_model
HAVING SUM(t.num_preemptions_delta) > 0
ORDER BY preemptions_24h DESC, p.pod_id ASC;""",
    },
    {
        "id": "SQL50",
        "category": "window_cte",
        "question": "For each node_id, calculate the total hourly revenue generated by its pods (SUM(hourly_rate_usd)) and its rank within its datacenter_region using RANK(). Return datacenter_region, node_id, node_hourly_revenue, and region_rank, ordered by datacenter_region ASC, region_rank ASC, node_id ASC.",
        "ground_truth_sql": """WITH node_rev AS (
    SELECT g.datacenter_region, g.node_id, COALESCE(SUM(p.hourly_rate_usd), 0.0) AS node_hourly_revenue
    FROM gpu_nodes g
    LEFT JOIN pod_deployments p ON g.node_id = p.node_id
    GROUP BY g.datacenter_region, g.node_id
)
SELECT datacenter_region, node_id, node_hourly_revenue,
       RANK() OVER (PARTITION BY datacenter_region ORDER BY node_hourly_revenue DESC) AS region_rank
FROM node_rev
ORDER BY datacenter_region ASC, region_rank ASC, node_id ASC;""",
    },
]


def extract_sql(raw_text: str) -> str:
    """Extract SQL query from `<sql>...</sql>`, markdown fenced blocks, or raw SQL."""
    if not raw_text:
        return ""
    m = re.search(r"<sql>\s*(.*?)\s*</sql>", raw_text, re.DOTALL | re.IGNORECASE)
    if m:
        sql = m.group(1).strip()
    else:
        m2 = re.search(r"```(?:sql|sqlite)?\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
        if m2:
            sql = m2.group(1).strip()
        else:
            m3 = re.search(r"((?:WITH|SELECT)\b.*)", raw_text, re.DOTALL | re.IGNORECASE)
            sql = m3.group(1).strip() if m3 else raw_text.strip()

    # Keep only the first SQL statement if multiple are separated by semicolon
    if ";" in sql:
        first_stmt = sql.split(";", 1)[0].strip()
        if first_stmt:
            sql = first_stmt + ";"
    return sql


def normalize_cell(val: Any) -> Any:
    """Normalize SQLite values so minor float/int precision differences match cleanly."""
    if val is None:
        return None
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        if math.isnan(val) or math.isinf(val):
            return str(val)
        # Round floats to 2 decimal places; normalize integral floats like 10.0 -> 10.0
        return round(float(val), 2)
    return str(val).strip()


def normalize_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return [tuple(normalize_cell(c) for c in row) for row in rows]


def execute_with_timeout(conn: sqlite3.Connection, sql: str, timeout_s: float = 2.0) -> list[tuple[Any, ...]]:
    start = time.monotonic()

    def progress_handler() -> int:
        if time.monotonic() - start > timeout_s:
            return 1  # abort query
        return 0

    conn.set_progress_handler(progress_handler, 5000)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(5000)
        return rows
    finally:
        conn.set_progress_handler(None, 0)


def evaluate_sql_prediction(
    conn: sqlite3.Connection,
    pred_text: str,
    gt_sql: str,
) -> dict[str, Any]:
    """Verify syntax validity (`EXPLAIN QUERY PLAN`) and exact execution equivalence."""
    pred_sql = extract_sql(pred_text)
    if not pred_sql:
        return {"syntax_valid": False, "exec_match": False, "error": "empty_sql", "pred_sql": ""}

    # Prevent destructive SQL even on in-memory DB
    upper = pred_sql.lstrip().upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return {"syntax_valid": False, "exec_match": False, "error": "non_select_query", "pred_sql": pred_sql}

    try:
        conn.execute(f"EXPLAIN QUERY PLAN {pred_sql}")
    except Exception as exc:
        return {"syntax_valid": False, "exec_match": False, "error": f"syntax: {exc}", "pred_sql": pred_sql}

    try:
        pred_rows = normalize_rows(execute_with_timeout(conn, pred_sql))
        gt_rows = normalize_rows(execute_with_timeout(conn, gt_sql))
    except Exception as exc:
        return {"syntax_valid": True, "exec_match": False, "error": f"exec: {exc}", "pred_sql": pred_sql}

    # Exact ordered match OR exact multiset match if row order differs only on ties
    ordered_match = pred_rows == gt_rows
    multiset_match = sorted(pred_rows, key=lambda r: tuple("" if x is None else str(x) for x in r)) == sorted(
        gt_rows, key=lambda r: tuple("" if x is None else str(x) for x in r)
    )
    return {
        "syntax_valid": True,
        "exec_match": bool(ordered_match or multiset_match),
        "ordered_match": bool(ordered_match),
        "error": None if (ordered_match or multiset_match) else f"mismatch (got {len(pred_rows)} rows vs gt {len(gt_rows)} rows)",
        "pred_sql": pred_sql,
    }


def validate_local_benchmark() -> None:
    """Verify locally that the SQLite sandbox and all 50 ground-truth SQL queries run cleanly."""
    conn = create_sandbox_db(seed=42)
    assert len(SQL_BENCHMARK_TASKS) == 50, f"Expected 50 tasks, got {len(SQL_BENCHMARK_TASKS)}"

    print(f"Validating {len(SQL_BENCHMARK_TASKS)} ground-truth SQL queries against seeded SQLite DB...")
    for task in SQL_BENCHMARK_TASKS:
        rows = execute_with_timeout(conn, task["ground_truth_sql"])
        if not rows:
            raise AssertionError(f"Task {task['id']} returned 0 rows! Query:\n{task['ground_truth_sql']}")
        res = evaluate_sql_prediction(conn, f"<sql>{task['ground_truth_sql']}</sql>", task["ground_truth_sql"])
        if not (res["syntax_valid"] and res["exec_match"]):
            raise AssertionError(f"Self-check failed on {task['id']}: {res}")
        print(f"  [{task['id']}] ({task['category']}) -> {len(rows)} rows | sample: {rows[0]}")
    print("SUCCESS: All 50 ground-truth SQL queries returned non-empty deterministic results!")


# ---------------------------------------------------------------------------
# Live vLLM Benchmark Runner & 4-Way Quantization Ablation
# ---------------------------------------------------------------------------

def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def choose_safe_gpu_util(target_gb: float = 28.0) -> float:
    """Compute a safe --gpu-memory-utilization respecting already-running VRAM users (e.g. ComfyUI)."""
    import torch
    free_b, total_b = torch.cuda.mem_get_info(0)
    free_gb = free_b / (1024 ** 3)
    total_gb = total_b / (1024 ** 3)
    # Leave at least 12 GiB headroom if total VRAM > 40 GiB, else use 0.85
    if total_gb <= 32.0:
        return min(0.85, round((free_gb * 0.90) / total_gb, 2))
    safe_cap_gb = min(target_gb, free_gb - 10.0)
    util = max(0.20, min(0.80, round(safe_cap_gb / total_gb, 2)))
    print(f"Detected shared GPU ({free_gb:.1f}/{total_gb:.1f} GiB free) -> using --gpu-memory-utilization {util}")
    return util


async def run_sql_wave(
    endpoint_url: str,
    model_name: str,
    concurrency: int = 16,
    max_tokens: int = 256,
) -> dict[str, Any]:
    import aiohttp

    conn = create_sandbox_db(seed=42)
    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=180)

    async def eval_one(session: aiohttp.ClientSession, task: dict[str, str]) -> dict[str, Any]:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task["question"]},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
        async with semaphore:
            t0 = time.perf_counter()
            first_token_t: Optional[float] = None
            ttft_ms = 0.0
            out_tokens = 0
            pieces: list[str] = []
            try:
                async with session.post(endpoint_url, json=payload) as resp:
                    if resp.status != 200:
                        txt = await resp.text()
                        return {"id": task["id"], "category": task["category"], "ok": False, "error": f"HTTP {resp.status}: {txt[:120]}"}
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
                full_text = "".join(pieces)
                verdict = evaluate_sql_prediction(conn, full_text, task["ground_truth_sql"])
                return {
                    "id": task["id"],
                    "category": task["category"],
                    "ok": out_tokens > 0,
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms,
                    "e2e_ms": e2e_ms,
                    "out_tokens": out_tokens,
                    **verdict,
                }
            except Exception as exc:
                return {"id": task["id"], "category": task["category"], "ok": False, "error": str(exc)}

    wall_t0 = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Send 1 warmup request so prefix cache + CUDA graphs/triton JIT stabilize before timing
        await eval_one(session, SQL_BENCHMARK_TASKS[0])
        wave_t0 = time.perf_counter()
        results = await asyncio.gather(*(eval_one(session, t) for t in SQL_BENCHMARK_TASKS))
    wall_s = time.perf_counter() - wave_t0

    ok_res = [r for r in results if r.get("ok")]
    n_tasks = len(SQL_BENCHMARK_TASKS)
    syntax_cnt = sum(1 for r in results if r.get("syntax_valid"))
    exec_cnt = sum(1 for r in results if r.get("exec_match"))
    total_toks = sum(r.get("out_tokens", 0) for r in ok_res)

    ttfts = sorted(r["ttft_ms"] for r in ok_res)
    tpots = sorted(r["tpot_ms"] for r in ok_res if r.get("out_tokens", 0) > 1)

    cat_breakdown = {}
    for cat in ("filter_null", "join_having", "case_ratio", "window_cte"):
        sub = [r for r in results if r["category"] == cat]
        cat_breakdown[cat] = round(100.0 * sum(1 for r in sub if r.get("exec_match")) / max(1, len(sub)), 1)

    return {
        "total_tasks": n_tasks,
        "succeeded_requests": len(ok_res),
        "syntax_valid_count": syntax_cnt,
        "syntax_valid_pct": round(100.0 * syntax_cnt / n_tasks, 1),
        "exec_match_count": exec_cnt,
        "exec_accuracy_pct": round(100.0 * exec_cnt / n_tasks, 1),
        "category_accuracy_pct": cat_breakdown,
        "throughput_tok_per_s": round(total_toks / wall_s, 2) if wall_s > 0 else 0.0,
        "ttft_p50_ms": round(percentile(ttfts, 50), 1),
        "ttft_p99_ms": round(percentile(ttfts, 99), 1),
        "tpot_p50_ms": round(percentile(tpots, 50), 1),
        "tpot_p99_ms": round(percentile(tpots, 99), 1),
        "wall_time_s": round(wall_s, 2),
        "failures_sample": [
            {"id": r["id"], "error": r.get("error"), "pred_sql": r.get("pred_sql", "")[:240]}
            for r in results if not r.get("exec_match")
        ][:10],
    }


def parse_kv_telemetry(log_text: str) -> dict[str, Any]:
    kv_mem_gb = None
    kv_tokens = None
    max_conc_4k = None
    m1 = re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", log_text)
    if m1:
        kv_mem_gb = float(m1.group(1))
    m2 = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens,\s*Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x", log_text)
    if m2:
        kv_tokens = int(m2.group(1).replace(",", ""))
        max_conc_4k = float(m2.group(2))
    return {"kv_cache_gib": kv_mem_gb, "kv_cache_tokens": kv_tokens, "max_conc_4k": max_conc_4k}


def _build_vllm_env() -> dict[str, str]:
    env = os.environ.copy()
    venv_bin = str(Path(sys.executable).parent)
    cuda_bin = "/usr/local/cuda/bin"
    extra_paths = [p for p in (venv_bin, cuda_bin) if Path(p).exists()]
    env["PATH"] = os.pathsep.join(extra_paths + [env.get("PATH", "")])
    if Path("/usr/local/cuda").exists():
        env.setdefault("CUDA_HOME", "/usr/local/cuda")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    return env


def _run_single_setup(
    vllm_bin: str,
    gpu_name: str,
    util: float,
    concurrency: int,
    cfg: dict[str, Any],
    log_tag: str,
    env: dict[str, str],
) -> dict[str, Any]:
    import urllib.request

    print(
        f"\n=== Running setup [{log_tag}]: {cfg['label']} "
        f"({cfg['model']}, kv={cfg['kv_cache_dtype']}, extra={cfg.get('extra_args', [])}) ===",
        flush=True,
    )
    subprocess.run(["pkill", "-f", "vllm serve"], check=False)
    time.sleep(2)

    log_file = Path(f"/tmp/vllm_ablation_{log_tag}.log")
    cmd = [
        vllm_bin, "serve", cfg["model"],
        "--host", "127.0.0.1",
        "--port", "8000",
        "--gpu-memory-utilization", str(util),
        "--max-model-len", "4096",
        "--enforce-eager",
        "--kv-cache-dtype", cfg["kv_cache_dtype"],
    ]
    if cfg.get("enable_prefix_caching", True):
        cmd.append("--enable-prefix-caching")
    else:
        cmd.append("--no-enable-prefix-caching")
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
            print("\n".join(log_text.splitlines()[-40:]), flush=True)
            return {**cfg, "status": "failed_to_start", "log_tail": log_text.splitlines()[-25:]}

        kv_info = parse_kv_telemetry(log_text)
        print(f"vLLM ready! KV telemetry: {kv_info}", flush=True)
        metrics = asyncio.run(
            run_sql_wave(
                endpoint_url="http://127.0.0.1:8000/v1/chat/completions",
                model_name=cfg["model"],
                concurrency=concurrency,
                max_tokens=512,
            )
        )
        entry = {
            "label": cfg["label"],
            "model": cfg["model"],
            "kv_cache_dtype": cfg["kv_cache_dtype"],
            "extra_args": cfg.get("extra_args", []),
            "enable_prefix_caching": cfg.get("enable_prefix_caching", True),
            "gpu": gpu_name,
            "gpu_mem_util": util,
            "concurrency": concurrency,
            **kv_info,
            **metrics,
        }
        print(
            f"RESULT [{cfg['label']}]: Syntax={entry['syntax_valid_pct']}% | "
            f"ExecAcc={entry['exec_accuracy_pct']}% | "
            f"Tput={entry['throughput_tok_per_s']} tok/s | "
            f"TTFT_P99={entry['ttft_p99_ms']}ms | TPOT_P99={entry['tpot_p99_ms']}ms",
            flush=True,
        )
        return entry
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


def run_ablation(out_path: str, concurrency: int = 16) -> None:
    import torch

    vllm_bin = str(Path(sys.executable).parent / "vllm")
    if not Path(vllm_bin).exists():
        vllm_bin = "vllm"

    gpu_name = torch.cuda.get_device_name(0)
    util = choose_safe_gpu_util(target_gb=28.0)
    env = _build_vllm_env()

    configs = [
        {"label": "BF16 + FP16 KV", "model": "Qwen/Qwen2.5-7B-Instruct", "kv_cache_dtype": "auto", "extra_args": []},
        {"label": "BF16 + FP8 KV", "model": "Qwen/Qwen2.5-7B-Instruct", "kv_cache_dtype": "fp8", "extra_args": []},
        {"label": "AWQ 4-bit + FP16 KV", "model": "Qwen/Qwen2.5-7B-Instruct-AWQ", "kv_cache_dtype": "auto", "extra_args": []},
        {"label": "AWQ 4-bit + FP8 KV", "model": "Qwen/Qwen2.5-7B-Instruct-AWQ", "kv_cache_dtype": "fp8", "extra_args": []},
    ]

    all_results = [
        _run_single_setup(vllm_bin, gpu_name, util, concurrency, cfg, str(idx), env)
        for idx, cfg in enumerate(configs, 1)
    ]

    payload = {
        "gpu": gpu_name,
        "concurrency": concurrency,
        "num_tasks": len(SQL_BENCHMARK_TASKS),
        "results": all_results,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved full ablation results to {out_path}")


def test_fp8_backends(out_path: str, concurrency: int = 16) -> None:
    """Test alternative attention backends / prefix caching for FP8 KV on Blackwell and merge into out_path."""
    import torch

    vllm_bin = str(Path(sys.executable).parent / "vllm")
    if not Path(vllm_bin).exists():
        vllm_bin = "vllm"

    gpu_name = torch.cuda.get_device_name(0)
    util = choose_safe_gpu_util(target_gb=28.0)
    env = _build_vllm_env()

    existing_path = Path(out_path)
    if existing_path.exists():
        payload = json.loads(existing_path.read_text(encoding="utf-8"))
        base_results = []
        for r in payload.get("results", []):
            # Rename original default FlashInfer FP8 runs so labels stay unambiguous
            if r.get("label") == "BF16 + FP8 KV":
                r["label"] = "BF16 + FP8 KV (FlashInfer Default)"
            elif r.get("label") == "AWQ 4-bit + FP8 KV":
                r["label"] = "AWQ 4-bit + FP8 KV (FlashInfer Default)"
            if not str(r.get("label", "")).endswith("(TRITON_ATTN)") and "No PrefixCache" not in str(r.get("label", "")) and "(FLASH_ATTN)" not in str(r.get("label", "")):
                base_results.append(r)
    else:
        base_results = []

    probe_configs = [
        {
            "label": "BF16 + FP8 KV (TRITON_ATTN)",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "kv_cache_dtype": "fp8",
            "extra_args": ["--attention-backend", "TRITON_ATTN"],
        },
        {
            "label": "AWQ 4-bit + FP8 KV (TRITON_ATTN)",
            "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "kv_cache_dtype": "fp8",
            "extra_args": ["--attention-backend", "TRITON_ATTN"],
        },
        {
            "label": "BF16 + FP8 KV (FlashInfer, No PrefixCache)",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "kv_cache_dtype": "fp8",
            "enable_prefix_caching": False,
            "extra_args": ["--attention-backend", "FLASHINFER"],
        },
    ]

    new_results: list[dict[str, Any]] = []
    triton_worked = False
    for idx, cfg in enumerate(probe_configs, 10):
        res = _run_single_setup(vllm_bin, gpu_name, util, concurrency, cfg, f"fp8_{idx}", env)
        new_results.append(res)
        if "TRITON_ATTN" in cfg["label"] and res.get("syntax_valid_pct", 0.0) > 10.0:
            triton_worked = True

    if not triton_worked:
        # Fallback check with FLASH_ATTN if TRITON_ATTN didn't succeed cleanly
        for idx, cfg in enumerate([
            {
                "label": "BF16 + FP8 KV (FLASH_ATTN)",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "kv_cache_dtype": "fp8",
                "extra_args": ["--attention-backend", "FLASH_ATTN"],
            },
            {
                "label": "AWQ 4-bit + FP8 KV (FLASH_ATTN)",
                "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                "kv_cache_dtype": "fp8",
                "extra_args": ["--attention-backend", "FLASH_ATTN"],
            },
        ], 20):
            new_results.append(_run_single_setup(vllm_bin, gpu_name, util, concurrency, cfg, f"fp8_{idx}", env))

    payload = {
        "gpu": gpu_name,
        "concurrency": concurrency,
        "num_tasks": len(SQL_BENCHMARK_TASKS),
        "results": base_results + new_results,
    }
    existing_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nMerged FP8 backend diagnostics into {out_path}")


def test_fp8_e5m2(out_path: str, concurrency: int = 16) -> None:
    """Test fp8_e5m2 (and fallback TRITON_ATTN / fp8_per_token_head) and merge into out_path."""
    import torch

    vllm_bin = str(Path(sys.executable).parent / "vllm")
    if not Path(vllm_bin).exists():
        vllm_bin = "vllm"

    gpu_name = torch.cuda.get_device_name(0)
    util = choose_safe_gpu_util(target_gb=28.0)
    env = _build_vllm_env()

    existing_path = Path(out_path)
    payload = json.loads(existing_path.read_text(encoding="utf-8")) if existing_path.exists() else {"results": []}
    # Deduplicate prior e5m2/per_token runs if re-running
    base_results = [
        r for r in payload.get("results", [])
        if "FP8_E5M2" not in str(r.get("label", "")) and "FP8_PER_TOKEN_HEAD" not in str(r.get("label", ""))
    ]

    new_results: list[dict[str, Any]] = []
    for idx, (base_label, model_name) in enumerate([
        ("BF16 + FP8_E5M2 KV", "Qwen/Qwen2.5-7B-Instruct"),
        ("AWQ 4-bit + FP8_E5M2 KV", "Qwen/Qwen2.5-7B-Instruct-AWQ"),
    ], 30):
        cfg = {"label": base_label, "model": model_name, "kv_cache_dtype": "fp8_e5m2", "extra_args": []}
        res = _run_single_setup(vllm_bin, gpu_name, util, concurrency, cfg, f"e5m2_{idx}", env)
        if res.get("status") == "failed_to_start":
            cfg_triton = {
                "label": f"{base_label} (TRITON_ATTN)",
                "model": model_name,
                "kv_cache_dtype": "fp8_e5m2",
                "extra_args": ["--attention-backend", "TRITON_ATTN"],
            }
            res = _run_single_setup(vllm_bin, gpu_name, util, concurrency, cfg_triton, f"e5m2_tr_{idx}", env)
        new_results.append(res)

    # If e5m2 still failed or gave 0% syntax, test fp8_per_token_head on BF16
    if not any(r.get("syntax_valid_pct", 0.0) > 10.0 for r in new_results):
        cfg_pth = {
            "label": "BF16 + FP8_PER_TOKEN_HEAD KV",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "kv_cache_dtype": "fp8_per_token_head",
            "extra_args": [],
        }
        res_pth = _run_single_setup(vllm_bin, gpu_name, util, concurrency, cfg_pth, "pth_40", env)
        if res_pth.get("status") == "failed_to_start":
            cfg_pth_tr = {
                "label": "BF16 + FP8_PER_TOKEN_HEAD KV (TRITON_ATTN)",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "kv_cache_dtype": "fp8_per_token_head",
                "extra_args": ["--attention-backend", "TRITON_ATTN"],
            }
            res_pth = _run_single_setup(vllm_bin, gpu_name, util, concurrency, cfg_pth_tr, "pth_tr_40", env)
        new_results.append(res_pth)

    merged = {
        "gpu": gpu_name,
        "concurrency": concurrency,
        "num_tasks": len(SQL_BENCHMARK_TASKS),
        "results": base_results + new_results,
    }
    existing_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"\nMerged fp8_e5m2 results into {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-local", action="store_true")
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--test-fp8-backends", action="store_true")
    parser.add_argument("--test-fp8-e5m2", action="store_true")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--out", default="/content/results_sql_accuracy_ablation.json")
    args = parser.parse_args()

    if args.validate_local:
        validate_local_benchmark()
    elif args.run_ablation:
        run_ablation(out_path=args.out, concurrency=args.concurrency)
    elif args.test_fp8_backends:
        test_fp8_backends(out_path=args.out, concurrency=args.concurrency)
    elif args.test_fp8_e5m2:
        test_fp8_e5m2(out_path=args.out, concurrency=args.concurrency)
    else:
        parser.print_help()
