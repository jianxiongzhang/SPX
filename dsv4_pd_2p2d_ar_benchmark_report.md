# DeepSeek-V4 PD 2P2D Benchmark — AR OFF vs AR ON DDP ON

**Generated:** 2026-06-28 03:33 · **Results:** `/root/jianxiong/logs/bench_pd_2p2d_ar_20260627_234625` · **Completion:** AR OFF 17/17 · AR ON 17/17 · 32 GPU · NCCLEP · NIXL RDMA

## Test Topology (2P2D Layer-1)

```
                         ┌─────────────────────────────────────┐
                         │  Router  10.6.142.10:32050          │
                         │  (sglang_router, PD disaggregation) │
                         └──────────────┬──────────────────────┘
                    prefill            │            decode
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│ Prefill-0           │      │ Prefill-1           │      │ Decode-0            │
│ 10.6.142.10:30000   │      │ 10.6.142.14:30001   │      │ 10.6.142.11:30000   │
│ tp=8  ep=8  (8 GPU) │      │ tp=8  ep=8  (8 GPU) │      │ tp=8  ep=8  (8 GPU) │
│ bootstrap :19000    │      │ bootstrap :19001    │      │                     │
└──────────┬──────────┘      └──────────┬──────────┘      └──────────┬──────────┘
           │                            │                            │
           │         KV transfer (NIXL RDMA, mlx5 IB)                 │
           └────────────────────────────┴────────────────────────────┘
                                        │
                              ┌─────────────────────┐
                              │ Decode-1            │
                              │ 10.6.142.15:30001   │
                              │ tp=8  ep=8  (8 GPU) │
                              └─────────────────────┘

Control / Bench host: 10.6.142.11 (/root/jianxiong)
Model: DeepSeek-V4-Flash-FP8
MoE: NCCLEP low_latency (fp8 dispatch)
KV:  NIXL RDMA (disaggregation-transfer-backend=nixl)
Total: 4 independent jobs × 8 GPU = 32 GPU
```

- **2P2D:** 2 Prefill nodes + 2 Decode nodes; router handles PD routing
- **tp=8 ep=8 per node:** MoE expert parallelism within node; cross-node traffic is KV only (no cross-IB EP)
- **Comparison dimension:** NCCL Adaptive Routing OFF vs ON (DDP ON); topology and model config otherwise identical

## AR OFF vs AR ON DDP ON — How Parameters Are Passed

**Key finding:** The entire benchmark switches between modes via a single shell variable `NCCL_AR_MODE` with values `ar_off` or `ar_on_ddp_on`. The script translates this into 4 NCCL IB environment variables injected into each prefill/decode process's `sglang.launch_server` environment inside Docker **before process start**. Between modes, the cluster must be **fully restarted** (stop → start); otherwise old processes retain prior NCCL settings.

### 1. Entry Point: Who Sets NCCL_AR_MODE?

| Scenario | Command / Code Location | Input |
|----------|---------------------------|-------|
| Manual cluster start | `export NCCL_AR_MODE=ar_off` or `ar_on_ddp_on`, then `run_dsv4_ncclep_pd_ep16_2p2d.sh start` | Environment variable |
| Full benchmark | `bench_dsv4_ncclep_pd_2p2d_ar.sh all` → internally calls `run_all_cases ar_off` and `run_all_cases ar_on_ddp_on` | Function arg → export |
| Long-context retest | `bench_dsv4_ncclep_pd_2p2d_smoke6.sh <dir> [ar_on_ddp_on]` → `start_cluster` | Script 2nd argument |
| AR showcase retest | `bench_dsv4_ncclep_pd_2p2d_ar.sh ar_showcase <dir>` → `run_ar_showcase_cases` | 1st arg + BASE_OUT 2nd arg |

```
bench_dsv4_ncclep_pd_2p2d_ar.sh
  run_all_cases(ar_mode)
    export NCCL_AR_MODE="$ar_mode"          # ① Set mode
    NCCL_AR_MODE="$ar_mode" bash run_dsv4_ncclep_pd_ep16_2p2d.sh start   # ② Explicitly pass to start

run_dsv4_ncclep_pd_ep16_2p2d.sh
  NCCL_AR_MODE="${NCCL_AR_MODE:-ar_off}"    # ③ Default ar_off
  export NCCL_AR_MODE
  launch_prefill / launch_decode
    docker exec ... bash -lc '
      $(sglang_ncclep_block "${NCCL_AR_MODE}")   # ④ Expand NCCL env block
      nohup python3 -m sglang.launch_server ...
    '
```

### 2. Translation: NCCL_AR_MODE → 4 NCCL Variables

Logic in `pd/env_pd_ep16_2p2d.sh` function `nccl_ar_overlay()`. `sglang_ncclep_block(mode)` embeds the function's `export` statements into each job's startup shell.

| NCCL_AR_MODE | NCCL_IB_ADAPTIVE_ROUTING | NCCL_IB_OOO_RQ | NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME | NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS | Report Label |
|--------------|--------------------------|----------------|---------------------------------------|---------------------------------------|--------------|
| `ar_off` (default) | 0 | 0 | 0 | 0 | AR_OFF |
| `ar_on_ddp_on` | 1 | 1 | 1 | 1 | AR_ON_DDP_ON |

**Naming:** **AR ON** = InfiniBand Adaptive Routing enabled; **DDP ON** = OOO RQ, Receiver-side Matching, and Prepost Receive WR also enabled (consistent with `ar_on_ddp_on` in ar_bench 3-case config). `ar_label()` is only for log/result directory naming (`AR_OFF` / `AR_ON_DDP_ON`), not NCCL parameter passing.

### 3. Injection Point: Which Processes Receive Parameters?

**Affected:** 4 `sglang.launch_server` processes (2× prefill + 2× decode), each tp=8 ep=8 **NCCLEP MoE intra-node NCCL communication**

**Not affected:**
- **Cross-node KV transfer:** Uses NIXL/UCX (`nixl_rdma_block()`); independent of the 4 NCCL AR variables above
- **Router** (`sglang_router.launch_router`): No NCCL AR overlay; HTTP PD routing only
- **bench_serving client:** Connects to router URL only; no NCCL parameters

```
# Actual injection order in launch_prefill / launch_decode (sglang_ncclep_block):
source ncclep_env.sh                    # NCCL lib path, LD_PRELOAD
export PYTHONPATH=...
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# --- nccl_ar_overlay(NCCL_AR_MODE) expanded ---
export NCCL_IB_ADAPTIVE_ROUTING=0|1
export NCCL_IB_OOO_RQ=0|1
export NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0|1
export NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0|1
# --- nccl_single_node_block (identical in both modes) ---
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_P2P_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_4,mlx5_6
...
# --- nixl_rdma_block (identical in both modes, for KV) ---
export PD_TRANSFER_BACKEND=nixl
export UCX_NET_DEVICES=mlx5_0:1,...
```

### 4. How Are the Two Modes Isolated?

1. After AR OFF completes all cases, `bench_dsv4_ncclep_pd_2p2d_ar.sh` calls `run_dsv4_ncclep_pd_ep16_2p2d.sh stop` (kills all sglang processes)
2. Then `export NCCL_AR_MODE=ar_on_ddp_on` and `start`, relaunching 4 jobs + router
3. Results written separately to `.../AR_OFF/` and `.../AR_ON_DDP_ON/`; log filenames include `AR_OFF` / `AR_ON_DDP_ON` suffix

### 5. Quick Reproduction for Mode Switching

```bash
# AR OFF
export NCCL_AR_MODE=ar_off
bash /root/jianxiong/run_dsv4_ncclep_pd_ep16_2p2d.sh start

# AR ON DDP ON (must stop first, or start will auto-kill)
export NCCL_AR_MODE=ar_on_ddp_on
bash /root/jianxiong/run_dsv4_ncclep_pd_ep16_2p2d.sh start

# Verify current mode (check prefill log or process env)
# Logs: logs/pd_ep16_2p2d/prefill_*_AR_OFF.log  vs  *_AR_ON_DDP_ON.log
```

Δ% comparisons use AR OFF as baseline. Negative TTFT/TPOT means AR ON is lower (better); positive Out tok/s means AR ON is higher (better).

## Case Design (17 cases: 13 baseline + 4 AR showcase)

First 13 cases align with `dsv4_pd_1p1d_benchmark_report.html`; Group F is 2P2D AR-specific supplement to amplify NCCLEP MoE intra-node NCCL communication differences (see "AR OFF vs AR ON DDP ON — How Parameters Are Passed").

### Group A: Decode-focused — short sequence, high concurrency; primarily tests decode-side MoE/NCCLEP and PD decode throughput

- `pd_decode_lat_c1` — ISL=128, OSL=128, conc=1: Short prompt, single concurrency; isolates decode single-request TTFT/TPOT baseline
- `pd_decode_thr_c8` — ISL=128, OSL=128, conc=8: Fixed short sequence, ramp concurrency; tests decode pool saturation throughput
- `pd_decode_thr_c16` — ISL=128, OSL=128, conc=16: Further concurrency scaling; observe decode queuing and TPOT degradation
- `pd_decode_thr_c32` — ISL=128, OSL=128, conc=32: Near decode-side max_running_requests; tests peak PD throughput

### Group B: Mixed workload — typical chat ratio (ISL≈2K, OSL≈512); tests full PD path

- `pd_mixed_c1` — ISL=2048, OSL=512, conc=1: Medium ISL + medium OSL, single-concurrency end-to-end PD path
- `pd_mixed_c4` — ISL=2048, OSL=512, conc=4: Mixed workload concurrency; covers prefill+decode+KV coordination
- `pd_mixed_c8` — ISL=2048, OSL=512, conc=8: Higher-concurrency mixed stress; observe router scheduling and KV transfer

### Group C: Prefill-focused — long input; tests prefill node compute and TTFT

- `pd_prefill_c1` — ISL=4096, OSL=128, conc=1: Long input short output; prefill compute dominant
- `pd_prefill_c4` — ISL=4096, OSL=128, conc=4: Prefill concurrency; tests P-node batching and TTFT scaling
- `pd_long_prefill_c1` — ISL=8064, OSL=128, conc=1: Near context-limit prefill; tests long-prefix KV generation

### Group D: Long context — ISL=8192 + OSL=1024; requires ctx≥10240, P mem=0.55 (retest-specific config)

- `pd_long_ctx_c1` — ISL=8192, OSL=1024, conc=1: ISL+OSL near limit; single-concurrency long-context stability
- `pd_long_ctx_c4` — ISL=8192, OSL=1024, conc=4: Long-context concurrency; critical case for prefill OOM/KV pressure

### Group E: KV transfer stress — sustained multi-concurrency KV RDMA; tests NIXL/IB and decode-side reception

- `pd_kv_stress_c8` — ISL=4096, OSL=256, conc=8: Sustained multi-request KV over IB (NIXL); tests RDMA link stability

### Group F: AR-specific supplement — high-concurrency decode/mixed + long-output decode; amplifies NCCLEP MoE intra-node NCCL differences (recommended for AR showcase)

- `pd_decode_thr_c48` — ISL=128, OSL=128, conc=48: Beyond c32; approaches per-decode-node max_running_requests=32
- `pd_mixed_c16` — ISL=2048, OSL=512, conc=16: High-concurrency mixed in 2P2D 4-node pipeline; P+D MoE NCCL both busy
- `pd_mixed_c32` — ISL=2048, OSL=512, conc=32: Highest mixed concurrency; most likely to differentiate AR OFF vs ON on Out/TPOT
- `pd_decode_long_c16` — ISL=128, OSL=512, conc=16: Short prompt long OSL; increases per-request decode MoE steps and NCCL traffic

**Long-context retest note:** Group D cases OOM under default config (ctx=8192, mem=0.70); retest uses ctx=10240, mem-fraction-static=0.55, P_max_running_requests=4.

**AR showcase retest:** Group F four cases can be retested separately and merged into existing result directory: `bash bench_dsv4_ncclep_pd_2p2d_ar.sh ar_showcase <BASE_OUT>`

## Retest Records

Initial `bench_dsv4_ncclep_pd_2p2d_ar.sh all` run failed Group D/E long-context cases due to ctx=8192 and prefill mem=0.70 OOM; Group F was retested afterward via `ar_showcase` mode with results overwritten. Table below summarizes final metric sources and per-case logs.

### Batch A — Long-Context Retest (smoke6)

**Script:** `/root/jianxiong/bench_dsv4_ncclep_pd_2p2d_smoke6.sh`

```bash
# Retest AR OFF + AR ON (3 cases × 2 = 6 runs)
bash /root/jianxiong/bench_dsv4_ncclep_pd_2p2d_smoke6.sh /root/jianxiong/logs/bench_pd_2p2d_ar_20260627_234625

# Retest AR ON only (actually executed this run)
bash /root/jianxiong/bench_dsv4_ncclep_pd_2p2d_smoke6.sh /root/jianxiong/logs/bench_pd_2p2d_ar_20260627_234625 ar_on_ddp_on
```

Hardcoded parameters in script: ctx=10240 · mem-fraction-static=0.55 · P_max_running_requests=4 · ready-check-timeout=600 · warmup=1

#### Batch Logs

| Log File | AR_FILTER | Start | End | Failures |
|----------|-----------|-------|-----|----------|
| `bench_smoke6_20260628_021250.log` | ar_on_ddp_on | 2026-06-28T02:12:50+08:00 | 2026-06-28T02:27:28+08:00 | 0 |

#### Case Data Sources (smoke6 related)

| Case | AR OFF Source | AR ON Source | OFF per-case log | ON per-case log |
|------|-----------------|--------------|------------------|-----------------|
| `pd_long_ctx_c1` | Retest smoke6/manual | Retest smoke6 | — | `pd_long_ctx_c1_smoke6.log` |
| `pd_long_ctx_c4` | Retest smoke6 | Retest smoke6 | `pd_long_ctx_c4_smoke6.log` | `pd_long_ctx_c4_smoke6.log` |
| `pd_kv_stress_c8` | Retest smoke6 | Retest smoke6 | `pd_kv_stress_c8_smoke6.log` | `pd_kv_stress_c8_smoke6.log` |

### Batch B — AR Showcase Retest (Group F)

**Script:** `/root/jianxiong/bench_dsv4_ncclep_pd_2p2d_ar.sh ar_showcase` (4 cases × 2 AR = 8 runs, merged into existing directory)

```bash
bash /root/jianxiong/bench_dsv4_ncclep_pd_2p2d_ar.sh ar_showcase /root/jianxiong/logs/bench_pd_2p2d_ar_20260627_234625
```

Produces `summary_ar_showcase.md` and `fail_count_ar_showcase`; main table metrics from merged `.jsonl`.

#### Batch Logs

| Log File | Path |
|----------|------|
| `bench_ar_showcase_20260628_023529.log` | `/root/jianxiong/logs/pd_ep16_2p2d/bench_ar_showcase_20260628_023529.log` |

#### Case Data Sources (showcase)

| Case | AR OFF Source | AR ON Source | OFF per-case log | ON per-case log |
|------|-----------------|--------------|------------------|-----------------|
| `pd_decode_thr_c48` | Retest ar_showcase | Retest ar_showcase | `pd_decode_thr_c48.log` | `pd_decode_thr_c48.log` |
| `pd_mixed_c16` | Retest ar_showcase | Retest ar_showcase | `pd_mixed_c16.log` | `pd_mixed_c16.log` |
| `pd_mixed_c32` | Retest ar_showcase | Retest ar_showcase | `pd_mixed_c32.log` | `pd_mixed_c32.log` |
| `pd_decode_long_c16` | Retest ar_showcase | Retest ar_showcase | `pd_decode_long_c16.log` | `pd_decode_long_c16.log` |

Alternative long-context retest script: `bench_dsv4_ncclep_pd_2p2d_retry3.sh` (same logic as smoke6, legacy naming).

## Full Case Comparison (17 cases · Groups D/E/F highlighted as retest cases)

| Case | Description | ISL | OSL | Conc | Data Source OFF/ON | TTFT OFF | TTFT ON | Δ TTFT | TPOT OFF | TPOT ON | Δ TPOT | Out OFF | Out ON | Δ Out | St OFF | St ON |
|------|-------------|-----|-----|------|-------------------|----------|---------|--------|----------|---------|--------|---------|--------|-------|--------|-------|
| pd_decode_lat_c1 | Decode latency baseline | 128 | 128 | 1 | Initial all / Initial all | 327.3 | 327.0 | +0.1% | 160.13 | 160.53 | -0.3% | 6.2 | 6.1 | -0.3% | ok | ok |
| pd_decode_thr_c8 | Decode throughput | 128 | 128 | 8 | Initial all / Initial all | 954.6 | 920.2 | +3.6% | 158.41 | 157.80 | +0.4% | 43.1 | 43.6 | +1.2% | ok | ok |
| pd_decode_thr_c16 | Decode scaling | 128 | 128 | 16 | Initial all / Initial all | 1030.9 | 1071.7 | -4.0% | 171.92 | 170.84 | +0.6% | 78.7 | 78.4 | -0.3% | ok | ok |
| pd_decode_thr_c32 | Decode peak | 128 | 128 | 32 | Initial all / Initial all | 1710.4 | 1630.1 | +4.7% | 198.10 | 193.26 | +2.4% | 124.6 | 122.4 | -1.7% | ok | ok |
| pd_mixed_c1 | Mixed E2E | 2048 | 512 | 1 | Initial all / Initial all | 3338.8 | 3316.3 | +0.7% | 160.21 | 160.39 | -0.1% | 5.8 | 5.8 | -0.1% | ok | ok |
| pd_mixed_c4 | Mixed concurrency | 2048 | 512 | 4 | Initial all / Initial all | 5372.9 | 5367.3 | +0.1% | 153.39 | 153.43 | -0.0% | 21.6 | 21.6 | -0.0% | ok | ok |
| pd_mixed_c8 | Mixed stress | 2048 | 512 | 8 | Initial all / Initial all | 8825.5 | 8611.1 | +2.4% | 146.75 | 149.86 | -2.1% | 40.6 | 41.6 | +2.3% | ok | ok |
| pd_prefill_c1 | Prefill intensive | 4096 | 128 | 1 | Initial all / Initial all | 7214.7 | 7199.5 | +0.2% | 160.27 | 160.55 | -0.2% | 4.2 | 4.2 | -0.1% | ok | ok |
| pd_prefill_c4 | Prefill concurrency | 4096 | 128 | 4 | Initial all / Initial all | 21091.6 | 21068.1 | +0.1% | 75.59 | 75.99 | -0.5% | 7.0 | 7.0 | +0.1% | ok | ok |
| pd_long_prefill_c1 | Longest valid prefill | 8064 | 128 | 1 | Initial all / Initial all | 2844.2 | 2835.7 | +0.3% | 161.85 | 160.97 | +0.5% | 0.8 | 1.3 | +75.2% | ok | ok |
| pd_long_ctx_c1 | Long context | 8192 | 1024 | 1 | Retest smoke6/manual / Retest smoke6 | 17365.8 | 17368.2 | -0.0% | 142.43 | 161.00 | -13.0% | 4.6 | 4.2 | -7.6% | ok | ok |
| pd_long_ctx_c4 | Long context concurrency | 8192 | 1024 | 4 | Retest smoke6 / Retest smoke6 | 50985.1 | 43105.7 | +15.5% | 117.04 | 121.66 | -3.9% | 11.9 | 13.5 | +14.0% | ok | ok |
| pd_kv_stress_c8 | KV RDMA sustained load | 4096 | 256 | 8 | Retest smoke6 / Retest smoke6 | 26322.7 | 25043.7 | +4.9% | 106.88 | 92.31 | +13.6% | 25.0 | 25.0 | -0.2% | ok | ok |
| pd_decode_thr_c48 | Decode high concurrency+ | 128 | 128 | 48 | Retest ar_showcase / Retest ar_showcase | 2435.5 | 2858.7 | -17.4% | 205.39 | 204.60 | +0.4% | 168.9 | 165.8 | -1.9% | ok | ok |
| pd_mixed_c16 | Mixed high concurrency | 2048 | 512 | 16 | Retest ar_showcase / Retest ar_showcase | 18374.7 | 19630.7 | -6.8% | 132.04 | 130.16 | +1.4% | 68.3 | 66.8 | -2.1% | ok | ok |
| pd_mixed_c32 | Mixed peak | 2048 | 512 | 32 | Retest ar_showcase / Retest ar_showcase | 29070.9 | 29010.0 | +0.2% | 140.12 | 136.04 | +2.9% | 102.7 | 106.3 | +3.5% | ok | ok |
| pd_decode_long_c16 | Long-output decode | 128 | 512 | 16 | Retest ar_showcase / Retest ar_showcase | 957.6 | 959.1 | -0.2% | 179.96 | 180.03 | -0.0% | 78.7 | 78.7 | +0.0% | ok | ok |

## Reproduction Steps and Scripts

### 1. Environment and Topology Configuration

```bash
source /root/jianxiong/pd/env_pd_ep16_2p2d.sh
# Key variables (env_pd_ep16_2p2d.sh):
#   P: 10.6.142.10:30000 + 10.6.142.14:30001
#   D: 10.6.142.11:30000 + 10.6.142.15:30001
#   Router: http://10.6.142.10:32050
#   TP_SIZE=8 EP_SIZE=8 / node, TOTAL_GPUS=32
#   NCCLEP low_latency, KV=NIXL RDMA (mlx5)
```

### 2. Start / Stop Cluster (Including AR Mode Switching)

AR parameter passing details in **"AR OFF vs AR ON DDP ON — How Parameters Are Passed"** section above.

```bash
# AR OFF
export NCCL_AR_MODE=ar_off
bash /root/jianxiong/run_dsv4_ncclep_pd_ep16_2p2d.sh start

# AR ON DDP ON (start auto-kills old processes first)
export NCCL_AR_MODE=ar_on_ddp_on
bash /root/jianxiong/run_dsv4_ncclep_pd_ep16_2p2d.sh start

# Check health / stop
bash /root/jianxiong/run_dsv4_ncclep_pd_ep16_2p2d.sh status
bash /root/jianxiong/run_dsv4_ncclep_pd_ep16_2p2d.sh stop
```

Scripts: `/root/jianxiong/run_dsv4_ncclep_pd_ep16_2p2d.sh` · Translation logic: `nccl_ar_overlay()` / `sglang_ncclep_block()` in `pd/env_pd_ep16_2p2d.sh`

### 3. Full 17 Cases × 2 AR Modes Benchmark

```bash
# Run on control host 10.6.142.11 in foreground (recommended for live output)
bash /root/jianxiong/bench_dsv4_ncclep_pd_2p2d_ar.sh all

# AR showcase Group F only (4 cases × 2 AR, ~30–60 min), merge into existing directory
bash /root/jianxiong/bench_dsv4_ncclep_pd_2p2d_ar.sh ar_showcase /root/jianxiong/logs/bench_pd_2p2d_ar_20260627_234625

# AR OFF or AR ON only (full 17 cases)
bash /root/jianxiong/bench_dsv4_ncclep_pd_2p2d_ar.sh ar_off
bash /root/jianxiong/bench_dsv4_ncclep_pd_2p2d_ar.sh ar_on_ddp_on
```

Script: `/root/jianxiong/bench_dsv4_ncclep_pd_2p2d_ar.sh`

Output directories: `/root/jianxiong/logs/bench_pd_2p2d_ar_20260627_234625/AR_OFF` and `/root/jianxiong/logs/bench_pd_2p2d_ar_20260627_234625/AR_ON_DDP_ON`; each case produces `<case>.jsonl` + `<case>.log` + `summary.md`

### 4. Long-Context Failed Case Retest (smoke method)

For `pd_long_ctx_c1/c4` and `pd_kv_stress_c8`, retest with long-context-specific config and merge into existing result directory:

```bash
# Retest all 6 runs (3 cases × AR OFF + AR ON)
bash /root/jianxiong/bench_dsv4_ncclep_pd_2p2d_smoke6.sh /root/jianxiong/logs/bench_pd_2p2d_ar_20260627_234625

# Retest AR ON only
bash /root/jianxiong/bench_dsv4_ncclep_pd_2p2d_smoke6.sh /root/jianxiong/logs/bench_pd_2p2d_ar_20260627_234625 ar_on_ddp_on

# Hardcoded retest parameters (in script, consistent with smoke validation):
#   CONTEXT_LENGTH=10240
#   MEM_FRACTION_STATIC=0.55
#   P_MAX_RUNNING_REQUESTS=4
#   --ready-check-timeout-sec 600
#   --warmup-requests 1
#   --tokenizer /root/menyu/models/DeepSeek-V4-Flash-FP8
```

Script: `/root/jianxiong/bench_dsv4_ncclep_pd_2p2d_smoke6.sh` (alternative: `bench_dsv4_ncclep_pd_2p2d_retry3.sh`)

### 5. AR Showcase Group F Retest

Initial `all` run already includes Group F; to rerun showcase cases separately and overwrite jsonl:

```bash
bash /root/jianxiong/bench_dsv4_ncclep_pd_2p2d_ar.sh ar_showcase /root/jianxiong/logs/bench_pd_2p2d_ar_20260627_234625
```

Script: `/root/jianxiong/bench_dsv4_ncclep_pd_2p2d_ar.sh` (`run_ar_showcase_cases` function)

Output: `summary_ar_showcase.md`, `fail_count_ar_showcase`, per-case `.jsonl` / `.log`

### 6. Generate / Update HTML Report

```bash
python3 /root/jianxiong/reports/generate_dsv4_pd_2p2d_ar_report.py /root/jianxiong/logs/bench_pd_2p2d_ar_20260627_234625
```

Report output: `/root/jianxiong/reports/dsv4_pd_2p2d_ar_benchmark_report.html`

Report reads metrics from `summary.md` first; if cases were updated via retest, automatically merges latest valid results from `.jsonl` and annotates sources in "Retest Records" section.

### 7. Key Fix Records (Debugging Reference)

- **NCCLEP LL buffer race:** Full TP rank barrier after DeepGEMM JIT; ncclep.py shape mismatch auto-realloc (`compile_utils.py`, `ncclep.py`)
- **Tokenizer 401:** Bench script adds `--tokenizer /root/menyu/models/DeepSeek-V4-Flash-FP8`
- **Long-context OOM:** ctx 8192 insufficient (ISL+OSL=9216); prefill mem 0.70 OOM → retest uses 0.55
- **Retest env override:** Must hardcode long-context parameters; cannot use `${VAR:-10240}` (env already sets 8192, won't apply)

### Script Inventory

| Script | Purpose |
|--------|---------|
| `pd/env_pd_ep16_2p2d.sh` | Topology, ports, NCCL AR mode, model paths |
| `run_dsv4_ncclep_pd_ep16_2p2d.sh` | 4-job start/stop + router + health check |
| `bench_dsv4_ncclep_pd_2p2d_ar.sh` | 17-case benchmark; `ar_showcase` for Group F only |
| `bench_dsv4_ncclep_pd_2p2d_smoke6.sh` | Long-context 3-case retest (smoke method) |
| `bench_dsv4_ncclep_pd_2p2d_retry3.sh` | Long-context retest (same as smoke6, legacy) |
| `reports/generate_dsv4_pd_2p2d_ar_report.py` | Generate this HTML report |

## Full Script Text

The original HTML report embeds complete script sources read from disk at report generation time for offline reproduction and audit. Scripts are available at:

| Script | Lines | Absolute Path |
|--------|-------|---------------|
| `pd/env_pd_ep16_2p2d.sh` | 167 | `/root/jianxiong/pd/env_pd_ep16_2p2d.sh` |
| `run_dsv4_ncclep_pd_ep16_2p2d.sh` | 248 | `/root/jianxiong/run_dsv4_ncclep_pd_ep16_2p2d.sh` |
| `bench_dsv4_ncclep_pd_2p2d_ar.sh` | 259 | `/root/jianxiong/bench_dsv4_ncclep_pd_2p2d_ar.sh` |
| `bench_dsv4_ncclep_pd_2p2d_smoke6.sh` | 141 | `/root/jianxiong/bench_dsv4_ncclep_pd_2p2d_smoke6.sh` |
| `bench_dsv4_ncclep_pd_2p2d_retry3.sh` | 145 | `/root/jianxiong/bench_dsv4_ncclep_pd_2p2d_retry3.sh` |
| `reports/generate_dsv4_pd_2p2d_ar_report.py` | 789 | `/root/jianxiong/reports/generate_dsv4_pd_2p2d_ar_report.py` |

Full script contents are in the source HTML file `dsv4_pd_2p2d_ar_benchmark_report.html` (sections under "脚本全文").
