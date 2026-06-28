# DeepSeek-V4-Flash Training Performance Benchmark Report

**Nodes:** 10.6.142.10 + 10.6.142.11 (2×8 GPU = 16 ranks) · **Model reference:** `/root/menyu/models/DeepSeek-V4-Flash-FP8` · **Runs:** 22 (11 cases × AR OFF/ON) · **Generated:** 2026-06-28T15:23:33

## Test Objectives

Evaluate step throughput for two DeepSeek-V4-Flash training workloads on 2 nodes / 16 GPUs, comparing **AR OFF** (ECMP baseline) vs **AR ON DDP ON** (Adaptive Routing + SuperNIC out-of-order reordering).

- **MoE EP training** — Simulates MoE dispatch/combine; primary communication is `All-to-All`
- **Dense DDP training** — Simulates Attention backbone; primary communication is `AllReduce`

## 1. Hardware Topology

```
┌─────────────────────────────────────┐     bond0 (10.6.142.0/24)     ┌─────────────────────────────────────┐
│  R6KD-CX8aaS-GPU-10  (10.6.142.10)  │◄─────── IB RDMA cross-node ──►│  R6KD-CX8aaS-GPU-11  (10.6.142.11)  │
│  8 × NVIDIA GPU (~73 GB HBM each)   │                               │  8 × NVIDIA GPU (~73 GB HBM each)   │
│  GPU PCI: 06/09/76/79/86/89/F6/F9   │                               │  Same topology                      │
└─────────────────────────────────────┘                               └─────────────────────────────────────┘
         rank 0–7                                                              rank 8–15
                    └────────────────── 16 ranks total ──────────────────────┘
```

| Attribute | Value |
|-----------|-------|
| Master node | 10.6.142.10 (R6KD-CX8aaS-GPU-10) |
| Worker node | 10.6.142.11 (R6KD-CX8aaS-GPU-11) |
| GPUs per node | 8 |
| Total GPUs (world_size) | 16 |
| Memory | ~73 GB / GPU |
| Compute precision | BF16 |

## 2. Network Topology

Each node has 8 IB HCAs; NCCL uses 4 of them for collective communication:

| HCA | Interface | NCCL Selected |
|-----|-----------|---------------|
| mlx5_0 | enp115s0f0np0 | ✓ |
| mlx5_2 | enp3s0f0np0 | ✓ |
| mlx5_4 | enp243s0f0np0 | ✓ |
| mlx5_6 | enp131s0f0np0 | ✓ |
| mlx5_1/3/5/7 | corresponding f1 ports | Not selected |

### Key NCCL / MPI Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `NCCL_IB_HCA` | mlx5_0,mlx5_2,mlx5_4,mlx5_6 | 4 HCAs × 2 nodes |
| `NCCL_P2P_DISABLE` | 1 | Disable GPU P2P; force IB |
| `NCCL_SHM_DISABLE` | 1 | Disable SHM |
| `NCCL_SOCKET_IFNAME` | bond0 | OOB / bootstrap |
| `NCCL_IB_QPS_PER_CONNECTION` | 8 | QPs per connection |
| `NCCL_ALGO` | Ring | Ring collective |
| `UCX_TLS` | ib | OpenMPI UCX over IB |
| `LD_PRELOAD` | libnccl.so.2 (custom build) | `/root/jianxiong/nccl/build/lib/` |

## 3. Software Stack

| Component | Version / Path |
|-----------|----------------|
| Python | 3.12 (conda env `pcie`) |
| PyTorch | 2.12.0+cu130 |
| Launcher | OpenMPI `mpirun` + UCX PML |
| hostfile | `/root/jianxiong/5k_test/hostfile_2node_10_11` |
| Run script | `/root/jianxiong/5k_test/run_dsv4_train_2node_10_11.sh` |

## 4. Parallel Training Topology

### 4.1 Rank Mapping

```
hostfile_2node_10_11:
  10.6.142.10  slots=8   →  rank 0–7   (local GPU 0–7)
  10.6.142.11  slots=8   →  rank 8–15  (local GPU 0–7)

mpirun: -npernode 8 --map-by slot --bind-to none
MASTER_ADDR=10.6.142.10  MASTER_PORT=29502
```

### 4.2 MoE EP=16 (M1–M6, M8)

```
EP Group (size=16):
  Node .10: rank 0,1,2,3,4,5,6,7
  Node .11: rank 8,9,10,11,12,13,14,15

Each MoE layer:
  LayerNorm → All-to-All(dispatch) → expert FFN → All-to-All(combine) → residual
  2 All-to-All per layer, completed cross-node over IB
```

### 4.3 MoE EP=8 (M7)

```
EP Group 0: rank 0–7   (DP replica 0)
EP Group 1: rank 8–15  (DP replica 1)
All-to-All only within 8-rank groups; cross-node traffic halved
```

### 4.4 Dense DDP (D1–D3)

```
16 ranks global DDP: full model replica per rank
Gradient AllReduce after backward (Ring, 16 GPU)
```

## 5. AR OFF vs AR ON DDP ON

| Parameter | AR OFF | AR ON DDP ON |
|-----------|--------|--------------|
| `NCCL_IB_ADAPTIVE_ROUTING` | 0 | 1 |
| `NCCL_IB_OOO_RQ` | 0 | 1 |
| `NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME` | 0 | 1 |
| `NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS` | 0 | 1 |
| Other NCCL/MPI parameters | Identical | Identical |

**AR OFF:** ECMP multi-path hash-based splitting; packets may arrive out of order.

**AR ON DDP ON:** Adaptive Routing + receiver-side out-of-order reordering; suited for multi-stream elephant flows such as All-to-All.

Each case runs both **AR OFF** and **AR ON DDP ON** with all other variables held constant.

## 6. Case Design

### 6.1 MoE EP Training (M1–M8)

Reference DSV4-Flash: hidden=4096, ffn=2048, topk=6, experts=256. Benchmark uses **128 experts** (each rank holds full expert replicas; 256 would OOM).

| Case | Description | EP | seq | bs/gpu | layers | hidden | ffn | experts |
|------|-------------|-----|-----|--------|--------|--------|-----|---------|
| M1 | Reduced-dimension baseline | 16 | 512 | 1 | 2 | 2048 | 1024 | 128 |
| M2 | Long sequence | 16 | 1024 | 1 | 2 | 2048 | 1024 | 128 |
| M3 | Large micro-batch | 16 | 512 | 2 | 2 | 2048 | 1024 | 128 |
| M4 | DSV4 true hidden/ffn | 16 | 512 | 1 | 2 | 4096 | 2048 | 128 |
| M5 | DSV4 + long sequence | 16 | 1024 | 1 | 2 | 4096 | 2048 | 128 |
| M6 | Deeper MoE stack | 16 | 512 | 1 | 4 | 4096 | 2048 | 128 |
| M7 | EP topology variant | 8 | 512 | 1 | 2 | 4096 | 2048 | 128 |
| M8 | Long-context stress test | 16 | 2048 | 1 | 2 | 4096 | 2048 | 128 |

### 6.2 Dense Attention DDP (D1–D3)

Matches DSV4 attention: hidden=4096, heads=64, vocab=129280; uses 4 layers instead of full 43 layers.

| Case | Description | seq | bs/gpu | layers |
|------|-------------|-----|--------|--------|
| D1 | Baseline | 512 | 1 | 4 |
| D2 | Long sequence | 1024 | 1 | 4 |
| D3 | Large batch | 512 | 2 | 4 |

### 6.3 Timing Method

- Warmup: 2 steps (excluded from statistics)
- Benchmark: 6 steps (CUDA-synchronized timing)
- `tokens_per_step = batch_per_gpu × seq_len × 16`
- `tokens/s = tokens_per_step / (ms_step_mean / 1000)`

## 7. Execution Steps

1. **Environment setup:** Sync `bench_*.py`, `hostfile_2node_10_11` to .10 / .11
2. **Launch benchmark:** `bash /root/jianxiong/5k_test/run_dsv4_train_2node_10_11.sh`
3. **Per-case execution** (22 runs total):
   - SSH to MASTER (.10), activate conda `pcie`
   - Inject NCCL environment (AR OFF or AR ON DDP ON)
   - `mpirun` launches 16 processes (UCX/IB, passing NCCL variables)
   - Run `bench_moe_train_dsv4.py` or `bench_dense_train_dsv4.py`
   - rank 0 writes JSON → SCP back to control host → append CSV
4. **Case order:** M1(AR_OFF→ON) … M8(AR_OFF→ON) → D1 … D3
5. **Generate report:** `generate_dsv4_train_2node_report.py`

## 8. Test Results — MoE EP Training

| Case | EP | Seq | BS | L | ms/step OFF | ms/step ON | Δ ms | tok/s OFF | tok/s ON | Δ tok/s |
|------|-----|-----|----|---|-------------|------------|------|-----------|----------|---------|
| **M1** (reduced-dim baseline, h=2048, E=128) | 16 | 512 | 1 | 2 | 272.5 | 259.4 | -4.8% | 30,059 | 31,579 | +5.1% |
| **M2** (long sequence, sl=1024) | 16 | 1024 | 1 | 2 | 281.8 | 283.8 | +0.7% | 58,151 | 57,722 | -0.7% |
| **M3** (large micro-batch, bs=2) | 16 | 512 | 2 | 2 | 292.8 | 301.3 | +2.9% | 55,964 | 54,373 | -2.8% |
| **M4** (DSV4 true hidden/ffn, h=4096, ffn=2048) | 16 | 512 | 1 | 2 | 307.9 | 302.7 | -1.7% | 26,608 | 27,062 | +1.7% |
| **M5** (DSV4 + long sequence) | 16 | 1024 | 1 | 2 | 325.2 | 318.9 | -1.9% | 50,385 | 51,371 | +2.0% |
| **M6** (deeper MoE stack, L=4) | 16 | 512 | 1 | 4 | 579.8 | 582.9 | +0.5% | 14,129 | 14,055 | -0.5% |
| **M7** (EP topology variant, EP=8, 2 DP groups) | 8 | 512 | 1 | 2 | 287.3 | 289.9 | +0.9% | 28,516 | 28,258 | -0.9% |
| **M8** (long-context stress, sl=2048) | 16 | 2048 | 1 | 2 | 326.8 | 323.6 | -1.0% | 100,270 | 101,258 | +1.0% |

## 9. Test Results — Dense DDP Training

| Case | EP | Seq | BS | L | ms/step OFF | ms/step ON | Δ ms | tok/s OFF | tok/s ON | Δ tok/s |
|------|-----|-----|----|---|-------------|------------|------|-----------|----------|---------|
| **D1** (Attention DDP baseline, sl=512, L=4) | — | 512 | 1 | 4 | 259.7 | 258.8 | -0.4% | 31,543 | 31,654 | +0.4% |
| **D2** (Attention long sequence, sl=1024) | — | 1024 | 1 | 4 | 271.0 | 270.2 | -0.3% | 60,449 | 60,633 | +0.3% |
| **D3** (Attention large batch, bs=2) | — | 512 | 2 | 4 | 270.5 | 269.9 | -0.2% | 60,561 | 60,710 | +0.2% |

## 10. Conclusions

- **MoE All-to-All:** AR ON yields **+1% ~ +5%** throughput improvement on M1/M4/M5/M8; M1 baseline shows the largest gain (273→259 ms/step, +5.1%).
- **Dense AllReduce:** AR impact **<0.5%**; compute-bound, communication is not the bottleneck.
- **MoE layer count doubled (M6):** ms/step ~308ms → ~580ms, nearly linear (4 layers × 2 All-to-All per layer).
- **EP=8 vs EP=16 (M7 vs M4):** tok/s similar (~28K vs ~27K); EP topology has limited impact on 2 nodes.

## 11. Known Limitations

- Not real-weight training: DSV4-Flash FP8 checkpoint not loaded; only architecture and communication patterns simulated.
- Experts not sharded: each rank holds 128 expert replicas (real EP would shard across 16 ranks).
- MoE routing is simplified Python implementation; compute share higher than production.
- AR comparison only at QPS=8; no QPS sweep performed.
- Single 6-step sample; no multi-run confidence intervals.

## 12. Output Files

| Type | Path |
|------|------|
| Run script | `/root/jianxiong/5k_test/run_dsv4_train_2node_10_11.sh` |
| Report generator | `/root/jianxiong/5k_test/generate_dsv4_train_2node_report.py` |
| Summary CSV | `/root/jianxiong/5k_test/results/dsv4_train_2node_summary_20260628_144914_fixed.csv` |
| Per-case JSON | `/root/jianxiong/5k_test/results/dsv4_*_TIMESTAMP.json` |
| Full log | `/root/jianxiong/5k_test/logs/dsv4_train_2node_full_run.log` |
