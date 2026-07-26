# gpu-reaper

Find wasted GPU allocations on a Slurm cluster. Alert on them, drain the nodes,
or cancel the job — in that order, and only as far as you tell it to go.

```
make demo
```

No cluster, no GPUs, no risk. Fake `squeue` on `PATH`, simulated telemetry,
observe mode. You will watch two jobs get classified as `hung` and escalate
from `alert` to `drain` in about thirty seconds.

---

## The problem

A researcher requests 16 H100s for a 48-hour run. Four hours in, the training
script deadlocks on a NCCL all-reduce because one rank died. The allocation is
still held. Slurm is content — the job is `RUNNING`, the nodes are busy, the
queue is moving.

Nothing in Slurm notices that the GPUs are doing no work, because Slurm does not
look at GPUs. It looks at allocations.

That is 16 GPUs × 44 hours = **704 GPU-hours** of a resource with a waiting
queue, consumed by a process doing nothing.

## What it does

```
┌──────────────┐        ┌──────────────┐
│   squeue     │        │  nvidia-smi  │
│  slurmrestd  │        │   (per node) │
└──────┬───────┘        └──────┬───────┘
       │ running jobs          │ util, mem, power, PIDs
       │ nodes, TRES           │
       └───────────┬───────────┘
                   ▼
         ┌───────────────────┐
         │   policy engine   │  window · warmup · gap detection
         │                   │  signature classification
         └─────────┬─────────┘
                   │ Finding{verdict, signature, wasted_gpu_hours}
       ┌───────────┼───────────┬──────────────┐
       ▼           ▼           ▼              ▼
    ┌──────┐  ┌────────┐  ┌────────┐   ┌────────────┐
    │ log  │  │ slack  │  │ drain  │   │  scancel   │
    └──────┘  └────────┘  └────────┘   └────────────┘
                            └── only in mode: enforce ──┘
```

## Why it will not kill your job

Killing a healthy job is much worse than letting a wasted one run another hour.
A researcher whose 40-hour run dies at hour 39 loses the run *and* their trust in
the tool. A wasted allocation surviving one more cycle costs one more cycle.

Every default follows from that asymmetry:

| Guard | Behaviour |
| --- | --- |
| **Observe by default** | `mode: observe` never cancels or drains. Enforcement is opt-in. |
| **Warmup** | Jobs younger than `warmup` (15m) are never judged — container pulls and dataset staging legitimately show zero utilization. |
| **Sustained window** | A breach must hold across the entire `window` (20m). One busy sample clears the finding. |
| **Peak, not mean** | Evaluated on peak utilization. Any real work anywhere in the window means alive. |
| **Gap detection** | A hole in the samples larger than `max_sample_gap` is treated as a collector fault, not idleness. *This is the guard that stops a monitoring outage from cancelling the cluster.* |
| **Signature gating** | `starved` and `unknown` never escalate past `alert`. A slow dataloader is a tuning problem, not a capital offence. |
| **Cancel needs history** | A job cannot be cancelled unless a previous cycle already alerted and drained. No config change can jump straight to termination. |
| **Shared nodes skipped** | Where a node hosts several GPU jobs, per-GPU attribution is ambiguous, so the job is skipped rather than guessed at. |
| **Exemptions** | By user, account, partition, QOS, or job-name regex. |

Each of these is covered by a test in
[`internal/policy/policy_test.go`](internal/policy/policy_test.go) — that file is
the best short description of what the tool will and will not do.

## Signatures

Low utilization has several causes and they deserve different responses.

| Signature | Evidence | Meaning | Max escalation |
| --- | --- | --- | --- |
| `idle` | No memory held, no processes, idle power draw | Allocation is empty | `cancel` |
| `hung` | Memory held, processes resident, **zero** compute | Deadlocked collective, stuck ring | `cancel` |
| `starved` | Memory held, low but nonzero compute | Dataloader bottleneck — real waste, but a tuning problem | `alert` |
| `unknown` | Breaching, no signature match | Not modelled; never acted on | `alert` |

## Configuration

```yaml
mode: observe            # observe | enforce
interval: 2m
metrics_addr: ":9835"

slurm:
  source: squeue         # squeue | rest
  # rest_url: http://slurmrestd:6820
  # rest_version: v0.0.42
  # username: slurm
  # token_env: SLURM_JWT

gpu:
  source: nvidia-smi     # nvidia-smi | simulator

thresholds:
  util_pct: 15           # SM utilization below this counts as a breach
  mem_held_fraction: 0.05
  idle_power_watts: 60
  window: 20m            # breach must be sustained this long
  warmup: 15m            # grace period after job start
  min_samples: 8
  max_sample_gap: 3m     # bigger hole ⇒ assume collector fault

stages:
  - { after: 0m,  verdict: alert }
  - { after: 60m, verdict: drain }
  # - { after: 4h, verdict: cancel }   # requires mode: enforce

exemptions:
  users: [ci-bot]
  partitions: [interactive]
  name_pattern: "^(debug|jupyter)-"

slack:
  webhook_env: SLACK_WEBHOOK_URL
  min_verdict: alert
```

`interval` must not exceed `window`, or a verdict could never gather enough
samples. The config validator rejects that at startup rather than running
silently forever.

## Metrics

Exposed at `/metrics`. **No `job_id` label anywhere** — a busy cluster cycles
through millions of job IDs and a per-job label set would take Prometheus down
before it told anyone anything. Per-job detail is in the structured log.

| Metric | Type | Labels |
| --- | --- | --- |
| `gpu_reaper_findings` | gauge | `verdict`, `signature`, `partition` |
| `gpu_reaper_wasted_gpu_hours` | gauge | `partition`, `signature` |
| `gpu_reaper_gpus_held_breaching` | gauge | `partition`, `signature` |
| `gpu_reaper_jobs_evaluated` | gauge | — |
| `gpu_reaper_actions_total` | counter | `actor`, `verdict`, `enforced` |
| `gpu_reaper_action_errors_total` | counter | `actor` |
| `gpu_reaper_scrape_duration_seconds` | histogram | — |
| `gpu_reaper_scrape_errors_total` | counter | `source` |
| `gpu_reaper_skipped_shared_node_total` | counter | — |

The number worth putting on a dashboard:

```promql
sum(gpu_reaper_wasted_gpu_hours) by (partition, signature)
```

A rising `gpu_reaper_skipped_shared_node_total` means your cluster packs
multiple GPU jobs per node and this tool is not covering them — see Limitations.

## Deployment

Runs per GPU node (a DaemonSet, or a systemd unit) so `nvidia-smi` is local.
Only the node's own GPUs are read; job state comes from the controller.

```bash
gpu-reaper --config /etc/gpu-reaper/config.yaml
gpu-reaper --config ... --once     # single cycle, for cron or testing
```

Enforcement needs the daemon to run as a user permitted to `scancel` and
`scontrol update`. Start in `observe`, watch `gpu_reaper_findings` for a week,
and only then decide whether the findings are trustworthy enough to act on.

## Limitations

Stated plainly, because they bound what the tool can claim:

- **Shared-node attribution.** GPU samples identify a node, not a job. Where one
  node hosts several GPU jobs, the reaper skips them rather than guess. Correct
  attribution requires walking the Slurm cgroup hierarchy
  (`/sys/fs/cgroup/.../slurm/uid_*/job_*/`) to map GPU PIDs back to job IDs.
  That is deliberately out of scope here and is the subject of a companion tool.
- **`nvidia-smi`, not DCGM.** Utilization is coarse: a kernel occupying one SM
  reports the same 100% as a saturated device. That is why *high* utilization is
  never treated as proof of health — only *low* utilization as evidence of a
  problem. DCGM would give real SM occupancy and tensor-core activity.
- **NVIDIA only.** No ROCm or Habana backend.
- **No MIG awareness.** MIG-partitioned GPUs report per-instance; not modelled.
- **Single-cluster.** One controller per daemon.

## Development

```bash
make test           # race detector + coverage
make demo           # the daemon, no hardware needed
make demo-once      # a single cycle
make run-scenarios  # every scenario's verdict side by side
```

Adding a telemetry backend means implementing `gpu.Source`; adding a job source
means `slurm.Source`. Both are small interfaces, and the simulator is the
reference implementation.

## License

MIT
