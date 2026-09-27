# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This release responds to an external audit that reproduced, with fakes and
scratch tests, several ways the v0.1.0 guardrails could be bypassed. Every
reproduced scenario is now a regression test. As before, nothing here has been
run against a real Slurm cluster or a real GPU; all testing uses a simulator,
fake binaries, and synthetic fixtures.

### Fixed — safety

- **A telemetry outage could turn into a Cancel.** The breach clock and
  escalation history survived periods with no observations, and pre-outage
  samples were pruned before the gap check could see the hole. The audit's
  scenario (3h50m of breach, a 40-minute outage, one snapshot) produced
  `cancel`. Now any hole, stale newest sample, or uncovered window resets the
  breach clock *and* the history, so the window must be covered again and the
  ladder climbed again from Alert. The same applies when a job is skipped
  (shared node, partial evidence).
- **Unreadable nvidia-smi values were parsed as 0**, so `[N/A]` telemetry
  looked like an idle GPU and was cancel-eligible. Every field is now parsed as
  (value, known). Unreadable utilization or memory makes the job unjudgeable.
  Unknown power or process data can never satisfy `idle` or `hung`. MIG-enabled
  GPUs (`mig.mode.current`) are refused. A malformed row fails the sample
  instead of silently dropping a GPU. `gpu.Sample` validity flags default to
  "unknown", so a new backend that forgets one fails safe.
- **A per-node daemon judged, drained and cancelled whole multi-node jobs on
  its own GPUs alone.** A job is now judged only with telemetry for every one
  of its nodes, covering at least as many GPUs as it holds. A per-node daemon
  therefore judges only single-node jobs on its own node (and asks squeue only
  for those, via `--nodelist`). Skips are counted.
- **One snapshot could confirm a "sustained" breach.** `min_samples` counted
  per-GPU samples, so one 8-GPU snapshot passed it. It now counts distinct
  observation times per node, and the evidence must cover the window.
- **Warmup samples counted as evidence**, so a job was judged, and its waste
  counted, from its start. Samples taken during warmup are now discarded.
- **squeue start times were parsed as UTC but printed in local time**, which
  bypassed warmup west of UTC and blinded the tool east of it. squeue now runs
  with `TZ=UTC` and `SLURM_TIME_FORMAT=standard`. A start time that does not
  parse rejects the record instead of becoming "age 0". The REST source no
  longer turns an unset `start_time` into 1970.
- **Cancel trusted the computed verdict, not the executed action.** Cancel now
  needs a drain that was executed (or dry-run) and confirmed back to the
  engine at least one window earlier. A failed drain is retried and blocks
  Cancel. Drain itself now needs an Alert on a previous cycle.
- **A requeued job inherited its previous incarnation's history.** State is
  now per JobID *and* StartTime.
- **`hung` was cancel-eligible**, contradicting the code's own warning that it
  also describes a long checkpoint write. It now stops at Drain unless
  `allow_cancel_signatures` lists it (default: `[idle]`).
- **A `|` in a job name shifted squeue fields**, letting any user hide a job or
  borrow another user's exemption. Name is now the last field and split with a
  fixed count; malformed lines and duplicated job IDs are rejected and counted.
- **The cancel reason was dropped.** Cancel now sets the job's AdminComment
  first and refuses to cancel if that fails. `scancel --full`, which only
  affects non-SIGKILL signals, is no longer passed.
- **Drain was re-issued every cycle.** It is now issued once per breach
  episode. If the breach clock resets (the job recovers, or the evidence has a
  hole) and the job climbs back to Drain, the drain is issued again for the
  same job incarnation (`TestDrainIsReissuedAfterTheBreachClockResets`).

### Fixed — other

- The Slack webhook secret (its URL path) no longer appears in error logs.
- Config decoding is strict: unknown keys are errors. Thresholds and their
  relationships are validated (`interval < max_sample_gap < window`,
  `min_samples × interval ≤ window`, `0 < util_pct ≤ 100`, drain/cancel stages
  at least one window, and more; see README). The stage rule checks the
  stages the engine will run, including the defaults when `stages:` is
  omitted: checking only the file let `window: 90m` load against the default
  60m drain, which then fired one cycle after the first alert.
- `exemptions.name_pattern` was silently ignored: the field had no YAML tag,
  so yaml.v3 expected `namepattern`. Found while adding strict decoding (not in
  the audit); confirmed with a scratch program against yaml.v3 v3.0.1.
- `gpu_reaper_actions_total` hard-coded `enforced="true"`, so observe mode
  reported enforcement. The label is now `outcome` (`acted` or `dry_run`), and
  no-ops are not counted. **Breaking for dashboards using `enforced`.**
- GRES types with uppercase, hyphens or dots (`gres/gpu:A100=4`,
  `gres/gpu:1g.10gb=2`) parsed as 0 GPUs. Hostlists with suffixes, several
  bracket groups or dots were returned unexpanded; they now expand, and
  anything unparseable is rejected rather than kept as a literal. Expansion is
  capped at 100,000 names, checked before anything is allocated and without
  integer overflow (a span ending at 2^63−1 used to slip past the cap and
  allocate until memory ran out).
- Code comments that described things that did not exist or were not measured
  (a DCGM fallback, NVML shutdown, "a few milliseconds per node", "fixed
  widths", "one rung at a time") were corrected.

### Added

- **DCGM source (`gpu.source: dcgm`), unverified on real hardware.** Reads
  dcgm-exporter's Prometheus text endpoint, per node or centrally
  (`url: http://{node}:9400/metrics`). Metric and label names were checked
  against dcgm-exporter's source at commit fafd151 and its documentation; it is
  tested only against synthetic fixtures. Missing, stale, non-finite,
  out-of-range or disagreeing values are unknown, never idle. It carries
  `hpc_job` and pod labels, uses `hpc_job` only to *stop* a judgement on
  conflict, and cannot see processes, so its findings never pass Alert.
- `gpu_reaper_wasted_gpu_seconds_total` (counter; the old gauge could not be
  summed and over-counted multi-node jobs per node),
  `gpu_reaper_skipped_partial_evidence_total`,
  `gpu_reaper_skipped_attribution_conflict_total`,
  `gpu_reaper_source_rejected_records_total`,
  `gpu_reaper_gpu_jobs_without_samples`,
  `gpu_reaper_last_successful_cycle_timestamp_seconds`, and `/readyz`.
- `slurm.token_file` (re-read per request), `slurm.*_path` and
  `gpu.nvidia_smi_path` (in enforce mode, absolute paths are required for
  scancel, scontrol, and squeue or nvidia-smi when that source is in use,
  since each can steer a drain or cancel), `allow_cancel_signatures`,
  `slack.remind_every` (Slack now posts on verdict changes, not every cycle),
  `--cycles`, `--node-name`, and a per-cycle summary log line.
- Tests for every package under `cmd/` and `internal/`. `hack/demosvg`, the
  dev-only generator behind `make demo-record`, has none. Module coverage went
  from 30.1% (the audit's figure) to 89.7%, measured on this tree with
  `go test -race -coverpkg=./... -coverprofile=coverage.out ./...` and then
  `go tool cover -func=coverage.out` (Go 1.26.3, darwin/arm64; the same figure
  without `-race`). That total counts `hack/demosvg` at 0%. Over `cmd/` and
  `internal/` alone (`-coverpkg=./cmd/...,./internal/...`) it is 93.9%. The
  tests include a seeded randomized invariant test, fuzz targets for the
  hostlist, GRES and squeue parsers, fake-binary tests for
  nvidia-smi/squeue/scancel/scontrol, daemon-loop tests with fake sources,
  tests of the config-to-daemon wiring (source scope, `--nodelist`, tokens,
  `allow_cancel_signatures` reaching the engine), and a bounded end-to-end
  `run()` against the demo's fake squeue. Each safety guard listed in the
  README was disabled in turn in a scratch copy, by hand, and at least one
  test failed each time.
- `demo/check.sh`, which asserts each simulator scenario's outcome (for
  `flaky`, only that it never passes Alert, since whether a burst lands
  depends on the cycle count), and
  `make demo-record`, which captures `docs/demo.log` and generates
  `docs/demo.svg` from it with `hack/demosvg`.
- CI: gofmt, staticcheck, govulncheck, a coverage floor (80%), shellcheck, the
  demo scenarios under UTC, America/Los_Angeles and Asia/Tokyo, and a
  container build; `permissions: contents: read`; actions pinned by SHA; Go
  1.26.x with `check-latest`. A tag-triggered release workflow that builds
  linux/amd64 and linux/arm64 binaries with SHA256SUMS (not yet run).

### Changed

- `config.example.yaml` is now a production example (default stages, cancel
  off, no simulator block, no demo exemption). It previously drained after 30
  seconds. A test keeps it that way.
- The demo's fake squeue prints local time like real squeue, with start times
  fixed for the whole run. `make demo-once` now reports nothing, by design:
  one snapshot is never a breach.
- `docs/demo.svg` was an edited transcript that dropped the text showing the
  verdict fired on a zero-dwell snapshot. It is now generated from a real
  captured run, with the filtering stated in the image.
  `docs/escalation.svg` no longer shows a nonexistent `nccl-stall`
  signature or calls drain reversible.
- The container runtime is distroless (glibc) instead of Alpine (musl).
- README: claims now name the tests behind them; limitations added (drain is
  never undone, MIG, multi-node jobs per node, DCGM unverified, privileges,
  process visibility, node names, including that a per-node daemon cannot
  show a node-name mismatch in `gpu_reaper_gpu_jobs_without_samples`); related
  work added, compared only against what was read; the cgroup v2 path example
  corrected per Slurm's cgroup.conf documentation. Companion tools are
  described as their own READMEs describe them.

### Known gaps

Not done in this release, and why:

- No run on a real cluster or GPU, no trace-replay harness, and no measured
  false-positive rate. Needs hardware or recorded production telemetry.
- No Kubernetes manifests or Helm chart. The Slinky node-name and drain
  interactions are documented as open, not solved.
- No automatic node resume. Resuming nodes safely needs node-state reads and a
  health check this tool does not have.
- No per-job attribution on shared nodes (cgroup or `hpc_job` based).

## [0.1.0]

First public release.

Find and reclaim idle GPU allocations in Slurm, observe-by-default.

- Core tool implemented and covered by tests.
- `make demo` (or equivalent) runs against a synthetic backend, no special
  hardware required.
- CI runs the test suite on pushes to `main` and on pull requests. (This line
  originally said "on every push", which was not what the workflow did.)

This is a `0.x` release: the behaviour is tested and the safety properties are
asserted, but flags and metric names may still change before `1.0.0`. The
audit behind [Unreleased] found several of those asserted properties did not
hold.

[Unreleased]: https://github.com/Zhanyl-tech/gpu-reaper/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Zhanyl-tech/gpu-reaper/releases/tag/v0.1.0
