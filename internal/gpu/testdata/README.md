# Test fixtures

None of these files was recorded from real hardware. No GPU was available
when they were written.

`dcgm_*.prom` are synthetic dcgm-exporter pages. Their line shape copies the
renderer test in NVIDIA/dcgm-exporter
(`internal/pkg/rendermetrics/render_metrics_test.go`, main at fafd151): labels
`gpu`, `UUID`, `pci_bus_id`, `device`, `modelName`, `hostname`, then collector
attributes such as `hpc_job` or `pod`. The metric names and units come from
`etc/default-counters.csv` in the same commit. Values are invented to exercise
one behaviour per file.

`smi_*.csv` are synthetic `nvidia-smi --format=csv,noheader,nounits` outputs
for the query in `sources.go`. The `[N/A]`, `[Not Supported]` and
`[Unknown Error]` spellings follow the NVIDIA nvidia-smi manual's statement
that unsupported data is shown as N/A. The bracketed error forms are what the
audit's scratch test used; they were not captured from a device.

`smi_mig.csv` row 2 (MIG enabled, utilization readable) is deliberately not
what the manual leads one to expect, since it says utilization is not
supported on MIG-enabled GPUs. It exists so the MIG guard is tested on its
own: row 0 is already unjudgeable through its `[N/A]` utilization.
