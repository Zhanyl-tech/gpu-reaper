// Package metrics exposes the reaper's own telemetry.
//
// Cardinality note: job_id is deliberately absent from every metric. A busy
// cluster cycles through millions of job IDs, and a per-job label set would
// take Prometheus down long before it told anyone anything useful. Per-job
// detail belongs in the structured log; metrics carry aggregates only.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/Zhanyl-tech/gpu-reaper/internal/policy"
)

type Metrics struct {
	ScrapeDuration prometheus.Histogram
	ScrapeErrors   *prometheus.CounterVec
	SourceRejected *prometheus.CounterVec
	JobsEvaluated  prometheus.Gauge
	JobsNoSamples  prometheus.Gauge
	Findings       *prometheus.GaugeVec
	ActionsTotal   *prometheus.CounterVec
	ActionErrors   *prometheus.CounterVec
	WastedGPUHours *prometheus.GaugeVec
	WastedGPUSecs  *prometheus.CounterVec
	GPUsHeld       *prometheus.GaugeVec
	SkippedShared  prometheus.Counter
	SkippedPartial prometheus.Counter
	SkippedMapping prometheus.Counter
	LastSuccess    prometheus.Gauge
}

func New(reg prometheus.Registerer) *Metrics {
	f := promauto.With(reg)
	return &Metrics{
		ScrapeDuration: f.NewHistogram(prometheus.HistogramOpts{
			Name:    "gpu_reaper_scrape_duration_seconds",
			Help:    "Time to complete one full evaluation cycle.",
			Buckets: prometheus.DefBuckets,
		}),
		ScrapeErrors: f.NewCounterVec(prometheus.CounterOpts{
			Name: "gpu_reaper_scrape_errors_total",
			Help: "Collection failures, by source. A central GPU source counts one per node it could not scrape.",
		}, []string{"source"}),
		SourceRejected: f.NewCounterVec(prometheus.CounterOpts{
			Name: "gpu_reaper_source_rejected_records_total",
			Help: "Job records a Slurm source refused to parse (malformed line, bad start time, bad hostlist, duplicate job ID). Rejected jobs are never judged.",
		}, []string{"source"}),
		JobsEvaluated: f.NewGauge(prometheus.GaugeOpts{
			Name: "gpu_reaper_jobs_evaluated",
			Help: "GPU jobs with telemetry for every node that the policy engine evaluated in the last cycle.",
		}),
		JobsNoSamples: f.NewGauge(prometheus.GaugeOpts{
			Name: "gpu_reaper_gpu_jobs_without_samples",
			Help: "GPU jobs in scope in the last cycle with no telemetry for any of their nodes. In central mode, persistently non-zero usually means a node-name mismatch or an exporter outage. In per-node mode a node-name mismatch does not show here (the jobs are out of scope before this is counted); see the README's Node names limitation.",
		}),
		Findings: f.NewGaugeVec(prometheus.GaugeOpts{
			Name: "gpu_reaper_findings",
			Help: "Jobs in each verdict/signature state in the last cycle.",
		}, []string{"verdict", "signature", "partition"}),
		ActionsTotal: f.NewCounterVec(prometheus.CounterOpts{
			Name: "gpu_reaper_actions_total",
			Help: "Actions by actor and verdict. outcome is acted or dry_run (observe mode); no-ops are not counted.",
		}, []string{"actor", "verdict", "outcome"}),
		ActionErrors: f.NewCounterVec(prometheus.CounterOpts{
			Name: "gpu_reaper_action_errors_total",
			Help: "Action failures, by actor.",
		}, []string{"actor"}),
		WastedGPUHours: f.NewGaugeVec(prometheus.GaugeOpts{
			Name: "gpu_reaper_wasted_gpu_hours",
			Help: "Currently accruing: GPU-hours of the ongoing breach for jobs breaching now. Drops to zero when a job ends or recovers; use gpu_reaper_wasted_gpu_seconds_total for totals.",
		}, []string{"partition", "signature"}),
		WastedGPUSecs: f.NewCounterVec(prometheus.CounterOpts{
			Name: "gpu_reaper_wasted_gpu_seconds_total",
			Help: "GPU-seconds of observed, confirmed breach (attributable GPUs x observed breach time, warmup excluded). Use increase() for totals over a range.",
		}, []string{"partition", "signature"}),
		GPUsHeld: f.NewGaugeVec(prometheus.GaugeOpts{
			Name: "gpu_reaper_gpus_held_breaching",
			Help: "GPUs held by breaching allocations, by partition and signature.",
		}, []string{"partition", "signature"}),
		SkippedShared: f.NewCounter(prometheus.CounterOpts{
			Name: "gpu_reaper_skipped_shared_node_total",
			Help: "Jobs skipped because their nodes host multiple GPU jobs and " +
				"per-GPU attribution would be ambiguous. A high rate means this " +
				"cluster needs cgroup-based attribution to get coverage.",
		}),
		SkippedPartial: f.NewCounter(prometheus.CounterOpts{
			Name: "gpu_reaper_skipped_partial_evidence_total",
			Help: "Jobs skipped because telemetry covered some but not all of their nodes. A per-node daemon skips every multi-node job this way; judging those needs a central GPU source.",
		}),
		SkippedMapping: f.NewCounter(prometheus.CounterOpts{
			Name: "gpu_reaper_skipped_attribution_conflict_total",
			Help: "Jobs skipped because the telemetry backend attributed one of their GPUs to a different job (dcgm-exporter hpc_job label).",
		}),
		LastSuccess: f.NewGauge(prometheus.GaugeOpts{
			Name: "gpu_reaper_last_successful_cycle_timestamp_seconds",
			Help: "Unix time of the last cycle that listed jobs and collected telemetry. Alert on time() minus this.",
		}),
	}
}

// ObserveCycle replaces the per-cycle gauges and adds this cycle's waste to
// the counter. Gauges are reset first so a verdict that stopped occurring
// reports zero rather than going stale at its last value — stale gauges are
// how dashboards end up showing an incident that ended hours ago.
func (m *Metrics) ObserveCycle(findings []policy.Finding, withoutSamples int) {
	m.Findings.Reset()
	m.WastedGPUHours.Reset()
	m.GPUsHeld.Reset()

	m.JobsEvaluated.Set(float64(len(findings)))
	m.JobsNoSamples.Set(float64(withoutSamples))

	for _, f := range findings {
		m.Findings.WithLabelValues(
			f.Verdict.String(), string(f.Signature), f.Job.Partition,
		).Inc()

		if f.WastedGPUSeconds > 0 {
			m.WastedGPUSecs.WithLabelValues(f.Job.Partition, string(f.Signature)).Add(f.WastedGPUSeconds)
		}
		if f.Verdict >= policy.Alert {
			m.WastedGPUHours.WithLabelValues(f.Job.Partition, string(f.Signature)).
				Add(f.WastedGPUHours)
			m.GPUsHeld.WithLabelValues(f.Job.Partition, string(f.Signature)).
				Add(float64(f.Job.GPUCount))
		}
	}
}
