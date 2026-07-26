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
	JobsEvaluated  prometheus.Gauge
	Findings       *prometheus.GaugeVec
	ActionsTotal   *prometheus.CounterVec
	ActionErrors   *prometheus.CounterVec
	WastedGPUHours *prometheus.GaugeVec
	GPUsHeld       *prometheus.GaugeVec
	SkippedShared  prometheus.Counter
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
			Help: "Collection failures, by source.",
		}, []string{"source"}),
		JobsEvaluated: f.NewGauge(prometheus.GaugeOpts{
			Name: "gpu_reaper_jobs_evaluated",
			Help: "GPU jobs considered in the last cycle.",
		}),
		Findings: f.NewGaugeVec(prometheus.GaugeOpts{
			Name: "gpu_reaper_findings",
			Help: "Jobs in each verdict/signature state in the last cycle.",
		}, []string{"verdict", "signature", "partition"}),
		ActionsTotal: f.NewCounterVec(prometheus.CounterOpts{
			Name: "gpu_reaper_actions_total",
			Help: "Actions taken, by actor and verdict.",
		}, []string{"actor", "verdict", "enforced"}),
		ActionErrors: f.NewCounterVec(prometheus.CounterOpts{
			Name: "gpu_reaper_action_errors_total",
			Help: "Action failures, by actor.",
		}, []string{"actor"}),
		WastedGPUHours: f.NewGaugeVec(prometheus.GaugeOpts{
			Name: "gpu_reaper_wasted_gpu_hours",
			Help: "GPU-hours currently in a breaching allocation, by partition and signature.",
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
	}
}

// ObserveCycle replaces the per-cycle gauges. Reset first so a verdict that
// stopped occurring reports zero rather than going stale at its last value —
// stale gauges are how dashboards end up showing an incident that ended hours
// ago.
func (m *Metrics) ObserveCycle(findings []policy.Finding) {
	m.Findings.Reset()
	m.WastedGPUHours.Reset()
	m.GPUsHeld.Reset()

	m.JobsEvaluated.Set(float64(len(findings)))

	for _, f := range findings {
		m.Findings.WithLabelValues(
			f.Verdict.String(), string(f.Signature), f.Job.Partition,
		).Inc()

		if f.Verdict >= policy.Alert {
			m.WastedGPUHours.WithLabelValues(f.Job.Partition, string(f.Signature)).
				Add(f.WastedGPUHours)
			m.GPUsHeld.WithLabelValues(f.Job.Partition, string(f.Signature)).
				Add(float64(f.Job.GPUCount))
		}
	}
}
