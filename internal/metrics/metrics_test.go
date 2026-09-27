package metrics

import (
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/Zhanyl-tech/gpu-reaper/internal/policy"
	"github.com/Zhanyl-tech/gpu-reaper/internal/slurm"
)

func f(v policy.Verdict, gpus int, hours, secs float64) policy.Finding {
	return policy.Finding{
		Job:     slurm.Job{JobID: "1", Partition: "gpu", GPUCount: gpus},
		Verdict: v, Signature: policy.SigIdle, WastedGPUHours: hours, WastedGPUSeconds: secs,
	}
}

func TestGaugesResetEachCycleCounterAccumulates(t *testing.T) {
	reg := prometheus.NewRegistry()
	m := New(reg)

	m.ObserveCycle([]policy.Finding{f(policy.Alert, 8, 2.5, 9000)}, 1)
	if got := testutil.ToFloat64(m.WastedGPUHours.WithLabelValues("gpu", "idle")); got != 2.5 {
		t.Fatalf("gauge = %v", got)
	}
	m.ObserveCycle([]policy.Finding{f(policy.Alert, 8, 2.6, 960)}, 0)
	if got := testutil.ToFloat64(m.WastedGPUSecs.WithLabelValues("gpu", "idle")); got != 9960 {
		t.Fatalf("counter should accumulate increments, got %v", got)
	}

	// Job ended: the gauge must drop to nothing, the counter must not.
	m.ObserveCycle(nil, 0)
	if n := testutil.CollectAndCount(m.WastedGPUHours); n != 0 {
		t.Fatalf("stale gauge series survived: %d", n)
	}
	if got := testutil.ToFloat64(m.WastedGPUSecs.WithLabelValues("gpu", "idle")); got != 9960 {
		t.Fatalf("counter must survive the job ending, got %v", got)
	}
	if got := testutil.ToFloat64(m.JobsEvaluated); got != 0 {
		t.Fatalf("jobs evaluated = %v", got)
	}
}

func TestNoJobIDLabelAnywhere(t *testing.T) {
	reg := prometheus.NewRegistry()
	m := New(reg)
	m.ObserveCycle([]policy.Finding{f(policy.Drain, 8, 1, 1)}, 0)
	m.ActionsTotal.WithLabelValues("log", "drain", "acted").Inc()
	m.ScrapeErrors.WithLabelValues("nvidia-smi").Inc()
	m.SourceRejected.WithLabelValues("squeue").Inc()
	m.ActionErrors.WithLabelValues("slack").Inc()
	families, err := reg.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, fam := range families {
		if !strings.HasPrefix(fam.GetName(), "gpu_reaper_") {
			t.Errorf("metric %s lacks the gpu_reaper_ prefix", fam.GetName())
		}
		for _, mm := range fam.GetMetric() {
			for _, l := range mm.GetLabel() {
				if strings.Contains(strings.ToLower(l.GetName()), "job") {
					t.Errorf("%s has a job label %q; per-job detail belongs in logs", fam.GetName(), l.GetName())
				}
			}
		}
	}
}
