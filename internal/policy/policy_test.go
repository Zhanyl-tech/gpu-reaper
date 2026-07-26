package policy

import (
	"testing"
	"time"

	"github.com/Zhanyl-tech/gpu-reaper/internal/gpu"
	"github.com/Zhanyl-tech/gpu-reaper/internal/slurm"
)

var base = time.Date(2026, 7, 26, 12, 0, 0, 0, time.UTC)

func job(id string, gpus int, startedAgo time.Duration) slurm.Job {
	return slurm.Job{
		JobID: id, Name: "train", User: "alice", Account: "research",
		Partition: "gpu", QOS: "normal", State: "RUNNING",
		StartTime: base.Add(-startedAgo), Nodes: []string{"gpu001"}, GPUCount: gpus,
	}
}

// samples generates n samples ending at `base`, spaced `every`.
func samples(n int, every time.Duration, util, memFrac, power float64, pids int) []gpu.Sample {
	out := make([]gpu.Sample, 0, n)
	for i := n - 1; i >= 0; i-- {
		var p []int
		for k := 0; k < pids; k++ {
			p = append(p, 1000+k)
		}
		const total = 80 * 1024 * 1024 * 1024
		out = append(out, gpu.Sample{
			Timestamp:     base.Add(-time.Duration(i) * every),
			NodeName:      "gpu001",
			SMUtilPct:     util,
			MemUsedBytes:  uint64(memFrac * total),
			MemTotalBytes: total,
			PowerWatts:    power,
			PIDs:          p,
		})
	}
	return out
}

func engine(t *testing.T, mutate func(*Thresholds)) *Engine {
	t.Helper()
	th := DefaultThresholds()
	if mutate != nil {
		mutate(&th)
	}
	e, err := New(th, DefaultStages(), Exemptions{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return e
}

// ── The safety properties. These are the tests that matter. ────────────────

func TestWarmupIsNeverJudged(t *testing.T) {
	e := engine(t, nil)
	j := job("1", 8, 5*time.Minute) // younger than the 15m warmup
	e.Observe(j.JobID, samples(20, time.Minute, 0, 0, 20, 0), base)

	if got := e.Evaluate(j, base).Verdict; got != Healthy {
		t.Fatalf("a job still starting up must not be judged, got %s", got)
	}
}

func TestCollectorOutageIsNotIdleness(t *testing.T) {
	// The dangerous failure: monitoring goes down, every job looks idle, and a
	// naive reaper cancels the cluster.
	e := engine(t, nil)
	j := job("1", 8, 2*time.Hour)

	old := samples(6, time.Minute, 0, 0, 20, 0)
	for i := range old {
		old[i].Timestamp = old[i].Timestamp.Add(-14 * time.Minute)
	}
	recent := samples(6, time.Minute, 0, 0, 20, 0)
	e.Observe(j.JobID, append(old, recent...), base)

	f := e.Evaluate(j, base)
	if f.Verdict != Healthy {
		t.Fatalf("a gap in samples must not escalate, got %s (%s)", f.Verdict, f.Reason)
	}
}

func TestOneBusySampleClearsTheFinding(t *testing.T) {
	e := engine(t, nil)
	j := job("1", 8, 2*time.Hour)

	s := samples(20, time.Minute, 2, 0.9, 90, 1)
	s[7].SMUtilPct = 96 // one burst of real work
	e.Observe(j.JobID, s, base)

	f := e.Evaluate(j, base)
	if f.Verdict != Healthy {
		t.Fatalf("peak utilization above threshold means alive, got %s", f.Verdict)
	}
}

func TestThinDataIsNotEnough(t *testing.T) {
	e := engine(t, func(th *Thresholds) { th.MinSamples = 8 })
	j := job("1", 8, 2*time.Hour)
	e.Observe(j.JobID, samples(3, time.Minute, 0, 0, 20, 0), base)

	if got := e.Evaluate(j, base).Verdict; got != Healthy {
		t.Fatalf("3 samples must not produce a verdict, got %s", got)
	}
}

func TestCancelIsNeverReachedWithoutPriorEscalation(t *testing.T) {
	// The property that matters most: no configuration, clock jump, or stage
	// widening may take a job from healthy straight to terminated. A human must
	// have had an alert, and the nodes must have been drained, first.
	e, err := New(DefaultThresholds(),
		[]Stage{{After: 0, Verdict: Cancel}}, // maximally aggressive config
		Exemptions{})
	if err != nil {
		t.Fatal(err)
	}
	j := job("1", 8, 6*time.Hour)

	seen := []Verdict{}
	for cycle := 0; cycle < 4; cycle++ {
		now := base.Add(time.Duration(cycle) * 30 * time.Minute)
		s := samples(25, time.Minute, 0, 0, 20, 0)
		for i := range s {
			s[i].Timestamp = s[i].Timestamp.Add(time.Duration(cycle) * 30 * time.Minute)
		}
		e.Observe(j.JobID, s, now)
		seen = append(seen, e.Evaluate(j, now).Verdict)
	}

	if seen[0] != Alert {
		t.Fatalf("cycle 0 must be Alert even when the stage says Cancel, got %s", seen[0])
	}
	if seen[1] != Drain {
		t.Fatalf("cycle 1 should reach Drain, got %s", seen[1])
	}
	if seen[2] != Cancel {
		t.Fatalf("cycle 2 should finally reach Cancel, got %s", seen[2])
	}
	for i, v := range seen {
		if v == Cancel && i < 2 {
			t.Fatalf("Cancel appeared at cycle %d, before alert+drain", i)
		}
	}
}

func TestStarvedJobsAreNeverEscalatedPastAlert(t *testing.T) {
	// A slow dataloader is a tuning problem. Killing a run over it is how the
	// tool gets uninstalled.
	e := engine(t, nil)
	stages := []Stage{{After: 0, Verdict: Alert}, {After: time.Minute, Verdict: Cancel}}
	e, err := New(DefaultThresholds(), stages, Exemptions{})
	if err != nil {
		t.Fatal(err)
	}

	j := job("1", 8, 6*time.Hour)
	e.Observe(j.JobID, samples(30, time.Minute, 6, 0.85, 120, 2), base)

	f := e.Evaluate(j, base)
	if f.Signature != SigStarved {
		t.Fatalf("want starved signature, got %s", f.Signature)
	}
	if f.Verdict > Alert {
		t.Fatalf("starved must cap at Alert, got %s", f.Verdict)
	}
}

func TestUnknownSignatureCapsAtAlert(t *testing.T) {
	e, err := New(DefaultThresholds(),
		[]Stage{{After: 0, Verdict: Alert}, {After: time.Minute, Verdict: Cancel}},
		Exemptions{})
	if err != nil {
		t.Fatal(err)
	}
	j := job("1", 8, 6*time.Hour)
	// No memory held, but processes present and high power — doesn't match a
	// signature we model.
	e.Observe(j.JobID, samples(30, time.Minute, 0, 0.01, 200, 3), base)

	f := e.Evaluate(j, base)
	if f.Signature != SigUnknown {
		t.Fatalf("want unknown, got %s", f.Signature)
	}
	if f.Verdict > Alert {
		t.Fatalf("unknown must cap at Alert, got %s", f.Verdict)
	}
}

// ── Classification ─────────────────────────────────────────────────────────

func TestClassifyIdle(t *testing.T) {
	e := engine(t, nil)
	j := job("1", 4, 3*time.Hour)
	e.Observe(j.JobID, samples(20, time.Minute, 0, 0, 22, 0), base)

	if got := e.Evaluate(j, base).Signature; got != SigIdle {
		t.Fatalf("empty allocation should be idle, got %s", got)
	}
}

func TestClassifyHung(t *testing.T) {
	e := engine(t, nil)
	j := job("1", 4, 3*time.Hour)
	// Memory held, processes resident, zero compute — a stuck collective.
	e.Observe(j.JobID, samples(20, time.Minute, 0, 0.72, 95, 4), base)

	if got := e.Evaluate(j, base).Signature; got != SigHung {
		t.Fatalf("held memory with no compute should be hung, got %s", got)
	}
}

// ── Exemptions ─────────────────────────────────────────────────────────────

func TestExemptions(t *testing.T) {
	cases := []struct {
		name string
		ex   Exemptions
	}{
		{"user", Exemptions{Users: []string{"alice"}}},
		{"account", Exemptions{Accounts: []string{"research"}}},
		{"partition", Exemptions{Partitions: []string{"gpu"}}},
		{"qos", Exemptions{QOS: []string{"normal"}}},
		{"name pattern", Exemptions{NamePattern: "^tra"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e, err := New(DefaultThresholds(), DefaultStages(), tc.ex)
			if err != nil {
				t.Fatal(err)
			}
			j := job("1", 8, 5*time.Hour)
			e.Observe(j.JobID, samples(30, time.Minute, 0, 0, 20, 0), base)

			if got := e.Evaluate(j, base).Verdict; got != Healthy {
				t.Fatalf("exempt job escalated to %s", got)
			}
		})
	}
}

func TestExemptionMatchIsCaseInsensitive(t *testing.T) {
	e, err := New(DefaultThresholds(), DefaultStages(), Exemptions{Users: []string{"ALICE"}})
	if err != nil {
		t.Fatal(err)
	}
	j := job("1", 8, 5*time.Hour)
	e.Observe(j.JobID, samples(30, time.Minute, 0, 0, 20, 0), base)
	if got := e.Evaluate(j, base).Verdict; got != Healthy {
		t.Fatalf("case-insensitive user exemption failed, got %s", got)
	}
}

func TestBadNamePatternIsRejected(t *testing.T) {
	if _, err := New(DefaultThresholds(), DefaultStages(), Exemptions{NamePattern: "([unclosed"}); err == nil {
		t.Fatal("invalid regex should fail at construction, not at match time")
	}
}

// ── Bookkeeping ────────────────────────────────────────────────────────────

func TestCPUOnlyJobsAreIgnored(t *testing.T) {
	e := engine(t, nil)
	j := job("1", 0, 5*time.Hour)
	e.Observe(j.JobID, samples(30, time.Minute, 0, 0, 0, 0), base)

	if got := e.Evaluate(j, base).Verdict; got != Healthy {
		t.Fatalf("CPU-only job must be ignored, got %s", got)
	}
}

func TestWastedGPUHoursAccumulateAcrossCycles(t *testing.T) {
	// The breach clock persists as the sample window rolls forward, so waste is
	// measured from when the breach began rather than from the oldest retained
	// sample.
	e := engine(t, nil)
	j := job("1", 8, 6*time.Hour)

	var last Finding
	for cycle := 0; cycle < 5; cycle++ {
		now := base.Add(time.Duration(cycle) * 30 * time.Minute)
		s := samples(25, time.Minute, 0, 0, 20, 0)
		for i := range s {
			s[i].Timestamp = s[i].Timestamp.Add(time.Duration(cycle) * 30 * time.Minute)
		}
		e.Observe(j.JobID, s, now)
		last = e.Evaluate(j, now)
	}

	// ~2h of dwell across 8 GPUs.
	if last.WastedGPUHours < 8*1.5 {
		t.Fatalf("expected at least 12 GPU-hours after 2h breach on 8 GPUs, got %.1f", last.WastedGPUHours)
	}
}

func TestForgetReleasesCompletedJobs(t *testing.T) {
	e := engine(t, nil)
	e.Observe("1", samples(5, time.Minute, 0, 0, 20, 0), base)
	e.Observe("2", samples(5, time.Minute, 0, 0, 20, 0), base)

	e.Forget(map[string]bool{"1": true})

	if _, ok := e.states["2"]; ok {
		t.Fatal("state for a finished job should be released")
	}
	if _, ok := e.states["1"]; !ok {
		t.Fatal("state for a running job should be kept")
	}
}

func TestObserveDropsSamplesOutsideWindow(t *testing.T) {
	e := engine(t, func(th *Thresholds) { th.Window = 10 * time.Minute })
	s := samples(40, time.Minute, 0, 0, 20, 0) // spans 39 minutes
	e.Observe("1", s, base)

	if n := len(e.states["1"].samples); n > 11 {
		t.Fatalf("expected samples pruned to the window, kept %d", n)
	}
}

func TestRecoveryResetsBreachClock(t *testing.T) {
	e := engine(t, nil)
	j := job("1", 8, 4*time.Hour)

	e.Observe(j.JobID, samples(25, time.Minute, 0, 0, 20, 0), base)
	if f := e.Evaluate(j, base); f.Verdict != Alert {
		t.Fatalf("setup: expected Alert, got %s", f.Verdict)
	}

	later := base.Add(30 * time.Minute)
	busy := samples(25, time.Minute, 88, 0.8, 300, 2)
	for i := range busy {
		busy[i].Timestamp = busy[i].Timestamp.Add(30 * time.Minute)
	}
	e.Observe(j.JobID, busy, later)

	f := e.Evaluate(j, later)
	if f.Verdict != Healthy {
		t.Fatalf("a recovered job must return to Healthy, got %s", f.Verdict)
	}
	if !f.BreachedSince.IsZero() {
		t.Fatal("breach clock should reset on recovery")
	}
}

func TestConstructorRejectsUnusableConfig(t *testing.T) {
	for _, tc := range []struct {
		name string
		th   Thresholds
	}{
		{"zero window", Thresholds{Window: 0, MinSamples: 5}},
		{"zero min samples", Thresholds{Window: time.Minute, MinSamples: 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := New(tc.th, DefaultStages(), Exemptions{}); err == nil {
				t.Fatal("expected construction to fail")
			}
		})
	}
}
