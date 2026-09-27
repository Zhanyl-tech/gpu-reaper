package policy

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"
	"time"

	"github.com/Zhanyl-tech/gpu-reaper/internal/gpu"
	"github.com/Zhanyl-tech/gpu-reaper/internal/slurm"
)

// Fixtures model what the daemon actually sees: one snapshot per cycle, every
// GPU on the node stamped with the same instant, cycles two minutes apart. The
// previous fixtures fed one GPU at one-minute spacing, which hid every
// real-world failure shape the audit found.

var base = time.Date(2026, 7, 26, 12, 0, 0, 0, time.UTC)

const interval = 2 * time.Minute

func job(id string, gpus int, startedAgo time.Duration) slurm.Job {
	return slurm.Job{
		JobID: id, Name: "train", User: "alice", Account: "research",
		Partition: "gpu", QOS: "normal", State: "RUNNING",
		StartTime: base.Add(-startedAgo), Nodes: []string{"gpu001"}, GPUCount: gpus,
	}
}

// load describes what every GPU in a snapshot reports.
type load struct {
	util, memFrac, power float64
	pids                 int
}

var (
	idleLoad    = load{0, 0, 22, 0}
	hungLoad    = load{0, 0.72, 90, 2}
	starvedLoad = load{6, 0.85, 120, 2}
	busyLoad    = load{88, 0.8, 300, 2}
)

// snapshot returns one fully-readable sample per GPU on node, all at `at`.
func snapshot(at time.Time, node string, gpus int, l load) []gpu.Sample {
	const total = 80 * 1024 * 1024 * 1024
	out := make([]gpu.Sample, 0, gpus)
	for i := 0; i < gpus; i++ {
		var p []int
		for k := 0; k < l.pids; k++ {
			p = append(p, 1000+k)
		}
		out = append(out, gpu.Sample{
			Timestamp: at, NodeName: node, GPUIndex: i,
			SMUtilPct: l.util, MemUsedBytes: uint64(l.memFrac * total), MemTotalBytes: total,
			PowerWatts: l.power, PIDs: p,
			UtilKnown: true, MemKnown: true, PowerKnown: true, PIDsKnown: true,
		})
	}
	return out
}

func newEngine(t *testing.T, th Thresholds, stages []Stage) *Engine {
	t.Helper()
	e, err := New(th, stages, Exemptions{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return e
}

func engine(t *testing.T, mutate func(*Thresholds)) *Engine {
	t.Helper()
	th := DefaultThresholds()
	if mutate != nil {
		mutate(&th)
	}
	return newEngine(t, th, DefaultStages())
}

// driver runs the daemon's loop for one job: observe, evaluate, and confirm a
// drain the way main does when the cluster actor succeeds.
type driver struct {
	e          *Engine
	j          slurm.Job
	drainFails bool
	log        []Finding
}

func (d *driver) cycle(now time.Time, samples []gpu.Sample) Finding {
	d.e.Observe(d.j, samples, now)
	f := d.e.Evaluate(d.j, now)
	if f.Verdict == Drain && !f.DrainConfirmed && !d.drainFails {
		d.e.ConfirmDrain(d.j, now)
	}
	d.log = append(d.log, f)
	return f
}

// run drives cycles every interval from `from` while now < until, with every
// GPU reporting l. It returns the last finding.
func (d *driver) run(from, until time.Time, l load) (Finding, time.Time) {
	var f Finding
	now := from
	for ; now.Before(until); now = now.Add(interval) {
		f = d.cycle(now, snapshot(now, "gpu001", d.j.GPUCount, l))
	}
	return f, now
}

func firstAt(log []Finding, v Verdict) int {
	for i, f := range log {
		if f.Verdict == v {
			return i
		}
	}
	return -1
}

// ── The safety properties. These are the tests that matter. ────────────────

func TestWarmupIsNeverJudged(t *testing.T) {
	e := engine(t, nil)
	j := job("1", 8, 5*time.Minute) // younger than the 15m warmup
	d := &driver{e: e, j: j}
	f, _ := d.run(base.Add(-4*time.Minute), base.Add(time.Minute), idleLoad)
	if f.Verdict != Healthy {
		t.Fatalf("a job still starting up must not be judged, got %s", f.Verdict)
	}
}

// Samples taken during warmup are not evidence. Previously they counted, so a
// job was judged the moment warmup ended, with its staging time as "waste".
func TestClaim_WarmupSamplesAreNotEvidence(t *testing.T) {
	e := engine(t, nil)
	j := job("1", 8, 0) // starts at base
	d := &driver{e: e, j: j}
	d.run(base, base.Add(time.Hour), idleLoad)

	i := firstAt(d.log, Alert)
	if i < 0 {
		t.Fatal("an idle job should eventually alert")
	}
	th := DefaultThresholds()
	earliest := th.Warmup + th.Window - th.MaxSampleGap
	if at := time.Duration(i) * interval; at < earliest {
		t.Fatalf("first alert at %s; must be no earlier than warmup+window-gap (%s)", at, earliest)
	}
	if !d.log[i].BreachedSince.After(j.StartTime.Add(th.Warmup - time.Second)) {
		t.Fatalf("breach clock %s starts inside warmup", d.log[i].BreachedSince)
	}
	if max := 8 * th.Window.Hours(); d.log[i].WastedGPUHours > max {
		t.Fatalf("first alert reports %.2f GPU-hours; at most %.2f were observed", d.log[i].WastedGPUHours, max)
	}
}

// The audit's min_samples finding: one instantaneous 8-GPU snapshot passed
// min_samples=8 and alerted "for 0s across 8 samples".
func TestClaim_OneSnapshotIsNotASustainedBreach(t *testing.T) {
	e := engine(t, nil)
	j := job("1", 8, 2*time.Hour)
	d := &driver{e: e, j: j}
	f := d.cycle(base, snapshot(base, "gpu001", 8, hungLoad))
	if f.Verdict != Healthy {
		t.Fatalf("one snapshot must not confirm a breach, got %s (%s)", f.Verdict, f.Reason)
	}
	if !strings.Contains(f.Reason, "insufficient observations") {
		t.Fatalf("reason should explain, got %q", f.Reason)
	}
}

// The audit reproduced Cancel after 4 minutes of observation with stage
// {0: cancel}. Now the earliest possible Cancel is: the window covered, one
// Alert cycle, a confirmed Drain, then a full Window in Drain.
func TestClaim_CancelNeedsCoverageAlertDrainAndAWindow(t *testing.T) {
	th := DefaultThresholds()
	e := newEngine(t, th, []Stage{{After: 0, Verdict: Cancel}}) // maximally aggressive
	j := job("1", 8, 2*time.Hour)
	d := &driver{e: e, j: j}
	d.run(base, base.Add(2*time.Hour), idleLoad)

	a, dr, c := firstAt(d.log, Alert), firstAt(d.log, Drain), firstAt(d.log, Cancel)
	if a < 0 || dr < 0 || c < 0 {
		t.Fatalf("idle job with a cancel stage should reach every rung: alert=%d drain=%d cancel=%d", a, dr, c)
	}
	if !(a < dr && dr < c) {
		t.Fatalf("rungs out of order: alert=%d drain=%d cancel=%d", a, dr, c)
	}
	if got, min := time.Duration(a)*interval, th.Window-th.MaxSampleGap; got < min {
		t.Fatalf("alert after %s of observation, want >= %s", got, min)
	}
	if got := time.Duration(c-dr) * interval; got < th.Window {
		t.Fatalf("cancel only %s after the drain, want >= %s", got, th.Window)
	}
	for i := 0; i < c; i++ {
		if d.log[i].Verdict == Cancel {
			t.Fatalf("Cancel appeared at cycle %d", i)
		}
	}
}

func TestCancelIsNeverReachedWithoutPriorEscalation(t *testing.T) {
	// The property that matters most: no configuration, clock jump, or stage
	// widening may take a job from healthy straight to terminated. An alert
	// cycle, a confirmed drain, and a window in drain must come first.
	e := newEngine(t, DefaultThresholds(), []Stage{{After: 0, Verdict: Cancel}})
	j := job("1", 8, 6*time.Hour)
	d := &driver{e: e, j: j}
	d.run(base, base.Add(3*time.Hour), idleLoad)

	sawAlert, sawDrain := false, false
	for i, f := range d.log {
		switch f.Verdict {
		case Alert:
			sawAlert = true
		case Drain:
			if !sawAlert {
				t.Fatalf("Drain at cycle %d before any Alert", i)
			}
			sawDrain = true
		case Cancel:
			if !sawDrain {
				t.Fatalf("Cancel at cycle %d before any Drain", i)
			}
		}
	}
}

// If the drain command fails, Cancel must never follow. Previously the gate
// trusted the computed verdict, not the executed action.
func TestFailedDrainBlocksCancel(t *testing.T) {
	e := newEngine(t, DefaultThresholds(), []Stage{{After: 0, Verdict: Cancel}})
	d := &driver{e: e, j: job("1", 8, 6*time.Hour), drainFails: true}
	d.run(base, base.Add(4*time.Hour), idleLoad)
	if i := firstAt(d.log, Cancel); i >= 0 {
		t.Fatalf("Cancel at cycle %d although no drain was ever confirmed", i)
	}
	if last := d.log[len(d.log)-1]; last.Verdict != Drain || last.DrainConfirmed {
		t.Fatalf("should keep asking for a drain, got %s confirmed=%t", last.Verdict, last.DrainConfirmed)
	}
}

// The audit's critical outage scenario: 3h50m of observed breach, then 40
// minutes with no telemetry (the cycle aborts before Observe), then one
// snapshot. That used to yield Cancel "for 4h30m0s".
func TestClaim_OutageLongerThanWindowNeverEscalates(t *testing.T) {
	stages := []Stage{{After: 0, Verdict: Alert}, {After: 60 * time.Minute, Verdict: Drain}, {After: 4 * time.Hour, Verdict: Cancel}}
	e := newEngine(t, DefaultThresholds(), stages)
	if err := e.AllowCancel(SigIdle, SigHung); err != nil { // exactly the audit's setup
		t.Fatal(err)
	}
	j := job("1", 8, 10*time.Hour)
	d := &driver{e: e, j: j}

	before, now := d.run(base, base.Add(230*time.Minute), hungLoad)
	if before.Verdict != Drain {
		t.Fatalf("setup: expected Drain before the outage, got %s", before.Verdict)
	}

	now = now.Add(40 * time.Minute) // outage: no Observe, no Evaluate
	f := d.cycle(now, snapshot(now, "gpu001", 8, hungLoad))
	if f.Verdict != Healthy {
		t.Fatalf("first cycle after a 40m outage must not judge, got %s (%s)", f.Verdict, f.Reason)
	}

	// Keep observing the same breach. The ladder must restart from the bottom:
	// no Cancel until a new Window is covered, alerted, drained, and a further
	// Window has passed.
	d.log = nil
	d.run(now.Add(interval), now.Add(5*time.Hour), hungLoad)
	a, c := firstAt(d.log, Alert), firstAt(d.log, Cancel)
	if a < 0 {
		t.Fatal("breach should be re-detected once the window is covered again")
	}
	if dr := firstAt(d.log, Drain); dr >= 0 && dr < a {
		t.Fatalf("drain at %d before re-alert at %d", dr, a)
	}
	if c >= 0 && time.Duration(c)*interval < time.Hour {
		t.Fatalf("cancel only %s after the outage ended", time.Duration(c)*interval)
	}
	if !d.log[a].BreachedSince.After(now.Add(-time.Second)) {
		t.Fatalf("breach clock %s must restart after the outage (ended %s)", d.log[a].BreachedSince, now)
	}
}

func TestCollectorOutageInsideTheWindowIsNotIdleness(t *testing.T) {
	e := engine(t, nil)
	j := job("1", 8, 2*time.Hour)
	d := &driver{e: e, j: j}
	_, now := d.run(base, base.Add(12*time.Minute), idleLoad)
	now = now.Add(6 * time.Minute) // 8-minute hole between observations
	f, _ := d.run(now, now.Add(10*time.Minute), idleLoad)
	if f.Verdict != Healthy || !strings.Contains(f.Reason, "gap") {
		t.Fatalf("a gap in samples must not escalate, got %s (%s)", f.Verdict, f.Reason)
	}
}

// With the defaults (interval 2m, max_sample_gap 3m) a single failed cycle
// leaves a 4m hole between observations, which is beyond the gap and resets
// the breach clock and the ladder.
func TestOneFailedCycleResetsWithDefaultTiming(t *testing.T) {
	e := engine(t, nil)
	d := &driver{e: e, j: job("1", 8, 2*time.Hour)}
	before, now := d.run(base, base.Add(30*time.Minute), idleLoad)
	if before.Verdict < Alert {
		t.Fatalf("setup: want a confirmed breach, got %s (%s)", before.Verdict, before.Reason)
	}
	now = now.Add(interval) // one cycle fails: no Observe, no Evaluate
	f := d.cycle(now, snapshot(now, "gpu001", 8, idleLoad))
	if f.Verdict != Healthy || !strings.Contains(f.Reason, "gap") {
		t.Fatalf("a 4m hole with a 3m max gap must reset, got %s (%s)", f.Verdict, f.Reason)
	}
}

// The documented limit of the outage guard: only a hole larger than
// max_sample_gap resets. Where max_sample_gap is at least twice the interval,
// one failed cycle stays inside it and the breach carries on, by design: the
// observations either side are still contiguous evidence. Two failed cycles
// exceed it and reset.
func TestHoleWithinMaxSampleGapIsToleratedByDesign(t *testing.T) {
	e := engine(t, func(th *Thresholds) { th.MaxSampleGap = 5 * time.Minute })
	d := &driver{e: e, j: job("1", 8, 2*time.Hour)}
	before, now := d.run(base, base.Add(30*time.Minute), idleLoad)
	if before.Verdict < Alert {
		t.Fatalf("setup: want a confirmed breach, got %s (%s)", before.Verdict, before.Reason)
	}

	now = now.Add(interval) // one failed cycle: a 4m hole, inside the 5m gap
	f := d.cycle(now, snapshot(now, "gpu001", 8, idleLoad))
	if f.Verdict < Alert || !f.BreachedSince.Equal(before.BreachedSince) {
		t.Fatalf("a hole within max_sample_gap should not reset: %s since %s (was since %s): %s",
			f.Verdict, f.BreachedSince, before.BreachedSince, f.Reason)
	}

	now = now.Add(3 * interval) // two failed cycles: a 6m hole
	f = d.cycle(now, snapshot(now, "gpu001", 8, idleLoad))
	if f.Verdict != Healthy || !strings.Contains(f.Reason, "gap") {
		t.Fatalf("a hole beyond max_sample_gap must reset, got %s (%s)", f.Verdict, f.Reason)
	}
}

func TestStaleNewestSampleIsNotJudged(t *testing.T) {
	e := engine(t, nil)
	j := job("1", 8, 2*time.Hour)
	d := &driver{e: e, j: j}
	_, now := d.run(base, base.Add(30*time.Minute), idleLoad)
	// The node stops reporting but the job is still evaluated.
	f := d.cycle(now.Add(4*time.Minute), nil)
	if f.Verdict != Healthy || !strings.Contains(f.Reason, "old") {
		t.Fatalf("stale evidence must not be judged, got %s (%s)", f.Verdict, f.Reason)
	}
}

// A requeued job keeps its JobID but gets a new StartTime. It must not
// inherit the old incarnation's breach clock or drain.
func TestClaim_RequeuedJobStartsOver(t *testing.T) {
	stages := []Stage{{After: 0, Verdict: Alert}, {After: 60 * time.Minute, Verdict: Drain}, {After: 90 * time.Minute, Verdict: Cancel}}
	e := newEngine(t, DefaultThresholds(), stages)
	if err := e.AllowCancel(SigIdle, SigHung); err != nil { // the audit's setup
		t.Fatal(err)
	}
	orig := job("7", 8, 5*time.Hour)
	d := &driver{e: e, j: orig}
	// 80 minutes hung, observed: reaches Drain, not yet Cancel.
	last, now := d.run(base, base.Add(80*time.Minute), hungLoad)
	if last.Verdict != Drain {
		t.Fatalf("setup: expected Drain, got %s", last.Verdict)
	}

	requeued := orig
	requeued.StartTime = now
	d.j, d.log = requeued, nil
	// Its first minutes stage data at 0% utilization, legitimately.
	d.run(now, now.Add(40*time.Minute), hungLoad)
	for i, f := range d.log {
		if f.Verdict >= Drain {
			t.Fatalf("requeued job reached %s at cycle %d on the old incarnation's history", f.Verdict, i)
		}
	}
	// The new incarnation is tracked from scratch: judged again, but only
	// after its own warmup plus a covered window.
	th := DefaultThresholds()
	a := firstAt(d.log, Alert)
	if a < 0 {
		t.Fatal("the requeued job was never judged again")
	}
	if at := time.Duration(a) * interval; at < th.Warmup+th.Window-th.MaxSampleGap {
		t.Fatalf("requeued job alerted %s after restart, before warmup+window", at)
	}
	if !d.log[a].BreachedSince.After(requeued.StartTime.Add(th.Warmup - time.Second)) {
		t.Fatalf("breach clock %s predates the new incarnation's warmup", d.log[a].BreachedSince)
	}
}

func TestConfirmDrainIgnoredForAnotherIncarnation(t *testing.T) {
	e := newEngine(t, DefaultThresholds(), []Stage{{After: 0, Verdict: Cancel}})
	j := job("1", 8, 6*time.Hour)
	d := &driver{e: e, j: j, drainFails: true}
	d.run(base, base.Add(time.Hour), idleLoad)
	other := j
	other.StartTime = j.StartTime.Add(time.Hour)
	e.ConfirmDrain(other, base) // wrong incarnation
	if !e.states["1"].drainConfirmedAt.IsZero() {
		t.Fatal("a confirmation for another incarnation must be ignored")
	}
}

// When the daemon skips a job (shared node, partial evidence) it calls Drop.
// When it returns, nothing from before counts.
func TestClaim_SharedToUnsharedStartsOver(t *testing.T) {
	e := newEngine(t, DefaultThresholds(), []Stage{{After: 0, Verdict: Cancel}})
	j := job("1", 8, 6*time.Hour)
	d := &driver{e: e, j: j}
	last, now := d.run(base, base.Add(50*time.Minute), idleLoad)
	if last.Verdict < Drain {
		t.Fatalf("setup: expected at least Drain, got %s", last.Verdict)
	}
	e.Drop(j.JobID) // one cycle skipped as shared
	d.log = nil
	f := d.cycle(now.Add(interval), snapshot(now.Add(interval), "gpu001", 8, idleLoad))
	if f.Verdict != Healthy {
		t.Fatalf("after a skip the job must start over, got %s", f.Verdict)
	}
}

// Even without an explicit Drop, a skipped period is a hole the coverage
// checks catch.
func TestUnobservedPeriodResetsWithoutDrop(t *testing.T) {
	e := newEngine(t, DefaultThresholds(), []Stage{{After: 0, Verdict: Cancel}})
	j := job("1", 8, 6*time.Hour)
	d := &driver{e: e, j: j}
	_, now := d.run(base, base.Add(50*time.Minute), idleLoad)
	now = now.Add(10 * time.Minute) // not observed for 10 minutes
	d.log = nil
	d.run(now, now.Add(30*time.Minute), idleLoad)
	if c := firstAt(d.log, Cancel); c >= 0 {
		t.Fatalf("cancel at cycle %d right after an unobserved period", c)
	}
	if d.log[0].Verdict != Healthy {
		t.Fatalf("first cycle after the hole should not judge, got %s", d.log[0].Verdict)
	}
}

// The audit's N/A finding at the policy level: unreadable telemetry is never
// idle, and never escalates.
func TestClaim_UnreadableTelemetryIsNeverIdle(t *testing.T) {
	e := newEngine(t, DefaultThresholds(), []Stage{{After: 0, Verdict: Cancel}})
	j := job("1", 8, 6*time.Hour)
	d := &driver{e: e, j: j}
	now := base
	for ; now.Before(base.Add(2 * time.Hour)); now = now.Add(interval) {
		s := snapshot(now, "gpu001", 8, idleLoad)
		for i := range s {
			// What nvidia-smi's "[N/A]" now parses to.
			s[i].SMUtilPct, s[i].UtilKnown = 0, false
			s[i].PowerWatts, s[i].PowerKnown = 0, false
		}
		d.cycle(now, s)
	}
	for i, f := range d.log {
		if f.Verdict != Healthy {
			t.Fatalf("cycle %d: unreadable telemetry produced %s (%s)", i, f.Verdict, f.Reason)
		}
	}
}

func TestOneUnreadableGPUBlocksTheJob(t *testing.T) {
	e := engine(t, nil)
	j := job("1", 8, 6*time.Hour)
	d := &driver{e: e, j: j}
	var f Finding
	for now := base; now.Before(base.Add(time.Hour)); now = now.Add(interval) {
		s := snapshot(now, "gpu001", 8, idleLoad)
		s[3].UtilKnown = false // the one GPU we cannot read might be the busy one
		f = d.cycle(now, s)
	}
	if f.Verdict != Healthy {
		t.Fatalf("got %s (%s)", f.Verdict, f.Reason)
	}
}

func TestMIGGPUsAreNotJudged(t *testing.T) {
	e := engine(t, nil)
	d := &driver{e: e, j: job("1", 8, 6*time.Hour)}
	var f Finding
	for now := base; now.Before(base.Add(time.Hour)); now = now.Add(interval) {
		s := snapshot(now, "gpu001", 8, hungLoad)
		s[0].MIG = true
		f = d.cycle(now, s)
	}
	if f.Verdict != Healthy || !strings.Contains(f.Reason, "MIG") {
		t.Fatalf("got %s (%s)", f.Verdict, f.Reason)
	}
}

// Without positive process and power evidence, nothing is idle or hung.
func TestIdleAndHungNeedPositiveEvidence(t *testing.T) {
	cases := map[string]func(*gpu.Sample){
		"pids unknown":  func(s *gpu.Sample) { s.PIDsKnown, s.PIDs = false, nil },
		"power unknown": func(s *gpu.Sample) { s.PowerKnown, s.PowerWatts = false, 0 },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			e := newEngine(t, DefaultThresholds(), []Stage{{After: 0, Verdict: Cancel}})
			d := &driver{e: e, j: job("1", 8, 6*time.Hour)}
			for now := base; now.Before(base.Add(2 * time.Hour)); now = now.Add(interval) {
				s := snapshot(now, "gpu001", 8, idleLoad)
				for i := range s {
					mutate(&s[i])
				}
				f := d.cycle(now, s)
				if f.Signature == SigIdle {
					t.Fatalf("classified idle with %s", name)
				}
				if f.Verdict > Alert {
					t.Fatalf("escalated to %s with %s", f.Verdict, name)
				}
			}
		})
	}
	// Hung needs processes positively seen.
	e := newEngine(t, DefaultThresholds(), []Stage{{After: 0, Verdict: Cancel}})
	d := &driver{e: e, j: job("2", 8, 6*time.Hour)}
	for now := base; now.Before(base.Add(2 * time.Hour)); now = now.Add(interval) {
		s := snapshot(now, "gpu001", 8, hungLoad)
		for i := range s {
			s[i].PIDsKnown, s[i].PIDs = false, nil
		}
		if f := d.cycle(now, s); f.Signature == SigHung || f.Verdict > Alert {
			t.Fatalf("held memory with unknown processes must not be hung: %s %s", f.Signature, f.Verdict)
		}
	}
}

func TestPartialNodeEvidenceIsNotJudged(t *testing.T) {
	e := engine(t, nil)
	j := job("1", 16, 6*time.Hour)
	j.Nodes = []string{"gpu001", "gpu002"}
	d := &driver{e: e, j: j}
	var f Finding
	for now := base; now.Before(base.Add(time.Hour)); now = now.Add(interval) {
		f = d.cycle(now, snapshot(now, "gpu001", 8, idleLoad)) // gpu002 never seen
	}
	if f.Verdict != Healthy || !strings.Contains(f.Reason, "gpu002") {
		t.Fatalf("got %s (%s)", f.Verdict, f.Reason)
	}
}

func TestMultiNodeJobWithFullEvidenceIsJudged(t *testing.T) {
	e := engine(t, nil)
	j := job("1", 16, 6*time.Hour)
	j.Nodes = []string{"gpu001", "gpu002"}
	d := &driver{e: e, j: j}
	var f Finding
	for now := base; now.Before(base.Add(time.Hour)); now = now.Add(interval) {
		s := append(snapshot(now, "gpu001", 8, idleLoad), snapshot(now.Add(time.Millisecond), "gpu002", 8, idleLoad)...)
		f = d.cycle(now, s)
	}
	if f.Verdict < Alert || f.GPUs != 16 {
		t.Fatalf("got %s over %d GPUs (%s)", f.Verdict, f.GPUs, f.Reason)
	}
}

func TestFewerGPUsThanAllocatedIsNotJudged(t *testing.T) {
	e := engine(t, nil)
	d := &driver{e: e, j: job("1", 8, 6*time.Hour)}
	var f Finding
	for now := base; now.Before(base.Add(time.Hour)); now = now.Add(interval) {
		f = d.cycle(now, snapshot(now, "gpu001", 7, idleLoad))
	}
	if f.Verdict != Healthy || !strings.Contains(f.Reason, "holds 8") {
		t.Fatalf("got %s (%s)", f.Verdict, f.Reason)
	}
}

func TestVaryingGPUCountIsNotJudged(t *testing.T) {
	e := engine(t, nil)
	d := &driver{e: e, j: job("1", 4, 6*time.Hour)}
	var f Finding
	i := 0
	for now := base; now.Before(base.Add(time.Hour)); now = now.Add(interval) {
		n := 8
		if i%5 == 0 {
			n = 7 // a GPU drops out of one snapshot
		}
		f = d.cycle(now, snapshot(now, "gpu001", n, idleLoad))
		i++
	}
	if f.Verdict != Healthy || !strings.Contains(f.Reason, "varying") {
		t.Fatalf("got %s (%s)", f.Verdict, f.Reason)
	}
}

func TestOneBusySampleClearsTheFinding(t *testing.T) {
	e := engine(t, nil)
	d := &driver{e: e, j: job("1", 8, 2*time.Hour)}
	var f Finding
	i := 0
	for now := base; now.Before(base.Add(time.Hour)); now = now.Add(interval) {
		s := snapshot(now, "gpu001", 8, starvedLoad)
		if i%9 == 0 {
			s[5].SMUtilPct = 96 // one burst of real work on one GPU
		}
		f = d.cycle(now, s)
		i++
	}
	if f.Verdict != Healthy {
		t.Fatalf("peak utilization above threshold means alive, got %s (%s)", f.Verdict, f.Reason)
	}
}

// hung is what a long checkpoint write looks like, so it stops at Drain
// unless the operator opts in.
func TestHungIsNotCancelledByDefault(t *testing.T) {
	e := newEngine(t, DefaultThresholds(), []Stage{{After: 0, Verdict: Cancel}})
	d := &driver{e: e, j: job("1", 8, 6*time.Hour)}
	d.run(base, base.Add(6*time.Hour), hungLoad)
	if i := firstAt(d.log, Cancel); i >= 0 {
		t.Fatalf("hung cancelled at cycle %d without opt-in", i)
	}
	if firstAt(d.log, Drain) < 0 {
		t.Fatal("hung should still reach Drain")
	}
}

func TestHungCancelIsOptIn(t *testing.T) {
	e := newEngine(t, DefaultThresholds(), []Stage{{After: 0, Verdict: Cancel}})
	if err := e.AllowCancel(SigIdle, SigHung); err != nil {
		t.Fatal(err)
	}
	d := &driver{e: e, j: job("1", 8, 6*time.Hour)}
	d.run(base, base.Add(2*time.Hour), hungLoad)
	if firstAt(d.log, Cancel) < 0 {
		t.Fatal("with hung allowed, a hung job should eventually be cancel-eligible")
	}
}

func TestAllowCancelRejectsStarvedAndUnknown(t *testing.T) {
	e := engine(t, nil)
	for _, s := range []Signature{SigStarved, SigUnknown, "nccl-stall"} {
		if err := e.AllowCancel(s); err == nil {
			t.Errorf("AllowCancel(%s) should be rejected", s)
		}
	}
}

func TestStarvedJobsAreNeverEscalatedPastAlert(t *testing.T) {
	// A slow dataloader is a tuning problem. Killing a run over it is how the
	// tool gets uninstalled.
	e := newEngine(t, DefaultThresholds(), []Stage{{After: 0, Verdict: Alert}, {After: time.Minute, Verdict: Cancel}})
	d := &driver{e: e, j: job("1", 8, 6*time.Hour)}
	f, _ := d.run(base, base.Add(3*time.Hour), starvedLoad)
	if f.Signature != SigStarved {
		t.Fatalf("want starved signature, got %s", f.Signature)
	}
	for _, g := range d.log {
		if g.Verdict > Alert {
			t.Fatalf("starved must cap at Alert, got %s", g.Verdict)
		}
	}
}

func TestUnknownSignatureCapsAtAlert(t *testing.T) {
	e := newEngine(t, DefaultThresholds(), []Stage{{After: 0, Verdict: Alert}, {After: time.Minute, Verdict: Cancel}})
	d := &driver{e: e, j: job("1", 8, 6*time.Hour)}
	// No memory held, but processes present and high power — doesn't match a
	// signature we model.
	f, _ := d.run(base, base.Add(3*time.Hour), load{0, 0.01, 200, 3})
	if f.Signature != SigUnknown {
		t.Fatalf("want unknown, got %s", f.Signature)
	}
	for _, g := range d.log {
		if g.Verdict > Alert {
			t.Fatalf("unknown must cap at Alert, got %s", g.Verdict)
		}
	}
}

// ── Classification ─────────────────────────────────────────────────────────

func TestClassifyIdle(t *testing.T) {
	d := &driver{e: engine(t, nil), j: job("1", 4, 3*time.Hour)}
	if f, _ := d.run(base, base.Add(time.Hour), idleLoad); f.Signature != SigIdle {
		t.Fatalf("empty allocation should be idle, got %s (%s)", f.Signature, f.Reason)
	}
}

func TestClassifyHung(t *testing.T) {
	d := &driver{e: engine(t, nil), j: job("1", 4, 3*time.Hour)}
	// Memory held, processes resident, zero compute — a stuck collective.
	if f, _ := d.run(base, base.Add(time.Hour), hungLoad); f.Signature != SigHung {
		t.Fatalf("held memory with no compute should be hung, got %s", f.Signature)
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
			d := &driver{e: e, j: job("1", 8, 5*time.Hour)}
			d.run(base, base.Add(2*time.Hour), idleLoad)
			for _, f := range d.log {
				if f.Verdict != Healthy {
					t.Fatalf("exempt job escalated to %s", f.Verdict)
				}
			}
		})
	}
}

func TestExemptionMatchIsCaseInsensitive(t *testing.T) {
	e, err := New(DefaultThresholds(), DefaultStages(), Exemptions{Users: []string{"ALICE"}})
	if err != nil {
		t.Fatal(err)
	}
	d := &driver{e: e, j: job("1", 8, 5*time.Hour)}
	if f, _ := d.run(base, base.Add(time.Hour), idleLoad); f.Verdict != Healthy {
		t.Fatalf("case-insensitive user exemption failed, got %s", f.Verdict)
	}
}

func TestBadNamePatternIsRejected(t *testing.T) {
	if _, err := New(DefaultThresholds(), DefaultStages(), Exemptions{NamePattern: "([unclosed"}); err == nil {
		t.Fatal("invalid regex should fail at construction, not at match time")
	}
}

// ── Bookkeeping ────────────────────────────────────────────────────────────

func TestCPUOnlyJobsAreIgnored(t *testing.T) {
	d := &driver{e: engine(t, nil), j: job("1", 0, 5*time.Hour)}
	if f, _ := d.run(base, base.Add(time.Hour), idleLoad); f.Verdict != Healthy {
		t.Fatalf("CPU-only job must be ignored, got %s", f.Verdict)
	}
}

func TestStartTimeUnknownIsNotJudged(t *testing.T) {
	j := job("1", 8, 0)
	j.StartTime = time.Time{}
	d := &driver{e: engine(t, nil), j: j}
	f, _ := d.run(base, base.Add(time.Hour), idleLoad)
	if f.Verdict != Healthy || !strings.Contains(f.Reason, "start time unknown") {
		t.Fatalf("got %s (%s)", f.Verdict, f.Reason)
	}
}

// The counter increments must add up to GPUs x observed breach, with nothing
// counted twice and nothing from warmup.
func TestWastedGPUSecondsSumToObservedBreach(t *testing.T) {
	d := &driver{e: engine(t, nil), j: job("1", 8, 6*time.Hour)}
	last, _ := d.run(base, base.Add(2*time.Hour), idleLoad)
	var sum float64
	for _, f := range d.log {
		sum += f.WastedGPUSeconds
	}
	want := last.WastedGPUHours * 3600
	if diff := sum - want; diff > 1 || diff < -1 {
		t.Fatalf("sum of increments %.0f GPU-s, gauge says %.0f", sum, want)
	}
	if last.WastedGPUHours < 8*1.5 {
		t.Fatalf("expected at least 12 GPU-hours after ~2h breach on 8 GPUs, got %.1f", last.WastedGPUHours)
	}
}

func TestWasteIsCappedAtAllocatedGPUs(t *testing.T) {
	// The job holds 4 of the node's 8 GPUs (the node is not shared: the other
	// 4 are unallocated). All 8 are idle, but only 4 are this job's waste.
	d := &driver{e: engine(t, nil), j: job("1", 4, 6*time.Hour)}
	var f Finding
	var last time.Time
	for now := base; now.Before(base.Add(time.Hour)); now = now.Add(interval) {
		f = d.cycle(now, snapshot(now, "gpu001", 8, idleLoad))
		last = now
	}
	if f.GPUs != 8 {
		t.Fatalf("evidence should cover 8 GPUs, got %d", f.GPUs)
	}
	want := 4 * last.Sub(f.BreachedSince).Hours()
	if diff := f.WastedGPUHours - want; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("waste %.4f GPU-h, want 4 GPUs x dwell = %.4f", f.WastedGPUHours, want)
	}
}

func TestForgetReleasesCompletedJobs(t *testing.T) {
	e := engine(t, nil)
	e.Observe(job("1", 8, time.Hour), snapshot(base, "gpu001", 8, idleLoad), base)
	e.Observe(job("2", 8, time.Hour), snapshot(base, "gpu001", 8, idleLoad), base)

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
	j := job("1", 1, 5*time.Hour)
	for i := 39; i >= 0; i-- {
		at := base.Add(-time.Duration(i) * time.Minute)
		e.Observe(j, snapshot(at, "gpu001", 1, idleLoad), base)
	}
	if n := len(e.states["1"].samples); n > 11 {
		t.Fatalf("expected samples pruned to the window, kept %d", n)
	}
}

func TestRecoveryResetsBreachClock(t *testing.T) {
	d := &driver{e: engine(t, nil), j: job("1", 8, 4*time.Hour)}
	f, now := d.run(base, base.Add(40*time.Minute), idleLoad)
	if f.Verdict != Alert {
		t.Fatalf("setup: expected Alert, got %s", f.Verdict)
	}
	f = d.cycle(now, snapshot(now, "gpu001", 8, busyLoad))
	if f.Verdict != Healthy {
		t.Fatalf("a recovered job must return to Healthy, got %s", f.Verdict)
	}
	if !d.e.states["1"].breachedSince.IsZero() {
		t.Fatal("breach clock should reset on recovery")
	}
}

func TestRecoveryAfterDrainIsFlagged(t *testing.T) {
	d := &driver{e: newEngine(t, DefaultThresholds(), []Stage{{After: 0, Verdict: Drain}}), j: job("1", 8, 4*time.Hour)}
	f, now := d.run(base, base.Add(40*time.Minute), hungLoad)
	if f.Verdict != Drain || !f.DrainConfirmed {
		t.Fatalf("setup: expected a confirmed Drain, got %s %t", f.Verdict, f.DrainConfirmed)
	}
	f = d.cycle(now, snapshot(now, "gpu001", 8, busyLoad))
	if !f.RecoveredAfterDrain {
		t.Fatal("recovery of a drained job should be flagged so the daemon can say the node stays drained")
	}
}

func TestConstructorRejectsUnusableConfig(t *testing.T) {
	good := DefaultThresholds()
	for _, tc := range []struct {
		name   string
		mutate func(*Thresholds)
	}{
		{"zero window", func(t *Thresholds) { t.Window = 0 }},
		{"zero min samples", func(t *Thresholds) { t.MinSamples = 0 }},
		{"zero gap", func(t *Thresholds) { t.MaxSampleGap = 0 }},
		{"gap equals window", func(t *Thresholds) { t.MaxSampleGap = t.Window }},
		{"util over 100", func(t *Thresholds) { t.UtilPct = 150 }},
		{"util zero", func(t *Thresholds) { t.UtilPct = 0 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			th := good
			tc.mutate(&th)
			if _, err := New(th, DefaultStages(), Exemptions{}); err == nil {
				t.Fatal("expected construction to fail")
			}
		})
	}
}

// ── Randomized invariant ───────────────────────────────────────────────────

// Over random streams with outages, unreadable GPUs, bursts of work, jittered
// cycle times, random stage configurations and failing drains, check each
// verdict against an independent record of what was actually delivered:
//
//   - any verdict >= Alert at T: the deliveries in [T-Window, T] are all
//     readable breaching snapshots, the first is no later than T-Window+gap,
//     the last no earlier than T-gap, and no two consecutive ones are more
//     than gap apart. That is "a breach held across the whole window".
//   - any Cancel at T: a Drain was confirmed at some D <= T-Window, an Alert
//     was reported before D, and the same contiguity holds over [D-Window, T].
func TestInvariant_NoEscalationWithoutContiguousEvidence(t *testing.T) {
	th := DefaultThresholds()
	totalAlerts, totalCancels := 0, 0
	type delivery struct {
		at       time.Time
		breaches bool // readable, every GPU below threshold
	}
	for seed := int64(1); seed <= 300; seed++ {
		rng := rand.New(rand.NewSource(seed))
		stages := []Stage{{After: 0, Verdict: Alert}}
		if rng.Intn(2) == 0 {
			stages = append(stages, Stage{After: time.Duration(rng.Intn(90)) * time.Minute, Verdict: Drain})
		}
		stages = append(stages, Stage{After: time.Duration(rng.Intn(120)) * time.Minute, Verdict: Cancel})
		e := newEngine(t, th, stages)
		if rng.Intn(2) == 0 {
			_ = e.AllowCancel(SigIdle, SigHung)
		}
		j := job("1", 8, 6*time.Hour)

		var deliveries []delivery
		var confirmedAt, alertTimes []time.Time

		contiguous := func(from, to time.Time) error {
			var ds []delivery
			for _, d := range deliveries {
				if !d.at.Before(from) && !d.at.After(to) {
					ds = append(ds, d)
				}
			}
			if len(ds) == 0 {
				return fmt.Errorf("no deliveries in [%s, %s]", from.Sub(base), to.Sub(base))
			}
			if ds[0].at.After(from.Add(th.MaxSampleGap)) {
				return fmt.Errorf("first delivery %s leaves the start uncovered", ds[0].at.Sub(base))
			}
			if to.Sub(ds[len(ds)-1].at) > th.MaxSampleGap {
				return fmt.Errorf("last delivery %s is stale at %s", ds[len(ds)-1].at.Sub(base), to.Sub(base))
			}
			for i, d := range ds {
				if !d.breaches {
					return fmt.Errorf("non-breaching or unreadable delivery at %s", d.at.Sub(base))
				}
				if i > 0 && d.at.Sub(ds[i-1].at) > th.MaxSampleGap {
					return fmt.Errorf("hole of %s before %s", d.at.Sub(ds[i-1].at), d.at.Sub(base))
				}
			}
			return nil
		}

		now := base
		for c := 0; c < 250; c++ {
			now = now.Add(interval + time.Duration(rng.Intn(20)-10)*time.Second)
			var samples []gpu.Sample
			breaches := false
			switch r := rng.Intn(100); {
			case r < 3: // outage: nothing delivered this cycle
			case r < 5: // long outage
				now = now.Add(time.Duration(rng.Intn(60)) * time.Minute)
			case r < 7:
				samples = snapshot(now, "gpu001", 8, idleLoad)
				samples[rng.Intn(8)].UtilKnown = false
			case r < 9:
				samples = snapshot(now, "gpu001", 8, busyLoad)
			default:
				l := idleLoad
				if rng.Intn(2) == 0 {
					l = hungLoad
				}
				samples = snapshot(now, "gpu001", 8, l)
				breaches = true
			}
			if samples != nil {
				deliveries = append(deliveries, delivery{now, breaches})
			}
			e.Observe(j, samples, now)
			f := e.Evaluate(j, now)
			if f.Verdict >= Alert {
				if err := contiguous(now.Add(-th.Window), now); err != nil {
					t.Fatalf("seed %d cycle %d: %s without a covered window: %v", seed, c, f.Verdict, err)
				}
				alertTimes = append(alertTimes, now)
			}
			if f.Verdict >= Alert {
				totalAlerts++
			}
			if f.Verdict == Cancel {
				totalCancels++
				var d time.Time
				for _, ca := range confirmedAt {
					if !ca.After(now.Add(-th.Window)) {
						d = ca
					}
				}
				if d.IsZero() {
					t.Fatalf("seed %d cycle %d: Cancel without a drain confirmed >= Window earlier", seed, c)
				}
				alerted := false
				for _, a := range alertTimes {
					if a.Before(d) {
						alerted = true
					}
				}
				if !alerted {
					t.Fatalf("seed %d cycle %d: Cancel without an Alert before the drain", seed, c)
				}
				if err := contiguous(d.Add(-th.Window), now); err != nil {
					t.Fatalf("seed %d cycle %d: Cancel without contiguous evidence since before the drain: %v", seed, c, err)
				}
			}
			if f.Verdict == Drain && !f.DrainConfirmed && rng.Intn(4) != 0 {
				e.ConfirmDrain(j, now)
				confirmedAt = append(confirmedAt, now)
			}
		}
	}
	// Guard against a vacuous pass: the streams must actually reach the rungs
	// being checked.
	if totalAlerts == 0 || totalCancels == 0 {
		t.Fatalf("invariant never exercised: %d alerts, %d cancels", totalAlerts, totalCancels)
	}
	t.Logf("checked %d verdicts >= Alert and %d Cancels", totalAlerts, totalCancels)
}
