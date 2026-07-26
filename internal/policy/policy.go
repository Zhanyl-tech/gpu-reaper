// Package policy decides whether an allocation is being wasted, and what to do
// about it.
//
// The whole design is shaped by an asymmetry: killing a healthy job is far
// worse than letting a wasted one run another hour. A researcher whose 40-hour
// run is cancelled at hour 39 loses 39 hours and their trust in the tool; a
// wasted allocation that survives one extra cycle costs one extra cycle. So
// every default here is conservative, every escalation requires sustained
// evidence, and the destructive stages are opt-in.
//
// Concretely, five things must all hold before the engine escalates at all:
//
//  1. The job is past its warmup — startup, container pull, and dataset staging
//     legitimately show zero GPU utilization.
//  2. The observation window is fully covered by samples. A gap means the
//     collector failed, not that the GPU was idle.
//  3. Every sample in the window breaches the threshold. One good sample resets
//     the finding.
//  4. The job is not exempt by user, account, partition, QOS, or name.
//  5. The breach signature is one we actually understand — see classify().
package policy

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/Zhanyl-tech/gpu-reaper/internal/gpu"
	"github.com/Zhanyl-tech/gpu-reaper/internal/slurm"
)

// Verdict is the escalation ladder. Ordered, and the engine only ever moves one
// rung at a time.
type Verdict int

const (
	// Healthy: the allocation is doing work, or we lack evidence that it isn't.
	Healthy Verdict = iota
	// Watching: breaching, but not yet for long enough to report.
	Watching
	// Alert: sustained breach. Notify humans. Never destructive.
	Alert
	// Drain: stop scheduling new work onto these nodes, leave the job alone.
	Drain
	// Cancel: terminate the job. Requires mode=enforce and an explicit stage.
	Cancel
)

func (v Verdict) String() string {
	switch v {
	case Healthy:
		return "healthy"
	case Watching:
		return "watching"
	case Alert:
		return "alert"
	case Drain:
		return "drain"
	case Cancel:
		return "cancel"
	}
	return "unknown"
}

// Signature describes *why* an allocation looks wasted. Different signatures
// deserve different responses, and conflating them is how tools earn a
// reputation for killing good jobs.
type Signature string

const (
	// SigIdle: no compute, no memory held, no processes. Nothing is running.
	// The only signature safe to act on aggressively.
	SigIdle Signature = "idle"

	// SigHung: memory held and processes resident, but no compute. Classic
	// deadlock — a collective waiting on a peer that died, or a stuck NCCL
	// ring. Worth alerting loudly; dangerous to cancel automatically, because
	// it is also what a long checkpoint write looks like.
	SigHung Signature = "hung"

	// SigStarved: low but nonzero compute with memory held. Usually a data
	// loader bottleneck, which is a real waste but a *tuning* problem, not a
	// failure. Alert only. Cancelling someone's job over a slow dataloader is
	// how you get the tool banned.
	SigStarved Signature = "starved"

	// SigUnknown: breaching, but the signature does not match anything we
	// model. Never escalated past Alert.
	SigUnknown Signature = "unknown"
)

// Thresholds configures detection. Zero values are not useful defaults; use
// DefaultThresholds.
type Thresholds struct {
	// UtilPct is the SM-utilization ceiling below which a sample counts as a
	// breach.
	UtilPct float64
	// MemHeldFraction is the framebuffer occupancy above which we consider
	// memory "held" — the difference between idle and hung.
	MemHeldFraction float64
	// IdlePowerWatts is per-GPU draw below which the device is considered
	// genuinely idle rather than merely under-utilized.
	IdlePowerWatts float64
	// Window is the period over which a breach must be sustained.
	Window time.Duration
	// Warmup is how long after job start the engine refuses to judge.
	Warmup time.Duration
	// MinSamples guards against declaring a verdict on thin data.
	MinSamples int
	// MaxSampleGap is the largest acceptable hole in the window. A larger gap
	// means the collector was down, and absence of data is not evidence of
	// idleness.
	MaxSampleGap time.Duration
}

// DefaultThresholds are deliberately forgiving. A site that wants them tighter
// can say so; a site that gets surprised by an aggressive default will turn the
// whole thing off.
func DefaultThresholds() Thresholds {
	return Thresholds{
		UtilPct:         15,
		MemHeldFraction: 0.05,
		IdlePowerWatts:  60,
		Window:          20 * time.Minute,
		Warmup:          15 * time.Minute,
		MinSamples:      8,
		MaxSampleGap:    3 * time.Minute,
	}
}

// Stage maps dwell time in breach to an escalation.
type Stage struct {
	After   time.Duration
	Verdict Verdict
}

// DefaultStages: alert quickly, drain slowly, never cancel unless configured.
func DefaultStages() []Stage {
	return []Stage{
		{After: 0, Verdict: Alert},
		{After: 60 * time.Minute, Verdict: Drain},
	}
}

// Exemptions lists allocations the engine must never escalate.
type Exemptions struct {
	Users       []string
	Accounts    []string
	Partitions  []string
	QOS         []string
	NamePattern string

	nameRe *regexp.Regexp
}

// Compile prepares the name pattern. Call once before use.
func (e *Exemptions) Compile() error {
	if e.NamePattern == "" {
		return nil
	}
	re, err := regexp.Compile(e.NamePattern)
	if err != nil {
		return fmt.Errorf("exemption name_pattern: %w", err)
	}
	e.nameRe = re
	return nil
}

// Covers reports whether a job is exempt.
func (e *Exemptions) Covers(j slurm.Job) bool {
	if contains(e.Users, j.User) ||
		contains(e.Accounts, j.Account) ||
		contains(e.Partitions, j.Partition) ||
		contains(e.QOS, j.QOS) {
		return true
	}
	return e.nameRe != nil && e.nameRe.MatchString(j.Name)
}

func contains(haystack []string, needle string) bool {
	for _, h := range haystack {
		if strings.EqualFold(h, needle) {
			return true
		}
	}
	return false
}

// Finding is the engine's output for one job.
type Finding struct {
	Job       slurm.Job
	Verdict   Verdict
	Signature Signature
	// Reason is written into scancel/scontrol so the affected user can find out
	// what happened without reading our logs.
	Reason string

	MeanUtilPct   float64
	PeakUtilPct   float64
	MemHeldFrac   float64
	MeanPowerW    float64
	BreachedSince time.Time
	SampleCount   int
	// WastedGPUHours is the headline number: GPUs held x hours in breach.
	WastedGPUHours float64
}

// jobState is the engine's memory across evaluation cycles.
type jobState struct {
	samples       []gpu.Sample
	breachedSince time.Time
	lastVerdict   Verdict
}

// Engine evaluates jobs against thresholds. Not safe for concurrent use;
// the daemon drives it from a single loop.
type Engine struct {
	thresholds Thresholds
	stages     []Stage
	exemptions Exemptions
	states     map[string]*jobState
}

// New builds an Engine. Stages are sorted so evaluation can walk them in order.
func New(t Thresholds, stages []Stage, ex Exemptions) (*Engine, error) {
	if err := ex.Compile(); err != nil {
		return nil, err
	}
	if t.Window <= 0 {
		return nil, fmt.Errorf("thresholds.window must be positive")
	}
	if t.MinSamples < 1 {
		return nil, fmt.Errorf("thresholds.min_samples must be at least 1")
	}
	sorted := append([]Stage(nil), stages...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].After < sorted[j].After })
	return &Engine{
		thresholds: t,
		stages:     sorted,
		exemptions: ex,
		states:     map[string]*jobState{},
	}, nil
}

// Observe records samples for a job. Samples older than the window are dropped.
func (e *Engine) Observe(jobID string, samples []gpu.Sample, now time.Time) {
	st, ok := e.states[jobID]
	if !ok {
		st = &jobState{}
		e.states[jobID] = st
	}
	st.samples = append(st.samples, samples...)

	cutoff := now.Add(-e.thresholds.Window)
	kept := st.samples[:0]
	for _, s := range st.samples {
		if !s.Timestamp.Before(cutoff) {
			kept = append(kept, s)
		}
	}
	st.samples = kept
}

// Forget drops state for jobs that are no longer running, so the engine does
// not leak a map entry per job for the lifetime of the process.
func (e *Engine) Forget(activeJobIDs map[string]bool) {
	for id := range e.states {
		if !activeJobIDs[id] {
			delete(e.states, id)
		}
	}
}

// Evaluate returns a Finding for one job.
func (e *Engine) Evaluate(j slurm.Job, now time.Time) Finding {
	f := Finding{Job: j, Verdict: Healthy, Signature: SigUnknown}

	if j.GPUCount == 0 {
		f.Reason = "no GPUs allocated"
		return f
	}
	if e.exemptions.Covers(j) {
		f.Reason = "exempt"
		return f
	}
	if j.Age(now) < e.thresholds.Warmup {
		f.Reason = fmt.Sprintf("within warmup (%s)", e.thresholds.Warmup)
		return f
	}

	st := e.states[j.JobID]
	if st == nil || len(st.samples) == 0 {
		f.Reason = "no samples"
		return f
	}

	stats := summarize(st.samples)
	f.MeanUtilPct = stats.meanUtil
	f.PeakUtilPct = stats.peakUtil
	f.MemHeldFrac = stats.meanMemFrac
	f.MeanPowerW = stats.meanPower
	f.SampleCount = len(st.samples)

	if len(st.samples) < e.thresholds.MinSamples {
		f.Reason = fmt.Sprintf("insufficient samples (%d < %d)", len(st.samples), e.thresholds.MinSamples)
		return f
	}

	// A hole in the data means the collector failed. Treating that as idleness
	// would turn a monitoring outage into a cluster-wide cancellation event —
	// the single most dangerous failure mode this tool has.
	if gap := largestGap(st.samples); gap > e.thresholds.MaxSampleGap {
		st.breachedSince = time.Time{}
		f.Reason = fmt.Sprintf("sample gap %s exceeds max %s; assuming collector fault", gap.Round(time.Second), e.thresholds.MaxSampleGap)
		return f
	}

	// Peak, not mean: one busy sample anywhere in the window is enough to say
	// the allocation is alive.
	if stats.peakUtil >= e.thresholds.UtilPct {
		st.breachedSince = time.Time{}
		st.lastVerdict = Healthy
		f.Reason = fmt.Sprintf("peak utilization %.1f%% at or above %.1f%%", stats.peakUtil, e.thresholds.UtilPct)
		return f
	}

	// Sustained breach confirmed.
	if st.breachedSince.IsZero() {
		st.breachedSince = stats.first
	}
	f.BreachedSince = st.breachedSince
	dwell := now.Sub(st.breachedSince)
	f.WastedGPUHours = float64(j.GPUCount) * dwell.Hours()
	f.Signature = classify(stats, e.thresholds)

	target := e.stageFor(dwell)

	// Anything we do not positively understand stops at Alert. Same for
	// starvation, which is a tuning problem and not ours to kill over.
	if (f.Signature == SigUnknown || f.Signature == SigStarved) && target > Alert {
		target = Alert
	}

	// Cancel is the only irreversible action, so it is the only one gated on
	// history: a job must have been reported and drained on a previous cycle
	// before it can be killed. That makes it impossible for a config reload
	// that widens a stage — or a clock jump — to take a job straight from
	// healthy to terminated, and it guarantees a human saw an alert first.
	//
	// Alert and Drain are reachable immediately once a breach is confirmed.
	// Both are recoverable, and delaying them buys nothing but wasted GPU time.
	if target == Cancel && st.lastVerdict < Drain {
		if st.lastVerdict < Alert {
			target = Alert
		} else {
			target = st.lastVerdict + 1
		}
	}
	st.lastVerdict = target
	f.Verdict = target

	f.Reason = fmt.Sprintf(
		"%s: mean util %.1f%% (peak %.1f%%) below %.1f%% for %s across %d samples; mem held %.0f%%, mean power %.0fW; ~%.1f GPU-hours wasted",
		f.Signature, stats.meanUtil, stats.peakUtil, e.thresholds.UtilPct,
		dwell.Round(time.Minute), len(st.samples),
		stats.meanMemFrac*100, stats.meanPower, f.WastedGPUHours,
	)
	return f
}

func (e *Engine) stageFor(dwell time.Duration) Verdict {
	v := Watching
	for _, s := range e.stages {
		if dwell >= s.After {
			v = s.Verdict
		}
	}
	return v
}

// classify separates the failure modes that deserve different handling.
func classify(s stats, t Thresholds) Signature {
	memHeld := s.meanMemFrac >= t.MemHeldFraction
	hasProcs := s.maxPIDs > 0
	idlePower := s.meanPower < t.IdlePowerWatts

	switch {
	case !memHeld && !hasProcs && idlePower:
		// Nothing resident, nothing drawing power: the allocation is empty.
		return SigIdle
	case memHeld && hasProcs && s.peakUtil < 1.0:
		// Processes hold memory but do no compute at all.
		return SigHung
	case memHeld && s.peakUtil >= 1.0:
		// Some compute, just not much — a bottleneck, not a failure.
		return SigStarved
	default:
		return SigUnknown
	}
}

type stats struct {
	meanUtil    float64
	peakUtil    float64
	meanMemFrac float64
	meanPower   float64
	maxPIDs     int
	first       time.Time
}

func summarize(samples []gpu.Sample) stats {
	var st stats
	if len(samples) == 0 {
		return st
	}
	var sumUtil, sumMem, sumPower float64
	st.first = samples[0].Timestamp
	for _, s := range samples {
		sumUtil += s.SMUtilPct
		sumMem += s.MemUsedFraction()
		sumPower += s.PowerWatts
		if s.SMUtilPct > st.peakUtil {
			st.peakUtil = s.SMUtilPct
		}
		if n := len(s.PIDs); n > st.maxPIDs {
			st.maxPIDs = n
		}
		if s.Timestamp.Before(st.first) {
			st.first = s.Timestamp
		}
	}
	n := float64(len(samples))
	st.meanUtil = sumUtil / n
	st.meanMemFrac = sumMem / n
	st.meanPower = sumPower / n
	return st
}

// largestGap returns the biggest interval between consecutive samples.
func largestGap(samples []gpu.Sample) time.Duration {
	if len(samples) < 2 {
		return 0
	}
	ts := make([]time.Time, len(samples))
	for i, s := range samples {
		ts[i] = s.Timestamp
	}
	sort.Slice(ts, func(i, j int) bool { return ts[i].Before(ts[j]) })

	var max time.Duration
	for i := 1; i < len(ts); i++ {
		if d := ts[i].Sub(ts[i-1]); d > max {
			max = d
		}
	}
	return max
}
