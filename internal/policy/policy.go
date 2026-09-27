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
// Concretely, all of these must hold before the engine reports a breach:
//
//  1. The job has a known start time and is past its warmup. Samples taken
//     during warmup are discarded, not merely ignored: staging looks idle.
//  2. Every GPU sample in the window is readable (utilization and memory
//     known, not MIG). Unreadable is unknown, and unknown is never idle.
//  3. For every node of the job, the window is covered: at least MinSamples
//     distinct observation times (cycles, not per-GPU samples), the oldest no
//     later than Window-MaxSampleGap ago, the newest no older than
//     MaxSampleGap, and no hole between consecutive observations larger than
//     MaxSampleGap. A hole means the collector failed, not that the GPU was
//     idle.
//  4. The evidence covers at least as many GPUs as the job holds.
//  5. Every sample in the window breaches the threshold (peak, not mean).
//  6. The job is not exempt by user, account, partition, QOS, or name.
//
// Whenever 2, 3 or 4 fails, the job's breach clock and escalation history are
// reset, so evidence gathered before a hole never counts after it.
//
// Escalation past Alert is further gated: see Evaluate.
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

// Verdict is the escalation ladder, in order. Alert can be reached on the
// first confirmed breach; Drain only after a previous cycle reported Alert;
// Cancel only after a Drain was confirmed executed at least one Window earlier.
type Verdict int

const (
	// Healthy: the allocation is doing work, or we lack evidence that it isn't.
	Healthy Verdict = iota
	// Watching: breaching, but no configured stage has been reached yet.
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
	// SigIdle: no compute, no memory held, no processes (positively known),
	// idle power (positively known). Nothing is running. The only signature
	// that may be cancelled by default.
	SigIdle Signature = "idle"

	// SigHung: memory held and processes resident, but no compute. Classic
	// deadlock — a collective waiting on a peer that died, or a stuck NCCL
	// ring. Worth alerting loudly. It can be drained, but it is not
	// cancel-eligible unless the operator lists it in allow_cancel_signatures,
	// because it is also what a long checkpoint write looks like.
	SigHung Signature = "hung"

	// SigStarved: low but nonzero compute with memory held. Usually a data
	// loader bottleneck, which is a real waste but a *tuning* problem, not a
	// failure. Alert only.
	SigStarved Signature = "starved"

	// SigUnknown: breaching, but the evidence does not match anything we
	// model — including when process or power data is unavailable. Never
	// escalated past Alert.
	SigUnknown Signature = "unknown"
)

// Thresholds configures detection. Zero values are not useful defaults; use
// DefaultThresholds.
type Thresholds struct {
	// UtilPct is the utilization ceiling below which a sample counts as a
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
	// Warmup is how long after job start the engine refuses to judge, and
	// how long after start samples are discarded.
	Warmup time.Duration
	// MinSamples is the minimum number of distinct observation times per node
	// in the window. It counts cycles, not GPUs: one snapshot of an 8-GPU node
	// is one observation.
	MinSamples int
	// MaxSampleGap is the largest acceptable hole between observations. A
	// larger gap means the collector was down, and absence of data is not
	// evidence of idleness.
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

// Stage maps dwell time in breach to an escalation. Dwell is measured from the
// first breaching observation in the covered window, so it is already close to
// Window when a breach is first confirmed.
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

// DefaultCancelSignatures is the set of signatures a Cancel stage may apply
// to when the operator does not say otherwise.
func DefaultCancelSignatures() []Signature { return []Signature{SigIdle} }

// Exemptions lists allocations the engine must never escalate.
type Exemptions struct {
	Users       []string `yaml:"users"`
	Accounts    []string `yaml:"accounts"`
	Partitions  []string `yaml:"partitions"`
	QOS         []string `yaml:"qos"`
	NamePattern string   `yaml:"name_pattern"`

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
	// Reason is human-readable. The cluster actor derives the shorter string
	// it writes into Slurm (drain Reason, cancel AdminComment) from the
	// signature and job ID.
	Reason string

	MeanUtilPct   float64
	PeakUtilPct   float64
	MemHeldFrac   float64
	MeanPowerW    float64
	PowerKnown    bool
	BreachedSince time.Time
	// Observations is the number of distinct observation times in the window;
	// GPUs is the number of distinct GPUs they cover.
	Observations int
	GPUs         int

	// WastedGPUHours is GPUs x dwell for the current breach: a gauge-style
	// "currently accruing" figure that drops to zero when the breach ends.
	WastedGPUHours float64
	// WastedGPUSeconds is the waste observed since the previous evaluation of
	// this job (the first evaluation of a breach contributes the whole
	// observed breach). Summed across cycles it is a total that does not
	// vanish when a job ends.
	WastedGPUSeconds float64

	// DrainConfirmed is true once a drain for the current breach of this job
	// incarnation has been executed (or, in observe mode, dry-run) and
	// confirmed back to the engine. The cluster actor uses it to drain once
	// per breach, not every cycle. Any reset (recovery, a hole in the
	// evidence) clears it, so a job that climbs back to Drain is drained again.
	DrainConfirmed bool
	// RecoveredAfterDrain is set when a job whose drain was confirmed shows
	// real work again. Its nodes stay drained; the daemon logs a warning.
	RecoveredAfterDrain bool
}

// jobState is the engine's memory across evaluation cycles for one job
// incarnation (JobID + StartTime).
type jobState struct {
	startTime        time.Time
	samples          []gpu.Sample
	breachedSince    time.Time
	lastVerdict      Verdict
	drainConfirmedAt time.Time
	accountedUntil   time.Time
}

// reset discards everything learned about the job's breach. Called whenever
// the evidence has a hole: after it, the window must be covered again and the
// ladder climbed again from Alert.
func (st *jobState) reset() {
	st.breachedSince = time.Time{}
	st.lastVerdict = Healthy
	st.drainConfirmedAt = time.Time{}
	st.accountedUntil = time.Time{}
}

// Engine evaluates jobs against thresholds. Not safe for concurrent use;
// the daemon drives it from a single loop.
type Engine struct {
	thresholds       Thresholds
	stages           []Stage
	exemptions       Exemptions
	cancelSignatures map[Signature]bool
	states           map[string]*jobState
}

// New builds an Engine. Stages are sorted so evaluation can walk them in order.
// Cancel is allowed only for DefaultCancelSignatures until AllowCancel says
// otherwise.
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
	// A gap tolerance of zero would make every pair of observations a gap; one
	// at or above the window would make a single snapshot "cover" the window.
	if t.MaxSampleGap <= 0 || t.MaxSampleGap >= t.Window {
		return nil, fmt.Errorf("thresholds.max_sample_gap must be positive and less than window")
	}
	if t.UtilPct <= 0 || t.UtilPct > 100 {
		return nil, fmt.Errorf("thresholds.util_pct must be in (0, 100]")
	}
	sorted := append([]Stage(nil), stages...)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].After < sorted[j].After })
	e := &Engine{
		thresholds: t,
		stages:     sorted,
		exemptions: ex,
		states:     map[string]*jobState{},
	}
	if err := e.AllowCancel(DefaultCancelSignatures()...); err != nil {
		return nil, err
	}
	return e, nil
}

// AllowCancel sets which signatures a Cancel stage applies to. Only idle and
// hung may be listed; starved and unknown are capped at Alert regardless.
func (e *Engine) AllowCancel(sigs ...Signature) error {
	m := map[Signature]bool{}
	for _, s := range sigs {
		if s != SigIdle && s != SigHung {
			return fmt.Errorf("signature %q can never be cancelled (allowed: idle, hung)", s)
		}
		m[s] = true
	}
	e.cancelSignatures = m
	return nil
}

func (e *Engine) state(j slurm.Job) *jobState {
	st := e.states[j.JobID]
	// A requeued job keeps its JobID but starts again. Nothing the previous
	// incarnation did may count against the new one.
	if st == nil || !st.startTime.Equal(j.StartTime) {
		st = &jobState{startTime: j.StartTime}
		e.states[j.JobID] = st
	}
	return st
}

// Observe records samples for a job. Samples taken before the end of warmup
// are discarded, as are samples older than the window.
func (e *Engine) Observe(j slurm.Job, samples []gpu.Sample, now time.Time) {
	st := e.state(j)
	judgedFrom := j.StartTime.Add(e.thresholds.Warmup)
	for _, s := range samples {
		if j.StartTime.IsZero() || s.Timestamp.Before(judgedFrom) {
			continue
		}
		st.samples = append(st.samples, s)
	}

	cutoff := now.Add(-e.thresholds.Window)
	kept := st.samples[:0]
	for _, s := range st.samples {
		if !s.Timestamp.Before(cutoff) {
			kept = append(kept, s)
		}
	}
	st.samples = kept
}

// Drop discards everything the engine knows about a job. The daemon calls it
// for a job it could not observe this cycle (shared node, partial evidence),
// so evidence from before the gap can never be combined with evidence after.
func (e *Engine) Drop(jobID string) { delete(e.states, jobID) }

// Forget drops state for jobs that are no longer running, so the engine does
// not leak a map entry per job for the lifetime of the process.
func (e *Engine) Forget(activeJobIDs map[string]bool) {
	for id := range e.states {
		if !activeJobIDs[id] {
			delete(e.states, id)
		}
	}
}

// ConfirmDrain records that the drain for this job incarnation was executed
// (or dry-run in observe mode). Cancel is gated on it. It is a no-op unless
// the engine's last verdict for exactly this incarnation was Drain, so a
// confirmation cannot leak across a requeue or a reset.
func (e *Engine) ConfirmDrain(j slurm.Job, at time.Time) {
	st := e.states[j.JobID]
	if st == nil || !st.startTime.Equal(j.StartTime) || st.lastVerdict != Drain {
		return
	}
	if st.drainConfirmedAt.IsZero() {
		st.drainConfirmedAt = at
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
	if j.StartTime.IsZero() {
		e.Drop(j.JobID)
		f.Reason = "start time unknown; not judging"
		return f
	}
	if j.Age(now) < e.thresholds.Warmup {
		f.Reason = fmt.Sprintf("within warmup (%s)", e.thresholds.Warmup)
		return f
	}

	st := e.states[j.JobID]
	if st == nil || !st.startTime.Equal(j.StartTime) || len(st.samples) == 0 {
		f.Reason = "no samples"
		return f
	}

	// Evidence checks. Any failure is a hole in the record: reset the breach
	// clock and the escalation history so nothing from before the hole counts.
	cov, why := e.coverage(st.samples, j, now)
	f.Observations, f.GPUs = cov.observations, cov.gpus
	if why != "" {
		wasDrained := !st.drainConfirmedAt.IsZero()
		st.reset()
		f.Reason = why
		if wasDrained {
			f.Reason += "; escalation history reset (nodes remain drained)"
		}
		return f
	}

	stats := summarize(st.samples)
	f.MeanUtilPct = stats.meanUtil
	f.PeakUtilPct = stats.peakUtil
	f.MemHeldFrac = stats.meanMemFrac
	f.MeanPowerW = stats.meanPower
	f.PowerKnown = stats.powerKnown

	// Peak, not mean: one busy sample anywhere in the window is enough to say
	// the allocation is alive.
	if stats.peakUtil >= e.thresholds.UtilPct {
		f.RecoveredAfterDrain = !st.drainConfirmedAt.IsZero()
		st.reset()
		f.Reason = fmt.Sprintf("peak utilization %.1f%% at or above %.1f%%", stats.peakUtil, e.thresholds.UtilPct)
		return f
	}

	// Sustained breach confirmed.
	if st.breachedSince.IsZero() {
		st.breachedSince = stats.first
	}
	f.BreachedSince = st.breachedSince
	dwell := now.Sub(st.breachedSince)
	gpus := cov.gpus
	if j.GPUCount < gpus {
		// Idle GPUs on the node that the job did not allocate are not its waste.
		gpus = j.GPUCount
	}
	f.WastedGPUHours = float64(gpus) * dwell.Hours()
	from := st.breachedSince
	if st.accountedUntil.After(from) {
		from = st.accountedUntil
	}
	if now.After(from) {
		f.WastedGPUSeconds = float64(gpus) * now.Sub(from).Seconds()
	}
	st.accountedUntil = now

	f.Signature = classify(stats, e.thresholds)
	target := e.stageFor(dwell)

	// Anything we do not positively understand stops at Alert. Same for
	// starvation, which is a tuning problem and not ours to kill over.
	if (f.Signature == SigUnknown || f.Signature == SigStarved) && target > Alert {
		target = Alert
	}
	// A signature the operator has not opted into cancelling stops at Drain.
	if target == Cancel && !e.cancelSignatures[f.Signature] {
		target = Drain
	}

	// The ladder. Alert is reachable on the first confirmed breach: it is only
	// a notification. Drain removes capacity until an admin resumes the node,
	// so it needs a previous cycle to have reported Alert (logged, and posted
	// to Slack if configured). Cancel is the only irreversible action, so it
	// needs a Drain that was confirmed executed, and at least one Window spent
	// in that state. No stage configuration, clock jump or restart can take a
	// job from healthy to terminated: every reset above sends it back to the
	// bottom, and a daemon restart starts with no history at all.
	if target >= Drain && st.lastVerdict < Alert {
		target = Alert
	}
	if target == Cancel {
		drained := !st.drainConfirmedAt.IsZero()
		if st.lastVerdict < Drain || !drained || now.Sub(st.drainConfirmedAt) < e.thresholds.Window {
			target = Drain
		}
	}
	st.lastVerdict = target
	f.Verdict = target
	f.DrainConfirmed = !st.drainConfirmedAt.IsZero()

	power := "unknown"
	if stats.powerKnown {
		power = fmt.Sprintf("%.0fW", stats.meanPower)
	}
	f.Reason = fmt.Sprintf(
		"%s: peak util %.1f%% (mean %.1f%%) below %.1f%% for %s across %d observations of %d GPU(s); mem held %.0f%%, mean power %s; ~%.1f GPU-hours in this breach",
		f.Signature, stats.peakUtil, stats.meanUtil, e.thresholds.UtilPct,
		dwell.Round(time.Second), cov.observations, cov.gpus,
		stats.meanMemFrac*100, power, f.WastedGPUHours,
	)
	return f
}

type coverageInfo struct {
	observations int // minimum distinct observation times over the job's nodes
	gpus         int // distinct (node, GPU index) pairs in the window
}

// coverage checks that the evidence in the window can support a verdict. It
// returns a reason string when it cannot.
func (e *Engine) coverage(samples []gpu.Sample, j slurm.Job, now time.Time) (coverageInfo, string) {
	t := e.thresholds
	var info coverageInfo

	type gpuKey struct {
		node string
		idx  int
	}
	gpus := map[gpuKey]bool{}
	byNode := map[string]map[time.Time]int{}
	for _, s := range samples {
		if !s.Judgeable() {
			what := "utilization or memory unreadable"
			if s.MIG {
				what = "MIG enabled"
			}
			return info, fmt.Sprintf("telemetry not judgeable on %s GPU %d (%s); not judging", s.NodeName, s.GPUIndex, what)
		}
		gpus[gpuKey{s.NodeName, s.GPUIndex}] = true
		if byNode[s.NodeName] == nil {
			byNode[s.NodeName] = map[time.Time]int{}
		}
		byNode[s.NodeName][s.Timestamp]++
	}
	info.gpus = len(gpus)

	nodes := j.Nodes
	if len(nodes) == 0 {
		return info, "job has no nodes; not judging"
	}
	info.observations = -1
	for _, n := range nodes {
		times := byNode[n]
		if len(times) == 0 {
			return info, fmt.Sprintf("no samples for node %s; not judging", n)
		}
		ts := make([]time.Time, 0, len(times))
		want := -1
		for at, count := range times {
			ts = append(ts, at)
			// Every observation of a node should see the same GPUs. One that
			// saw fewer may have missed the busy one.
			if want == -1 {
				want = count
			} else if count != want {
				return info, fmt.Sprintf("node %s reported a varying number of GPUs across observations; not judging", n)
			}
		}
		sort.Slice(ts, func(a, b int) bool { return ts[a].Before(ts[b]) })
		if info.observations == -1 || len(ts) < info.observations {
			info.observations = len(ts)
		}
		if len(ts) < t.MinSamples {
			return info, fmt.Sprintf("insufficient observations on %s (%d < %d)", n, len(ts), t.MinSamples)
		}
		if newest := ts[len(ts)-1]; now.Sub(newest) > t.MaxSampleGap {
			return info, fmt.Sprintf("newest sample for %s is %s old, beyond max gap %s; assuming collector fault",
				n, now.Sub(newest).Round(time.Second), t.MaxSampleGap)
		}
		if oldest := ts[0]; oldest.After(now.Add(-t.Window + t.MaxSampleGap)) {
			return info, fmt.Sprintf("window not yet covered on %s (evidence spans %s of %s)",
				n, now.Sub(oldest).Round(time.Second), t.Window)
		}
		for i := 1; i < len(ts); i++ {
			if gap := ts[i].Sub(ts[i-1]); gap > t.MaxSampleGap {
				return info, fmt.Sprintf("sample gap %s on %s exceeds max %s; assuming collector fault",
					gap.Round(time.Second), n, t.MaxSampleGap)
			}
		}
	}
	if info.gpus < j.GPUCount {
		return info, fmt.Sprintf("evidence covers %d GPU(s) but the job holds %d; not judging", info.gpus, j.GPUCount)
	}
	return info, ""
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

// classify separates the failure modes that deserve different handling. Idle
// and hung each need positive evidence about processes; power must be known
// for idle. Missing evidence falls through to starved or unknown, both capped
// at Alert.
func classify(s stats, t Thresholds) Signature {
	memHeld := s.meanMemFrac >= t.MemHeldFraction

	switch {
	case !memHeld && s.pidsKnown && s.maxPIDs == 0 && s.powerKnown && s.meanPower < t.IdlePowerWatts:
		// Nothing resident, nothing drawing power: the allocation is empty.
		return SigIdle
	case memHeld && s.maxPIDs > 0 && s.peakUtil < 1.0:
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
	powerKnown  bool // every sample had a readable power value
	pidsKnown   bool // every sample had a readable process list
	maxPIDs     int  // over samples whose process list was readable
	first       time.Time
}

func summarize(samples []gpu.Sample) stats {
	var st stats
	if len(samples) == 0 {
		return st
	}
	var sumUtil, sumMem, sumPower float64
	st.first = samples[0].Timestamp
	st.powerKnown, st.pidsKnown = true, true
	for _, s := range samples {
		sumUtil += s.SMUtilPct
		sumMem += s.MemUsedFraction()
		if s.PowerKnown {
			sumPower += s.PowerWatts
		} else {
			st.powerKnown = false
		}
		if s.SMUtilPct > st.peakUtil {
			st.peakUtil = s.SMUtilPct
		}
		if s.PIDsKnown {
			if n := len(s.PIDs); n > st.maxPIDs {
				st.maxPIDs = n
			}
		} else {
			st.pidsKnown = false
		}
		if s.Timestamp.Before(st.first) {
			st.first = s.Timestamp
		}
	}
	n := float64(len(samples))
	st.meanUtil = sumUtil / n
	st.meanMemFrac = sumMem / n
	if st.powerKnown {
		st.meanPower = sumPower / n
	}
	return st
}
