// Package gpu provides GPU telemetry, abstracted behind a Source interface.
//
// Three backends exist: nvidia-smi (needs nothing but the driver), a reader for
// dcgm-exporter's Prometheus endpoint, and a simulator. The simulator is what
// makes the whole daemon testable and demoable on a laptop with no NVIDIA
// hardware, which is what lets `make demo` work.
//
// The rule every backend follows: a value that could not be read is unknown,
// and unknown is never zero. Reading "[N/A]" as 0% utilization would turn
// missing telemetry into the strongest possible evidence of idleness, which is
// the opposite of what it is.
package gpu

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

// Sample is one observation of one GPU at one instant.
type Sample struct {
	Timestamp time.Time
	NodeName  string
	GPUIndex  int
	// UUID is the device UUID where the backend reports one. Informational.
	UUID string

	// SMUtilPct is the backend's activity measure in percent. For nvidia-smi
	// it is "utilization.gpu": the percentage of the sample period during which
	// at least one kernel was executing.
	//
	// It is a coarse signal. A kernel that occupies one SM out of 108 reports
	// the same 100% as a fully saturated device, which is exactly why the
	// policy engine does not treat high utilization as proof of health — only
	// low utilization as evidence of a problem.
	SMUtilPct float64

	// MemUsedBytes and MemTotalBytes describe framebuffer occupancy. Held
	// memory with no compute is the signature of a hung process rather than an
	// idle one, and the two want different responses.
	MemUsedBytes  uint64
	MemTotalBytes uint64

	// PowerWatts distinguishes a genuinely idle device from one that is busy in
	// a way SM utilization fails to capture.
	PowerWatts float64

	// PIDs are the compute processes resident on the device. Only meaningful
	// when PIDsKnown is true; an empty list is then a positive statement that
	// no process holds a context.
	PIDs []int

	// Validity. Each flag says whether the matching field above was actually
	// read. All of them default to false, and that default is deliberate: a
	// Source that forgets to set one produces a sample the policy engine
	// refuses to judge (utilization, memory) or cannot use as evidence of
	// idleness (power, processes). It never produces a sample read as idle.
	UtilKnown  bool
	MemKnown   bool
	PowerKnown bool
	PIDsKnown  bool

	// MIG is true when the GPU is in MIG mode. NVIDIA documents that
	// device-level utilization is not supported on MIG-enabled GPUs, so the
	// engine refuses to judge them rather than read the gap as 0%.
	MIG bool

	// Jobs lists the workload IDs the telemetry backend attributes this GPU to
	// (dcgm-exporter's hpc_job label, when its HPC job mapping is enabled).
	// Empty means the backend made no attribution, not that no job holds the
	// GPU. Used only as a cross-check that can stop a judgement, never to
	// start one.
	Jobs []string

	// Pods lists "namespace/pod" values from dcgm-exporter's Kubernetes
	// mapping. Informational only; no decision reads it.
	Pods []string

	// Note explains, for logs, why a field is unknown. Optional.
	Note string
}

// Judgeable reports whether the sample carries the two measurements the policy
// engine cannot do without, on a device whose utilization is meaningful.
func (s Sample) Judgeable() bool {
	return s.UtilKnown && s.MemKnown && !s.MIG
}

// MemUsedFraction returns framebuffer occupancy in [0,1]. It returns 0 when
// memory is unknown; callers must check MemKnown (or Judgeable) first.
func (s Sample) MemUsedFraction() float64 {
	if !s.MemKnown || s.MemTotalBytes == 0 {
		return 0
	}
	return float64(s.MemUsedBytes) / float64(s.MemTotalBytes)
}

// Source yields GPU samples for the GPUs on a set of nodes.
type Source interface {
	// Sample returns one observation per visible GPU. Implementations must not
	// block indefinitely; honour the context deadline.
	//
	// A per-node backend (nvidia-smi, or dcgm-exporter on localhost) ignores
	// nodes and reports only its own. A central backend samples every node it
	// is given and may return a *PartialError alongside the samples it did
	// collect.
	Sample(ctx context.Context, nodes []string) ([]Sample, error)

	// Name identifies the backend in logs and metrics.
	Name() string

	// Close releases any resources the backend holds.
	Close() error
}

// ErrUnavailable signals that a backend cannot run in this environment: no
// binary, no driver, no endpoint. There is no automatic fallback to another
// backend. The cycle fails, gpu_reaper_scrape_errors_total counts it, and no
// job is judged on data that was never collected.
var ErrUnavailable = errors.New("gpu source unavailable in this environment")

// PartialError reports nodes that could not be sampled by a central backend.
// The samples returned alongside it are valid for the other nodes. Jobs on a
// failed node get no evidence for this cycle, which the daemon treats as a hole
// in the record rather than as idleness.
type PartialError struct {
	Failed map[string]error
}

func (e *PartialError) Error() string {
	nodes := make([]string, 0, len(e.Failed))
	for n := range e.Failed {
		nodes = append(nodes, n)
	}
	sort.Strings(nodes)
	parts := make([]string, 0, len(nodes))
	for _, n := range nodes {
		parts = append(parts, fmt.Sprintf("%s: %v", n, e.Failed[n]))
	}
	return fmt.Sprintf("%d node(s) not sampled: %s", len(nodes), strings.Join(parts, "; "))
}
