// Package gpu provides GPU telemetry, abstracted behind a Source interface.
//
// The interface exists for two reasons. First, production clusters are split
// between DCGM and bare NVML, and some sites have neither exporter deployed and
// only nvidia-smi. Second, and more usefully: a simulator implementation makes
// the whole daemon testable and demoable on a laptop with no NVIDIA hardware,
// which is what lets `make demo` work in 60 seconds.
package gpu

import (
	"context"
	"fmt"
	"time"
)

// Sample is one observation of one GPU at one instant.
type Sample struct {
	Timestamp time.Time
	NodeName  string
	GPUIndex  int

	// SMUtilPct is streaming-multiprocessor utilization: the percentage of the
	// sample period during which at least one kernel was executing. This is
	// what nvidia-smi reports as "GPU-Util".
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
	// a way SM utilization fails to capture. Near-idle draw alongside near-zero
	// utilization is the strongest available evidence that nothing is running.
	PowerWatts float64

	// PIDs are the compute processes resident on the device. Empty means no
	// process holds a context — which is unambiguous, and rare.
	PIDs []int
}

// MemUsedFraction returns framebuffer occupancy in [0,1].
func (s Sample) MemUsedFraction() float64 {
	if s.MemTotalBytes == 0 {
		return 0
	}
	return float64(s.MemUsedBytes) / float64(s.MemTotalBytes)
}

// Source yields GPU samples for the GPUs allocated to a set of nodes.
type Source interface {
	// Sample returns one observation per visible GPU. Implementations must not
	// block indefinitely; honour the context deadline.
	Sample(ctx context.Context, nodes []string) ([]Sample, error)

	// Name identifies the backend in logs and metrics.
	Name() string

	// Close releases any handles (NVML must be shut down explicitly).
	Close() error
}

// ErrUnavailable signals that a backend cannot run in this environment — no
// driver, no DCGM socket, wrong architecture. Callers fall back rather than
// fail: a reaper that refuses to start because DCGM is missing is less useful
// than one that degrades to nvidia-smi and says so.
var ErrUnavailable = fmt.Errorf("gpu source unavailable in this environment")
