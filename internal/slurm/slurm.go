// Package slurm reads job state from a Slurm controller and acts on it.
//
// Two read backends: the REST API (slurmrestd) where a site runs it, and
// `squeue` shelling out where it does not. squeue is the fallback rather than
// the primary because parsing CLI output is brittle, but a great many clusters
// have no slurmrestd and a reaper that requires one is a reaper nobody deploys.
package slurm

import (
	"context"
	"time"
)

// Job is the subset of Slurm job state the policy engine needs.
type Job struct {
	JobID     string
	Name      string
	User      string
	Account   string
	Partition string
	QOS       string
	State     string

	StartTime time.Time
	Nodes     []string

	// GPUCount is the total GPUs allocated across all nodes, parsed from
	// TRES. Zero means a CPU-only job, which the reaper ignores entirely.
	GPUCount int
}

// Age is how long the job has been running.
func (j Job) Age(now time.Time) time.Duration {
	if j.StartTime.IsZero() {
		return 0
	}
	return now.Sub(j.StartTime)
}

// IsRunning reports whether the job currently holds an allocation.
func (j Job) IsRunning() bool {
	return j.State == "RUNNING"
}

// Source lists currently running jobs.
type Source interface {
	RunningJobs(ctx context.Context) ([]Job, error)
	Name() string
}

// Controller performs state-changing operations against the cluster.
//
// Every method is destructive or near-destructive, which is why the interface
// is separate from Source: it makes the read-only path impossible to confuse
// with the write path, and makes a no-op implementation trivial for dry runs.
type Controller interface {
	// Cancel terminates a job, recording reason in the accounting record so a
	// user asking "what happened to job 12345" gets an answer.
	Cancel(ctx context.Context, jobID, reason string) error

	// Drain marks a node unavailable for new work without disturbing what is
	// already running on it.
	Drain(ctx context.Context, node, reason string) error

	Name() string
}
