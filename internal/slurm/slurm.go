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

	// StartTime is when this incarnation of the job started. Zero means unknown,
	// which the policy engine refuses to judge. A requeued job keeps its JobID
	// but gets a new StartTime, and the engine treats it as a new job.
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
	// Cancel terminates a job. Implementations must record reason where the
	// affected user can find it (CLIController sets the job's AdminComment
	// first and refuses to cancel if it cannot), so "what happened to job
	// 12345" has an answer outside our logs.
	Cancel(ctx context.Context, jobID, reason string) error

	// Drain marks a node unavailable for new work without disturbing what is
	// already running on it. Slurm leaves a drained node DRAINED after its jobs
	// finish; nothing in this tool resumes it.
	Drain(ctx context.Context, node, reason string) error

	Name() string
}
