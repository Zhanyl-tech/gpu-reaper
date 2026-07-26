// Package action turns a policy verdict into something that happens.
//
// Actors are deliberately dumb: the policy engine decides, actors execute. That
// split means the destructive paths have no branching logic of their own to get
// wrong, and the interesting decisions are all covered by the policy tests.
package action

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/Zhanyl-tech/gpu-reaper/internal/policy"
	"github.com/Zhanyl-tech/gpu-reaper/internal/slurm"
)

// Actor handles a finding.
type Actor interface {
	Handle(ctx context.Context, f policy.Finding) error
	Name() string
}

// ── Logging ────────────────────────────────────────────────────────────────

// LogActor records findings. Always enabled — an action nobody can audit is
// worse than no action.
type LogActor struct{ Logger *slog.Logger }

func (l LogActor) Name() string { return "log" }

func (l LogActor) Handle(_ context.Context, f policy.Finding) error {
	lvl := slog.LevelInfo
	if f.Verdict >= policy.Drain {
		lvl = slog.LevelWarn
	}
	l.Logger.Log(context.Background(), lvl, "finding",
		"verdict", f.Verdict.String(),
		"signature", string(f.Signature),
		"job_id", f.Job.JobID,
		"user", f.Job.User,
		"account", f.Job.Account,
		"gpus", f.Job.GPUCount,
		"mean_util_pct", round1(f.MeanUtilPct),
		"peak_util_pct", round1(f.PeakUtilPct),
		"wasted_gpu_hours", round1(f.WastedGPUHours),
		"reason", f.Reason,
	)
	return nil
}

func round1(f float64) float64 { return float64(int(f*10+0.5)) / 10 }

// ── Slack ──────────────────────────────────────────────────────────────────

// SlackActor posts findings to an incoming webhook.
type SlackActor struct {
	WebhookURL string
	MinVerdict policy.Verdict
	Client     *http.Client
}

func NewSlackActor(url string, min policy.Verdict) *SlackActor {
	return &SlackActor{WebhookURL: url, MinVerdict: min, Client: &http.Client{Timeout: 10 * time.Second}}
}

func (s *SlackActor) Name() string { return "slack" }

func (s *SlackActor) Handle(ctx context.Context, f policy.Finding) error {
	if f.Verdict < s.MinVerdict {
		return nil
	}

	emoji := ":eyes:"
	if f.Verdict >= policy.Drain {
		emoji = ":rotating_light:"
	}

	text := fmt.Sprintf(
		"%s *%s* — job `%s` (%s, %s) holding *%d GPU(s)*\n"+
			"• signature: `%s`\n• mean util: %.1f%% (peak %.1f%%)\n"+
			"• wasted: *%.1f GPU-hours*\n• %s",
		emoji, f.Verdict.String(), f.Job.JobID, f.Job.User, f.Job.Account,
		f.Job.GPUCount, f.Signature, f.MeanUtilPct, f.PeakUtilPct,
		f.WastedGPUHours, f.Reason,
	)

	body, err := json.Marshal(map[string]string{"text": text})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.WebhookURL, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.Client.Do(req)
	if err != nil {
		return fmt.Errorf("slack webhook: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("slack webhook returned %s", resp.Status)
	}
	return nil
}

// ── Cluster actions ────────────────────────────────────────────────────────

// ClusterActor drains nodes and cancels jobs.
//
// Enforce gates every destructive call. When false the actor logs exactly what
// it would have done and returns — which is the mode every deployment should
// run in until its operators trust the findings.
type ClusterActor struct {
	Controller slurm.Controller
	Enforce    bool
	Logger     *slog.Logger
}

func (c *ClusterActor) Name() string {
	if c.Enforce {
		return "cluster(enforce)"
	}
	return "cluster(dry-run)"
}

func (c *ClusterActor) Handle(ctx context.Context, f policy.Finding) error {
	switch f.Verdict {
	case policy.Drain:
		return c.drain(ctx, f)
	case policy.Cancel:
		return c.cancel(ctx, f)
	default:
		return nil
	}
}

func (c *ClusterActor) drain(ctx context.Context, f policy.Finding) error {
	reason := fmt.Sprintf("gpu-reaper: %s job %s", f.Signature, f.Job.JobID)
	for _, node := range f.Job.Nodes {
		if !c.Enforce {
			c.Logger.Info("would drain", "node", node, "job_id", f.Job.JobID, "reason", reason)
			continue
		}
		if err := c.Controller.Drain(ctx, node, reason); err != nil {
			return err
		}
		c.Logger.Warn("drained", "node", node, "job_id", f.Job.JobID)
	}
	return nil
}

func (c *ClusterActor) cancel(ctx context.Context, f policy.Finding) error {
	reason := fmt.Sprintf("gpu-reaper: %s, %.1f GPU-hours wasted", f.Signature, f.WastedGPUHours)
	if !c.Enforce {
		c.Logger.Info("would cancel", "job_id", f.Job.JobID, "user", f.Job.User, "reason", reason)
		return nil
	}
	if err := c.Controller.Cancel(ctx, f.Job.JobID, reason); err != nil {
		return err
	}
	c.Logger.Warn("cancelled", "job_id", f.Job.JobID, "user", f.Job.User, "reason", reason)
	return nil
}
