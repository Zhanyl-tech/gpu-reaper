// Package action turns a policy verdict into something that happens.
//
// Actors are deliberately dumb: the policy engine decides, actors execute. The
// branching they do have (dry run versus enforce, drain once, notify on
// change) is small and covered by action_test.go.
package action

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"sync"
	"time"

	"github.com/Zhanyl-tech/gpu-reaper/internal/policy"
	"github.com/Zhanyl-tech/gpu-reaper/internal/slurm"
)

// Outcome says what an actor actually did with a finding. Metrics are labelled
// with it, so observe mode can never be reported as enforcement.
type Outcome int

const (
	// Noop: nothing to do for this finding (wrong verdict, already done,
	// suppressed as a repeat).
	Noop Outcome = iota
	// DryRun: the actor would have acted but is not allowed to.
	DryRun
	// Acted: the action was performed.
	Acted
)

func (o Outcome) String() string {
	switch o {
	case DryRun:
		return "dry_run"
	case Acted:
		return "acted"
	}
	return "noop"
}

// Actor handles a finding.
type Actor interface {
	Handle(ctx context.Context, f policy.Finding) (Outcome, error)
	Name() string
}

// Forgetter is implemented by actors that keep per-job state, so the daemon
// can release it when jobs end.
type Forgetter interface {
	Forget(active map[string]bool)
}

// ── Logging ────────────────────────────────────────────────────────────────

// LogActor records findings. Always enabled — an action nobody can audit is
// worse than no action. It logs every finding every cycle; de-duplication is
// for notifications, not for the audit trail.
type LogActor struct{ Logger *slog.Logger }

func (l LogActor) Name() string { return "log" }

func (l LogActor) Handle(ctx context.Context, f policy.Finding) (Outcome, error) {
	lvl := slog.LevelInfo
	if f.Verdict >= policy.Drain {
		lvl = slog.LevelWarn
	}
	l.Logger.Log(ctx, lvl, "finding",
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
	return Acted, nil
}

func round1(f float64) float64 { return float64(int(f*10+0.5)) / 10 }

// ── Slack ──────────────────────────────────────────────────────────────────

// SlackActor posts findings to an incoming webhook.
//
// It posts when a job's verdict changes and, if RemindEvery > 0, again after
// that long at the same verdict. Posting every breaching job every cycle
// (about 30 posts an hour per job at a 2-minute interval) trains people to
// mute the channel, which defeats the point of alerting before draining.
type SlackActor struct {
	WebhookURL  string
	MinVerdict  policy.Verdict
	RemindEvery time.Duration
	Client      *http.Client

	mu   sync.Mutex
	sent map[string]sentRecord
	now  func() time.Time
}

type sentRecord struct {
	start   time.Time // job incarnation
	verdict policy.Verdict
	at      time.Time
}

func NewSlackActor(webhook string, min policy.Verdict, remind time.Duration) *SlackActor {
	return &SlackActor{
		WebhookURL: webhook, MinVerdict: min, RemindEvery: remind,
		Client: &http.Client{Timeout: 10 * time.Second},
		sent:   map[string]sentRecord{},
		now:    time.Now,
	}
}

func (s *SlackActor) Name() string { return "slack" }

// Forget drops notification state for jobs that are no longer running.
func (s *SlackActor) Forget(active map[string]bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for id := range s.sent {
		if !active[id] {
			delete(s.sent, id)
		}
	}
}

func (s *SlackActor) Handle(ctx context.Context, f policy.Finding) (Outcome, error) {
	if f.Verdict < s.MinVerdict {
		return Noop, nil
	}
	now := s.now()
	s.mu.Lock()
	prev, seen := s.sent[f.Job.JobID]
	s.mu.Unlock()
	if seen && prev.start.Equal(f.Job.StartTime) && prev.verdict == f.Verdict &&
		(s.RemindEvery <= 0 || now.Sub(prev.at) < s.RemindEvery) {
		return Noop, nil
	}

	emoji := ":eyes:"
	if f.Verdict >= policy.Drain {
		emoji = ":rotating_light:"
	}
	text := fmt.Sprintf(
		"%s *%s* — job `%s` (%s, %s) holding *%d GPU(s)*\n"+
			"• signature: `%s`\n• mean util: %.1f%% (peak %.1f%%)\n"+
			"• in this breach: *%.1f GPU-hours*\n• %s",
		emoji, f.Verdict.String(), f.Job.JobID, f.Job.User, f.Job.Account,
		f.Job.GPUCount, f.Signature, f.MeanUtilPct, f.PeakUtilPct,
		f.WastedGPUHours, f.Reason,
	)
	if err := s.post(ctx, text); err != nil {
		return Noop, err
	}
	s.mu.Lock()
	s.sent[f.Job.JobID] = sentRecord{start: f.Job.StartTime, verdict: f.Verdict, at: now}
	s.mu.Unlock()
	return Acted, nil
}

func (s *SlackActor) post(ctx context.Context, text string) error {
	body, err := json.Marshal(map[string]string{"text": text})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.WebhookURL, bytes.NewReader(body))
	if err != nil {
		// The parse error would quote the URL, and the URL is the secret.
		return errors.New("slack webhook: invalid webhook URL")
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.Client.Do(req)
	if err != nil {
		return redactURLError(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("slack webhook returned %s", resp.Status)
	}
	return nil
}

// redactURLError rebuilds an http.Client error without the request URL. A
// Slack incoming webhook's secret is its path, and *url.Error's text includes
// the full URL, which would otherwise land in the ERROR log.
func redactURLError(err error) error {
	var ue *url.Error
	if !errors.As(err, &ue) {
		return errors.New("slack webhook: request failed")
	}
	host := "(unparseable URL)"
	if u, perr := url.Parse(ue.URL); perr == nil {
		host = u.Scheme + "://" + u.Host
	}
	return fmt.Errorf("slack webhook %s to %s failed: %v", ue.Op, host, ue.Err)
}

// ── Cluster actions ────────────────────────────────────────────────────────

// ClusterActor drains nodes and cancels jobs.
//
// Enforce gates every destructive call. When false the actor logs exactly what
// it would have done and returns DryRun — which is the mode every deployment
// should run in until its operators trust the findings. Observe mode also
// wires in slurm.NoopController, so the two gates are independent.
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

func (c *ClusterActor) Handle(ctx context.Context, f policy.Finding) (Outcome, error) {
	switch f.Verdict {
	case policy.Drain:
		if f.DrainConfirmed {
			// Already drained for this breach. Re-issuing it every cycle only
			// adds scontrol traffic and log noise. After a reset the engine
			// clears DrainConfirmed, so a new breach is drained again.
			return Noop, nil
		}
		return c.drain(ctx, f)
	case policy.Cancel:
		return c.cancel(ctx, f)
	default:
		return Noop, nil
	}
}

func (c *ClusterActor) drain(ctx context.Context, f policy.Finding) (Outcome, error) {
	reason := fmt.Sprintf("gpu-reaper: %s job %s", f.Signature, f.Job.JobID)
	if !c.Enforce {
		for _, node := range f.Job.Nodes {
			c.Logger.Info("would drain", "node", node, "job_id", f.Job.JobID, "reason", reason)
		}
		return DryRun, nil
	}
	for _, node := range f.Job.Nodes {
		// A failure part-way returns an error, so the drain is not confirmed
		// and is retried next cycle (draining an already-drained node is
		// harmless), and Cancel stays blocked.
		if err := c.Controller.Drain(ctx, node, reason); err != nil {
			return Noop, err
		}
		c.Logger.Warn("drained", "node", node, "job_id", f.Job.JobID,
			"note", "gpu-reaper never resumes nodes; resume with scontrol update NodeName="+node+" State=RESUME")
	}
	return Acted, nil
}

func (c *ClusterActor) cancel(ctx context.Context, f policy.Finding) (Outcome, error) {
	reason := fmt.Sprintf("gpu-reaper: cancelled %s job, %.1f GPU-hours in breach", f.Signature, f.WastedGPUHours)
	if !c.Enforce {
		c.Logger.Info("would cancel", "job_id", f.Job.JobID, "user", f.Job.User, "reason", reason)
		return DryRun, nil
	}
	if err := c.Controller.Cancel(ctx, f.Job.JobID, reason); err != nil {
		return Noop, err
	}
	c.Logger.Warn("cancelled", "job_id", f.Job.JobID, "user", f.Job.User, "reason", reason)
	return Acted, nil
}
