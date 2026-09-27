package action

import (
	"bytes"
	"context"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Zhanyl-tech/gpu-reaper/internal/policy"
	"github.com/Zhanyl-tech/gpu-reaper/internal/slurm"
)

type recordingController struct {
	mu        sync.Mutex
	calls     []string
	drainErr  error
	cancelErr error
}

func (r *recordingController) Name() string { return "recording" }
func (r *recordingController) Cancel(_ context.Context, id, reason string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls = append(r.calls, "cancel "+id+" "+reason)
	return r.cancelErr
}
func (r *recordingController) Drain(_ context.Context, node, reason string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls = append(r.calls, "drain "+node+" "+reason)
	return r.drainErr
}

func quietLogger() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

func finding(v policy.Verdict) policy.Finding {
	return policy.Finding{
		Job: slurm.Job{JobID: "42", User: "alice", Nodes: []string{"gpu001", "gpu002"}, GPUCount: 16,
			StartTime: time.Date(2026, 9, 26, 8, 0, 0, 0, time.UTC)},
		Verdict: v, Signature: policy.SigIdle, WastedGPUHours: 3,
	}
}

// The second of the two independent gates: with Enforce=false the controller
// is never called, whatever the verdict. (The first gate, NoopController in
// observe mode, is tested in cmd/gpu-reaper.)
func TestClusterActorNeverCallsControllerWhenNotEnforcing(t *testing.T) {
	for _, v := range []policy.Verdict{policy.Healthy, policy.Watching, policy.Alert, policy.Drain, policy.Cancel} {
		rc := &recordingController{}
		a := &ClusterActor{Controller: rc, Enforce: false, Logger: quietLogger()}
		out, err := a.Handle(context.Background(), finding(v))
		if err != nil {
			t.Fatal(err)
		}
		if len(rc.calls) != 0 {
			t.Fatalf("verdict %s: dry-run called the controller: %v", v, rc.calls)
		}
		want := Noop
		if v == policy.Drain || v == policy.Cancel {
			want = DryRun
		}
		if out != want {
			t.Fatalf("verdict %s: outcome %s, want %s", v, out, want)
		}
	}
}

func TestClusterActorEnforcing(t *testing.T) {
	cases := []struct {
		name      string
		f         policy.Finding
		wantCalls []string
		want      Outcome
	}{
		{"alert does nothing", finding(policy.Alert), nil, Noop},
		{"drain drains every node", finding(policy.Drain), []string{
			"drain gpu001 gpu-reaper: idle job 42", "drain gpu002 gpu-reaper: idle job 42"}, Acted},
		{"confirmed drain is not re-issued", func() policy.Finding {
			f := finding(policy.Drain)
			f.DrainConfirmed = true
			return f
		}(), nil, Noop},
		{"cancel cancels", finding(policy.Cancel), []string{
			"cancel 42 gpu-reaper: cancelled idle job, 3.0 GPU-hours in breach"}, Acted},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rc := &recordingController{}
			a := &ClusterActor{Controller: rc, Enforce: true, Logger: quietLogger()}
			out, err := a.Handle(context.Background(), tc.f)
			if err != nil {
				t.Fatal(err)
			}
			if strings.Join(rc.calls, "|") != strings.Join(tc.wantCalls, "|") {
				t.Fatalf("calls %v, want %v", rc.calls, tc.wantCalls)
			}
			if out != tc.want {
				t.Fatalf("outcome %s, want %s", out, tc.want)
			}
		})
	}
}

func TestClusterActorFailedDrainIsAnErrorNotAnAction(t *testing.T) {
	rc := &recordingController{drainErr: errors.New("permission denied")}
	a := &ClusterActor{Controller: rc, Enforce: true, Logger: quietLogger()}
	out, err := a.Handle(context.Background(), finding(policy.Drain))
	if err == nil || out != Noop {
		t.Fatalf("a failed drain must return an error and no outcome, got %s %v", out, err)
	}
	if len(rc.calls) != 1 {
		t.Fatalf("should stop at the first failure, calls %v", rc.calls)
	}
}

func TestLogActorLogsEveryFinding(t *testing.T) {
	var buf bytes.Buffer
	a := LogActor{Logger: slog.New(slog.NewTextHandler(&buf, nil))}
	out, err := a.Handle(context.Background(), finding(policy.Drain))
	if err != nil || out != Acted {
		t.Fatalf("got %s %v", out, err)
	}
	if !strings.Contains(buf.String(), "verdict=drain") || !strings.Contains(buf.String(), "level=WARN") {
		t.Fatalf("log line: %s", buf.String())
	}
}

// ── Slack ──────────────────────────────────────────────────────────────────

type slackServer struct {
	*httptest.Server
	mu    sync.Mutex
	posts []string
}

func newSlackServer(t *testing.T, status int) *slackServer {
	s := &slackServer{}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		s.mu.Lock()
		s.posts = append(s.posts, string(b))
		s.mu.Unlock()
		w.WriteHeader(status)
	}))
	t.Cleanup(s.Close)
	return s
}

func (s *slackServer) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.posts)
}

func TestSlackPostsOnChangeNotEveryCycle(t *testing.T) {
	srv := newSlackServer(t, http.StatusOK)
	a := NewSlackActor(srv.URL+"/services/T/B/X", policy.Alert, time.Hour)
	clock := time.Date(2026, 9, 26, 12, 0, 0, 0, time.UTC)
	a.now = func() time.Time { return clock }
	ctx := context.Background()

	outcomes := []Outcome{}
	for i := 0; i < 5; i++ { // five cycles at Alert
		o, err := a.Handle(ctx, finding(policy.Alert))
		if err != nil {
			t.Fatal(err)
		}
		outcomes = append(outcomes, o)
		clock = clock.Add(2 * time.Minute)
	}
	if srv.count() != 1 {
		t.Fatalf("five cycles at the same verdict should post once, posted %d (%v)", srv.count(), outcomes)
	}
	if _, err := a.Handle(ctx, finding(policy.Drain)); err != nil {
		t.Fatal(err)
	}
	if srv.count() != 2 {
		t.Fatalf("a verdict change should post, posted %d", srv.count())
	}
	clock = clock.Add(time.Hour)
	if _, err := a.Handle(ctx, finding(policy.Drain)); err != nil {
		t.Fatal(err)
	}
	if srv.count() != 3 {
		t.Fatalf("a reminder is due after RemindEvery, posted %d", srv.count())
	}

	// A requeued job (new StartTime) is a new incarnation and is announced.
	f := finding(policy.Drain)
	f.Job.StartTime = f.Job.StartTime.Add(time.Hour)
	if _, err := a.Handle(ctx, f); err != nil {
		t.Fatal(err)
	}
	if srv.count() != 4 {
		t.Fatalf("new incarnation should post, posted %d", srv.count())
	}

	a.Forget(map[string]bool{})
	if _, err := a.Handle(ctx, finding(policy.Drain)); err != nil {
		t.Fatal(err)
	}
	if srv.count() != 5 {
		t.Fatalf("after Forget the job is new again, posted %d", srv.count())
	}
}

func TestSlackRespectsMinVerdict(t *testing.T) {
	srv := newSlackServer(t, http.StatusOK)
	a := NewSlackActor(srv.URL, policy.Drain, 0)
	if o, _ := a.Handle(context.Background(), finding(policy.Alert)); o != Noop || srv.count() != 0 {
		t.Fatal("alert below min_verdict=drain must not post")
	}
}

func TestSlackFailedPostIsRetriedNextCycle(t *testing.T) {
	srv := newSlackServer(t, http.StatusInternalServerError)
	a := NewSlackActor(srv.URL, policy.Alert, 0)
	for i := 0; i < 2; i++ {
		if _, err := a.Handle(context.Background(), finding(policy.Alert)); err == nil {
			t.Fatal("a 500 must be an error")
		}
	}
	if srv.count() != 2 {
		t.Fatalf("a failed post must not be recorded as sent, posts %d", srv.count())
	}
}

// The audit reproduced: `slack webhook: Post "http://127.0.0.1:1/services/
// T000/B000/SECRETTOKEN": dial tcp ...`. The path is the secret.
func TestSlackErrorsNeverContainTheWebhookPath(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	ln.Close() // nothing listens here now: connection refused

	for _, hook := range []string{
		"http://" + addr + "/services/T000/B000/SECRETTOKEN",
		"http://[::1:bad/services/T000/B000/SECRETTOKEN", // unparseable
	} {
		a := NewSlackActor(hook, policy.Alert, 0)
		_, err := a.Handle(context.Background(), finding(policy.Alert))
		if err == nil {
			t.Fatalf("%s: expected an error", hook)
		}
		if strings.Contains(err.Error(), "SECRETTOKEN") || strings.Contains(err.Error(), "/services/") {
			t.Fatalf("error leaks the webhook secret: %v", err)
		}
	}
}

func TestOutcomeStrings(t *testing.T) {
	if Noop.String() != "noop" || DryRun.String() != "dry_run" || Acted.String() != "acted" {
		t.Fatal("outcome labels changed; they are metric label values")
	}
}
