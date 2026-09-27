package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/Zhanyl-tech/gpu-reaper/internal/action"
	"github.com/Zhanyl-tech/gpu-reaper/internal/config"
	"github.com/Zhanyl-tech/gpu-reaper/internal/gpu"
	"github.com/Zhanyl-tech/gpu-reaper/internal/metrics"
	"github.com/Zhanyl-tech/gpu-reaper/internal/policy"
	"github.com/Zhanyl-tech/gpu-reaper/internal/slurm"
)

// Everything here runs against fakes. No Slurm controller, no GPU.

var t0 = time.Date(2026, 9, 26, 12, 0, 0, 0, time.UTC)

type clock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *clock) Now() time.Time      { c.mu.Lock(); defer c.mu.Unlock(); return c.now }
func (c *clock) Add(d time.Duration) { c.mu.Lock(); c.now = c.now.Add(d); c.mu.Unlock() }

type fakeJobs struct {
	jobs []slurm.Job
	err  error
}

func (f *fakeJobs) Name() string { return "fake-slurm" }
func (f *fakeJobs) RunningJobs(context.Context) ([]slurm.Job, error) {
	return f.jobs, f.err
}

// fakeGPUs reports every GPU idle. perNode mimics nvidia-smi: only local.
type fakeGPUs struct {
	clk     *clock
	perNode string
	gpus    int
	util    float64
	fail    map[string]bool // central: nodes that fail
	failAll bool
	jobs    []string // hpc_job label to attach
	asked   [][]string
}

func (f *fakeGPUs) Name() string { return "fake-gpu" }
func (f *fakeGPUs) Close() error { return nil }
func (f *fakeGPUs) Sample(_ context.Context, nodes []string) ([]gpu.Sample, error) {
	f.asked = append(f.asked, nodes)
	if f.failAll {
		return nil, errors.New("collector down")
	}
	targets := nodes
	if f.perNode != "" {
		targets = []string{f.perNode}
	}
	now := f.clk.Now()
	var out []gpu.Sample
	failed := map[string]error{}
	for _, n := range targets {
		if f.fail[n] {
			failed[n] = errors.New("unreachable")
			continue
		}
		for i := 0; i < f.gpus; i++ {
			out = append(out, gpu.Sample{
				Timestamp: now, NodeName: n, GPUIndex: i,
				SMUtilPct: f.util, MemUsedBytes: 0, MemTotalBytes: 80 << 30, PowerWatts: 25,
				UtilKnown: true, MemKnown: true, PowerKnown: true, PIDsKnown: true,
				Jobs: f.jobs,
			})
		}
	}
	if len(failed) > 0 {
		return out, &gpu.PartialError{Failed: failed}
	}
	return out, nil
}

type recordingController struct {
	mu       sync.Mutex
	calls    []string
	drainErr error
}

func (r *recordingController) Name() string { return "recording" }
func (r *recordingController) Cancel(_ context.Context, id, _ string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls = append(r.calls, "cancel "+id)
	return nil
}
func (r *recordingController) Drain(_ context.Context, node, _ string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls = append(r.calls, "drain "+node)
	return r.drainErr
}

type harness struct {
	r    *reaper
	clk  *clock
	ctrl *recordingController
	reg  *prometheus.Registry
	m    *metrics.Metrics
	log  *bytes.Buffer
}

// newHarness builds a reaper with an aggressive engine ({0: cancel}, idle
// cancellable) so any path to a destructive call is exercised.
func newHarness(t *testing.T, jobs slurm.Source, gpus *fakeGPUs, enforce, perNode bool, local string) *harness {
	t.Helper()
	clk := &clock{now: t0}
	gpus.clk = clk
	eng, err := policy.New(policy.DefaultThresholds(), []policy.Stage{{After: 0, Verdict: policy.Cancel}}, policy.Exemptions{})
	if err != nil {
		t.Fatal(err)
	}
	reg := prometheus.NewRegistry()
	m := metrics.New(reg)
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, nil))
	ctrl := &recordingController{}
	r := &reaper{
		jobs: jobs, gpus: gpus, engine: eng,
		notifiers: []action.Actor{action.LogActor{Logger: logger}},
		cluster:   &action.ClusterActor{Controller: ctrl, Enforce: enforce, Logger: logger},
		metrics:   m, logger: logger,
		perNode: perNode, localNode: local,
		now: clk.Now,
	}
	return &harness{r: r, clk: clk, ctrl: ctrl, reg: reg, m: m, log: &buf}
}

func (h *harness) runFor(t *testing.T, d time.Duration) {
	t.Helper()
	end := h.clk.Now().Add(d)
	for h.clk.Now().Before(end) {
		_ = h.r.cycle(context.Background())
		h.clk.Add(2 * time.Minute)
	}
}

func gpuJob(id string, gpus int, nodes ...string) slurm.Job {
	return slurm.Job{JobID: id, User: "alice", Partition: "gpu", State: "RUNNING",
		StartTime: t0.Add(-6 * time.Hour), Nodes: nodes, GPUCount: gpus}
}

// The first of the two gates: observe mode must wire a controller that cannot
// change the cluster, whatever the actor does.
func TestObserveModeWiresNoopController(t *testing.T) {
	cfg := config.Default()
	a := newClusterActor(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if _, ok := a.Controller.(slurm.NoopController); !ok || a.Enforce {
		t.Fatalf("observe mode wired %T enforce=%t", a.Controller, a.Enforce)
	}
	cfg.Mode = "enforce"
	cfg.Slurm.ScancelPath, cfg.Slurm.ScontrolPath = "/usr/bin/scancel", "/usr/bin/scontrol"
	a = newClusterActor(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if _, ok := a.Controller.(*slurm.CLIController); !ok || !a.Enforce {
		t.Fatalf("enforce mode wired %T enforce=%t", a.Controller, a.Enforce)
	}
}

// The audit's critical per-node finding: the daemon on gpu001 drained gpu001
// AND gpu002 and cancelled a 2-node job it had only half observed.
func TestClaim_PerNodeDaemonNeverActsOnAMultiNodeJob(t *testing.T) {
	jobs := &fakeJobs{jobs: []slurm.Job{gpuJob("42", 16, "gpu001", "gpu002")}}
	h := newHarness(t, jobs, &fakeGPUs{perNode: "gpu001", gpus: 8}, true, true, "gpu001")
	h.runFor(t, 4*time.Hour)

	if len(h.ctrl.calls) != 0 {
		t.Fatalf("per-node daemon acted on a job it only half observed: %v", h.ctrl.calls)
	}
	if got := testutil.ToFloat64(h.m.SkippedPartial); got == 0 {
		t.Fatal("the skip should be counted in gpu_reaper_skipped_partial_evidence_total")
	}
}

func TestPerNodeDaemonActsOnlyOnItsSingleNodeJob(t *testing.T) {
	jobs := &fakeJobs{jobs: []slurm.Job{
		gpuJob("41", 8, "gpu001"),
		gpuJob("43", 8, "gpu002"), // another node's job: out of scope
	}}
	h := newHarness(t, jobs, &fakeGPUs{perNode: "gpu001", gpus: 8}, true, true, "gpu001")
	h.runFor(t, 2*time.Hour)
	for _, c := range h.ctrl.calls {
		if c != "drain gpu001" && c != "cancel 41" {
			t.Fatalf("per-node daemon touched something not on its node: %v", h.ctrl.calls)
		}
	}
	if !contains(h.ctrl.calls, "drain gpu001") {
		t.Fatalf("the local idle job should have been drained: %v", h.ctrl.calls)
	}
	if n := strings.Count(strings.Join(h.ctrl.calls, "|"), "drain gpu001"); n != 1 {
		t.Fatalf("drain should be issued once, not every cycle: %d times (%v)", n, h.ctrl.calls)
	}
}

func TestCentralModeJudgesMultiNodeJobsOnFullEvidence(t *testing.T) {
	jobs := &fakeJobs{jobs: []slurm.Job{gpuJob("42", 16, "gpu001", "gpu002")}}
	h := newHarness(t, jobs, &fakeGPUs{gpus: 8}, true, false, "")
	h.runFor(t, 2*time.Hour)
	if !contains(h.ctrl.calls, "drain gpu001") || !contains(h.ctrl.calls, "drain gpu002") {
		t.Fatalf("central mode should drain both nodes once judged: %v", h.ctrl.calls)
	}
}

func TestObserveModeNeverCallsController(t *testing.T) {
	jobs := &fakeJobs{jobs: []slurm.Job{gpuJob("41", 8, "gpu001")}}
	h := newHarness(t, jobs, &fakeGPUs{gpus: 8}, false, false, "")
	h.runFor(t, 3*time.Hour)
	if len(h.ctrl.calls) != 0 {
		t.Fatalf("observe mode called the controller: %v", h.ctrl.calls)
	}
	if !strings.Contains(h.log.String(), "would cancel") {
		t.Fatal("observe mode should still show the whole ladder in the log")
	}
}

// actions_total used to hard-code enforced="true", so observe mode reported
// enforcement, and counted the cluster actor on alerts where it does nothing.
func TestActionsMetricReportsWhatActuallyHappened(t *testing.T) {
	jobs := &fakeJobs{jobs: []slurm.Job{gpuJob("41", 8, "gpu001")}}
	h := newHarness(t, jobs, &fakeGPUs{gpus: 8}, false, false, "")
	h.runFor(t, 3*time.Hour)

	if n := testutil.ToFloat64(h.m.ActionsTotal.WithLabelValues("cluster(dry-run)", "drain", "dry_run")); n != 1 {
		t.Fatalf("dry-run drain count = %v, want 1", n)
	}
	for _, v := range []string{"alert", "drain", "cancel"} {
		if n := testutil.ToFloat64(h.m.ActionsTotal.WithLabelValues("cluster(dry-run)", v, "acted")); n != 0 {
			t.Fatalf("observe mode reported an acted %s", v)
		}
	}
	if n := testutil.ToFloat64(h.m.ActionsTotal.WithLabelValues("cluster(dry-run)", "alert", "dry_run")); n != 0 {
		t.Fatalf("the cluster actor does nothing on alert and must not be counted, got %v", n)
	}
	for _, v := range []string{"alert", "drain", "cancel"} {
		if n := testutil.ToFloat64(h.m.ActionsTotal.WithLabelValues("cluster(dry-run)", v, "noop")); n != 0 {
			t.Fatalf("no-ops must not be counted (%s: %v)", v, n)
		}
	}
}

func TestFailedDrainIsRetriedAndBlocksCancel(t *testing.T) {
	jobs := &fakeJobs{jobs: []slurm.Job{gpuJob("41", 8, "gpu001")}}
	h := newHarness(t, jobs, &fakeGPUs{gpus: 8}, true, false, "")
	h.ctrl.drainErr = errors.New("permission denied")
	h.runFor(t, 3*time.Hour)
	if contains(h.ctrl.calls, "cancel 41") {
		t.Fatalf("cancel followed a drain that never succeeded: %v", h.ctrl.calls)
	}
	if strings.Count(strings.Join(h.ctrl.calls, "|"), "drain gpu001") < 2 {
		t.Fatalf("a failed drain should be retried: %v", h.ctrl.calls)
	}
	if testutil.ToFloat64(h.m.ActionErrors.WithLabelValues("cluster(enforce)")) == 0 {
		t.Fatal("drain failures should be counted")
	}
}

func TestCollectorFailureNeverEscalates(t *testing.T) {
	jobs := &fakeJobs{jobs: []slurm.Job{gpuJob("41", 8, "gpu001")}}
	g := &fakeGPUs{gpus: 8}
	h := newHarness(t, jobs, g, true, false, "")
	h.runFor(t, 30*time.Minute) // alerted, maybe drained
	g.failAll = true
	h.runFor(t, 40*time.Minute) // collector down longer than the window
	g.failAll = false
	before := len(h.ctrl.calls)
	h.runFor(t, 16*time.Minute) // less than a window of fresh evidence
	if len(h.ctrl.calls) != before {
		t.Fatalf("acted within a window of the collector coming back: %v", h.ctrl.calls[before:])
	}
	if testutil.ToFloat64(h.m.ScrapeErrors.WithLabelValues("fake-gpu")) == 0 {
		t.Fatal("collector failures should be counted")
	}
}

func TestPartialScrapeSkipsOnlyAffectedJobs(t *testing.T) {
	jobs := &fakeJobs{jobs: []slurm.Job{gpuJob("41", 8, "gpu001"), gpuJob("42", 8, "gpu002")}}
	g := &fakeGPUs{gpus: 8, fail: map[string]bool{"gpu002": true}}
	h := newHarness(t, jobs, g, true, false, "")
	h.runFor(t, 2*time.Hour)
	for _, c := range h.ctrl.calls {
		if strings.Contains(c, "gpu002") || strings.Contains(c, "42") {
			t.Fatalf("acted on a job whose node could not be sampled: %v", h.ctrl.calls)
		}
	}
	if !contains(h.ctrl.calls, "drain gpu001") {
		t.Fatalf("the healthy-telemetry node's job should still be judged: %v", h.ctrl.calls)
	}
	if got := testutil.ToFloat64(h.m.JobsNoSamples); got != 1 {
		t.Fatalf("gpu_jobs_without_samples = %v, want 1", got)
	}
}

func TestSharedNodeJobsAreSkipped(t *testing.T) {
	jobs := &fakeJobs{jobs: []slurm.Job{gpuJob("41", 4, "gpu001"), gpuJob("42", 4, "gpu001")}}
	h := newHarness(t, jobs, &fakeGPUs{gpus: 8}, true, false, "")
	h.runFor(t, 2*time.Hour)
	if len(h.ctrl.calls) != 0 {
		t.Fatalf("shared-node jobs must not be acted on: %v", h.ctrl.calls)
	}
	if testutil.ToFloat64(h.m.SkippedShared) == 0 {
		t.Fatal("skips should be counted")
	}
}

func TestAttributionConflictSkipsTheJob(t *testing.T) {
	jobs := &fakeJobs{jobs: []slurm.Job{gpuJob("41", 8, "gpu001")}}
	h := newHarness(t, jobs, &fakeGPUs{gpus: 8, jobs: []string{"999"}}, true, false, "")
	h.runFor(t, 2*time.Hour)
	if len(h.ctrl.calls) != 0 {
		t.Fatalf("acted although telemetry attributes the GPUs to job 999: %v", h.ctrl.calls)
	}
	if testutil.ToFloat64(h.m.SkippedMapping) == 0 {
		t.Fatal("conflicts should be counted")
	}
}

func TestMatchingAttributionIsAllowed(t *testing.T) {
	jobs := &fakeJobs{jobs: []slurm.Job{gpuJob("41", 8, "gpu001")}}
	h := newHarness(t, jobs, &fakeGPUs{gpus: 8, jobs: []string{"41"}}, true, false, "")
	h.runFor(t, 2*time.Hour)
	if !contains(h.ctrl.calls, "drain gpu001") {
		t.Fatalf("a matching hpc_job label must not block judgement: %v", h.ctrl.calls)
	}
}

func TestOnlyGPUJobNodesAreSampled(t *testing.T) {
	cpu := gpuJob("40", 0, "cpu001")
	jobs := &fakeJobs{jobs: []slurm.Job{cpu, gpuJob("41", 8, "gpu001")}}
	g := &fakeGPUs{gpus: 8}
	h := newHarness(t, jobs, g, false, false, "")
	_ = h.r.cycle(context.Background())
	if len(g.asked) != 1 || strings.Join(g.asked[0], ",") != "gpu001" {
		t.Fatalf("central source should be asked only for GPU jobs' nodes, got %v", g.asked)
	}
}

func TestNodeNameMismatchIsLoud(t *testing.T) {
	jobs := &fakeJobs{jobs: []slurm.Job{gpuJob("41", 8, "gpu001")}}
	h := newHarness(t, jobs, &fakeGPUs{perNode: "gpu001.cluster", gpus: 8}, false, true, "gpu001.cluster")
	h.r.jobsFiltered = true
	_ = h.r.cycle(context.Background())
	if !strings.Contains(h.log.String(), "is gpu.node_name the Slurm NodeName") {
		t.Fatalf("mismatch should be logged, log:\n%s", h.log.String())
	}
	// What the README says about per-node mode: the mismatched job is out of
	// scope before gpu_jobs_without_samples is counted, so that gauge stays at
	// 0 and cannot reveal the mismatch; jobs_evaluated stays at 0 too.
	if got := testutil.ToFloat64(h.m.JobsNoSamples); got != 0 {
		t.Fatalf("gpu_jobs_without_samples = %v; the README says per-node mode cannot raise it on a mismatch", got)
	}
	if got := testutil.ToFloat64(h.m.JobsEvaluated); got != 0 {
		t.Fatalf("jobs_evaluated = %v, want 0", got)
	}
}

func TestListJobsFailureIsAnError(t *testing.T) {
	h := newHarness(t, &fakeJobs{err: errors.New("slurmctld down")}, &fakeGPUs{gpus: 8}, false, false, "")
	if err := h.r.cycle(context.Background()); err == nil {
		t.Fatal("want error")
	}
	if testutil.ToFloat64(h.m.ScrapeErrors.WithLabelValues("fake-slurm")) != 1 {
		t.Fatal("job listing failure should be counted")
	}
}

func TestReadyzReflectsLastSuccessfulCycle(t *testing.T) {
	jobs := &fakeJobs{jobs: []slurm.Job{gpuJob("41", 8, "gpu001")}}
	g := &fakeGPUs{gpus: 8}
	h := newHarness(t, jobs, g, false, false, "")
	srv := httptest.NewServer(newMux(h.reg, h.r.ready(6*time.Minute)))
	defer srv.Close()
	status := func(path string) int {
		resp, err := http.Get(srv.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}

	if status("/readyz") != http.StatusServiceUnavailable {
		t.Fatal("not ready before the first successful cycle")
	}
	_ = h.r.cycle(context.Background())
	if status("/readyz") != http.StatusOK {
		t.Fatal("ready after a successful cycle")
	}
	g.failAll = true
	h.clk.Add(10 * time.Minute)
	_ = h.r.cycle(context.Background())
	if status("/readyz") != http.StatusServiceUnavailable {
		t.Fatal("not ready when every recent cycle failed")
	}
	if status("/healthz") != http.StatusOK {
		t.Fatal("liveness stays 200")
	}
	if testutil.ToFloat64(h.m.LastSuccess) != float64(t0.Unix()) {
		t.Fatal("last success timestamp should not move on a failed cycle")
	}
}

// ── Wiring from config to the daemon. newHarness builds the reaper directly,
// so these pin what run() builds from a config file. ─────────────────────────

func discardLogger() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

func loadConfig(t *testing.T, yaml string) *config.Config {
	t.Helper()
	p := filepath.Join(t.TempDir(), "c.yaml")
	if err := os.WriteFile(p, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Load(p)
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestBuildGPUSourceScope(t *testing.T) {
	cases := []struct {
		name, yaml  string
		wantPerNode bool
	}{
		{"nvidia-smi reads only local GPUs", "gpu: {source: nvidia-smi}\n", true},
		{"dcgm with a fixed URL is per node", "gpu: {source: dcgm, dcgm: {url: 'http://localhost:9400/metrics'}}\n", true},
		{"dcgm with {node} is central", "gpu: {source: dcgm, dcgm: {url: 'http://{node}:9400/metrics'}}\n", false},
		{"the simulator stands in for the fleet", "gpu: {source: simulator}\n", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			src, perNode, err := buildGPUSource(loadConfig(t, tc.yaml), "gpu001")
			if err != nil {
				t.Fatal(err)
			}
			if perNode != tc.wantPerNode {
				t.Fatalf("%s: perNode = %t, want %t", src.Name(), perNode, tc.wantPerNode)
			}
		})
	}
}

// gpu.nvidia_smi_path is the binary that decides "idle"; it must reach the
// source, and the local node name must label its samples.
func TestBuildGPUSourceUsesConfiguredNvidiaSMI(t *testing.T) {
	src, _, err := buildGPUSource(loadConfig(t, "gpu: {nvidia_smi_path: /opt/nvidia/bin/nvidia-smi}\n"), "gpu001")
	if err != nil {
		t.Fatal(err)
	}
	smi, ok := src.(*gpu.SMISource)
	if !ok || smi.Binary != "/opt/nvidia/bin/nvidia-smi" || smi.NodeName != "gpu001" {
		t.Fatalf("got %T %+v", src, src)
	}
	src, _, _ = buildGPUSource(config.Default(), "gpu001")
	if smi := src.(*gpu.SMISource); smi.Binary != "nvidia-smi" {
		t.Fatalf("default binary = %q", smi.Binary)
	}
	cfg := config.Default()
	cfg.GPU.Source = "nvml"
	if _, _, err := buildGPUSource(cfg, "gpu001"); err == nil {
		t.Fatal("unknown gpu.source should be an error")
	}
}

// A per-node daemon asks squeue only for its own node, and only then is the
// node-name mismatch warning armed.
func TestBuildSlurmSourceScopesPerNodeSqueue(t *testing.T) {
	m := metrics.New(prometheus.NewRegistry())
	cfg := loadConfig(t, "slurm: {squeue_path: /opt/slurm/bin/squeue}\n")

	src, err := buildSlurmSource(cfg, true, "gpu001", m, discardLogger())
	if err != nil {
		t.Fatal(err)
	}
	sq, ok := src.(*slurm.SqueueSource)
	if !ok || sq.NodeList != "gpu001" || sq.Binary != "/opt/slurm/bin/squeue" {
		t.Fatalf("per-node squeue = %T %+v", src, src)
	}
	if !isNodeFiltered(src) {
		t.Fatal("a per-node squeue source is node-filtered")
	}

	// Rejected records are counted under the source's name.
	sq.OnReject("malformed line")
	if got := testutil.ToFloat64(m.SourceRejected.WithLabelValues("squeue")); got != 1 {
		t.Fatalf("source_rejected_records_total{source=squeue} = %v, want 1", got)
	}

	src, err = buildSlurmSource(cfg, false, "gpu001", m, discardLogger())
	if err != nil {
		t.Fatal(err)
	}
	if sq := src.(*slurm.SqueueSource); sq.NodeList != "" || isNodeFiltered(src) {
		t.Fatalf("a central daemon must see every node's jobs: NodeList=%q", sq.NodeList)
	}
}

func TestBuildSlurmSourceREST(t *testing.T) {
	m := metrics.New(prometheus.NewRegistry())
	cfg := loadConfig(t, "slurm: {source: rest, rest_url: 'http://slurmrestd:6820/', username: slurm, token_file: /run/secrets/slurm-jwt}\n")
	src, err := buildSlurmSource(cfg, true, "gpu001", m, discardLogger())
	if err != nil {
		t.Fatal(err)
	}
	rs, ok := src.(*slurm.RESTSource)
	if !ok || rs.BaseURL != "http://slurmrestd:6820" || rs.TokenFile != "/run/secrets/slurm-jwt" || rs.Username != "slurm" || rs.Token != "" {
		t.Fatalf("rest source = %T %+v", src, src)
	}
	if isNodeFiltered(src) {
		t.Fatal("slurmrestd returns every job; it is not node-filtered")
	}

	t.Setenv("GPU_REAPER_TEST_JWT", "jwt-from-env")
	cfg = loadConfig(t, "slurm: {source: rest, rest_url: 'http://slurmrestd:6820', token_env: GPU_REAPER_TEST_JWT}\n")
	src, err = buildSlurmSource(cfg, false, "", m, discardLogger())
	if err != nil {
		t.Fatal(err)
	}
	if rs := src.(*slurm.RESTSource); rs.Token != "jwt-from-env" || rs.TokenFile != "" {
		t.Fatalf("token_env not read: %+v", rs)
	}

	cfg.Slurm.RESTURL = ""
	if _, err := buildSlurmSource(cfg, false, "", m, discardLogger()); err == nil {
		t.Fatal("rest without rest_url should be an error")
	}
	cfg.Slurm.Source = "sacct"
	if _, err := buildSlurmSource(cfg, false, "", m, discardLogger()); err == nil {
		t.Fatal("unknown slurm.source should be an error")
	}
}

// run() builds its engine with newEngine. allow_cancel_signatures and the
// stages must reach it: without the AllowCancel call the engine keeps its
// built-in [idle], so an operator who emptied the list would still have idle
// jobs cancelled. The fake GPUs report every GPU idle.
func TestNewEngineAppliesConfiguredStagesAndCancelSignatures(t *testing.T) {
	const stages = "stages: [{after: 0s, verdict: alert}, {after: 20m, verdict: drain}, {after: 40m, verdict: cancel}]\n"
	cases := []struct {
		allow      string
		wantCancel bool
	}{
		{"[idle]", true},
		{"[]", false},     // cancellation turned off
		{"[hung]", false}, // these jobs are idle, not hung
	}
	for _, tc := range cases {
		t.Run(tc.allow, func(t *testing.T) {
			eng, err := newEngine(loadConfig(t, stages+"allow_cancel_signatures: "+tc.allow+"\n"))
			if err != nil {
				t.Fatal(err)
			}
			jobs := &fakeJobs{jobs: []slurm.Job{gpuJob("41", 8, "gpu001")}}
			h := newHarness(t, jobs, &fakeGPUs{gpus: 8}, true, false, "")
			h.r.engine = eng
			h.runFor(t, 3*time.Hour)
			if !contains(h.ctrl.calls, "drain gpu001") {
				t.Fatalf("the configured drain stage should have run: %v", h.ctrl.calls)
			}
			if got := contains(h.ctrl.calls, "cancel 41"); got != tc.wantCancel {
				t.Fatalf("allow_cancel_signatures %s: cancelled = %t, want %t (%v)", tc.allow, got, tc.wantCancel, h.ctrl.calls)
			}
		})
	}
}

// Drain is issued once per breach episode, not once per job incarnation: when
// the breach clock resets (here, one busy cycle) and the job climbs back to
// Drain, the drain is issued again for the same JobID and StartTime.
func TestDrainIsReissuedAfterTheBreachClockResets(t *testing.T) {
	jobs := &fakeJobs{jobs: []slurm.Job{gpuJob("41", 8, "gpu001")}}
	g := &fakeGPUs{gpus: 8}
	h := newHarness(t, jobs, g, true, false, "")
	drains := func() int { return strings.Count(strings.Join(h.ctrl.calls, "|"), "drain gpu001") }

	h.runFor(t, 24*time.Minute)
	if drains() != 1 {
		t.Fatalf("setup: want one drain, got %v", h.ctrl.calls)
	}
	g.util = 100 // one busy cycle: peak rule resets the breach
	h.runFor(t, 2*time.Minute)
	g.util = 0
	h.runFor(t, 40*time.Minute) // the busy sample ages out, the ladder is climbed again
	if drains() != 2 {
		t.Fatalf("want the drain re-issued after the reset, got %v", h.ctrl.calls)
	}
	if contains(h.ctrl.calls, "cancel 41") {
		t.Fatalf("cancel must wait a window after the new drain: %v", h.ctrl.calls)
	}
}

// ── run(), end to end, against the demo's fake squeue and the simulator. ────

// syncBuffer is a log sink the metrics goroutine and the cycle loop can both
// write to while the test reads it.
type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

func writeConfig(t *testing.T, yaml string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "c.yaml")
	if err := os.WriteFile(p, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestRunPrintsVersion(t *testing.T) {
	var out syncBuffer
	if err := run([]string{"--version"}, &out); err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(out.String()) != version {
		t.Fatalf("got %q", out.String())
	}
}

func TestRunRejectsBadInvocations(t *testing.T) {
	var out syncBuffer
	if err := run([]string{"--cycles", "-1"}, &out); err == nil {
		t.Fatal("negative --cycles should be an error")
	}
	// Validation runs before anything is built: enforce mode with a bare
	// squeue must not start.
	p := writeConfig(t, "mode: enforce\nslurm: {scancel_path: /usr/bin/scancel, scontrol_path: /usr/bin/scontrol}\n")
	if err := run([]string{"--config", p, "--once"}, &out); err == nil || !strings.Contains(err.Error(), "squeue_path") {
		t.Fatalf("want the relative squeue_path rejected, got %v", err)
	}
}

// A bounded run through run() itself: config load, source wiring, engine,
// observe-mode cluster actor, metrics server and the cycle loop. Verdicts are
// covered by the harness tests; this pins that run() puts the pieces together.
func TestRunBoundedCyclesEndToEnd(t *testing.T) {
	if _, err := exec.LookPath("bash"); err != nil {
		t.Skip("the demo's fake squeue needs bash")
	}
	squeue, err := filepath.Abs(filepath.Join("..", "..", "demo", "bin", "squeue"))
	if err != nil {
		t.Fatal(err)
	}
	p := writeConfig(t, "mode: observe\ninterval: 50ms\nmetrics_addr: \"127.0.0.1:0\"\n"+
		"slurm: {squeue_path: '"+squeue+"'}\n"+
		"gpu: {source: simulator, simulator: {scenario: idle}}\n"+
		"thresholds: {window: 2s, max_sample_gap: 500ms, min_samples: 3, warmup: 0s}\n"+
		"stages: [{after: 0s, verdict: alert}, {after: 2s, verdict: drain}]\n")
	var out syncBuffer
	if err := run([]string{"--config", p, "--cycles", "2"}, &out); err != nil {
		t.Fatalf("run: %v\n%s", err, out.String())
	}
	log := out.String()
	for _, want := range []string{
		"msg=starting", "mode=observe", "slurm_source=squeue", "gpu_source=simulator/idle", "scope=central",
		"observe mode: no jobs will be cancelled",
		// Three GPU jobs in the fake squeue (the CPU-only one is not).
		`msg="cycle complete" gpu_jobs_in_scope=3`,
	} {
		if !strings.Contains(log, want) {
			t.Errorf("log lacks %q", want)
		}
	}
	if n := strings.Count(log, `msg="cycle complete"`); n != 2 {
		t.Errorf("want 2 cycles, got %d", n)
	}
	for _, bad := range []string{"level=ERROR", "rejected job record"} {
		if strings.Contains(log, bad) {
			t.Errorf("log contains %q", bad)
		}
	}
	if t.Failed() {
		t.Logf("log:\n%s", log)
	}
}
