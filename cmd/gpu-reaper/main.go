// Command gpu-reaper finds wasted GPU allocations on a Slurm cluster and,
// optionally, does something about them.
//
// Defaults to observe mode. It will not touch the cluster unless told to.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"sort"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/Zhanyl-tech/gpu-reaper/internal/action"
	"github.com/Zhanyl-tech/gpu-reaper/internal/config"
	"github.com/Zhanyl-tech/gpu-reaper/internal/gpu"
	"github.com/Zhanyl-tech/gpu-reaper/internal/metrics"
	"github.com/Zhanyl-tech/gpu-reaper/internal/policy"
	"github.com/Zhanyl-tech/gpu-reaper/internal/slurm"
)

var version = "dev"

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintf(os.Stderr, "gpu-reaper: %v\n", err)
		os.Exit(1)
	}
}

// run is the whole daemon. It takes its arguments and log destination
// explicitly so a test can drive it end to end; main passes os.Args and
// os.Stdout.
func run(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("gpu-reaper", flag.ExitOnError)
	var (
		configPath = fs.String("config", "", "path to config file")
		once       = fs.Bool("once", false, "run a single cycle and exit (same as --cycles 1)")
		cycles     = fs.Int("cycles", 0, "run this many cycles, one interval apart, then exit (0 = run until stopped)")
		nodeName   = fs.String("node-name", "", "this host's Slurm NodeName; overrides gpu.node_name")
		showVer    = fs.Bool("version", false, "print version and exit")
	)
	_ = fs.Parse(args) // ExitOnError: a bad flag exits 2, as flag.Parse did

	if *showVer {
		fmt.Fprintln(stdout, version)
		return nil
	}
	if *once {
		*cycles = 1
	}
	if *cycles < 0 {
		return errors.New("--cycles must not be negative")
	}

	cfg, err := config.Load(*configPath)
	if err != nil {
		return err
	}
	if *nodeName != "" {
		cfg.GPU.NodeName = *nodeName
	}

	logger := newLogger(cfg.LogFormat, stdout)
	enforce := cfg.Mode == "enforce"

	reg := prometheus.NewRegistry()
	m := metrics.New(reg)

	localNode := cfg.GPU.NodeName
	if localNode == "" {
		localNode, _ = os.Hostname()
	}

	gpus, perNode, err := buildGPUSource(cfg, localNode)
	if err != nil {
		return err
	}
	defer gpus.Close()

	jobs, err := buildSlurmSource(cfg, perNode, localNode, m, logger)
	if err != nil {
		return err
	}

	engine, err := newEngine(cfg)
	if err != nil {
		return err
	}

	notifiers := []action.Actor{action.LogActor{Logger: logger}}
	if env := cfg.Slack.WebhookEnv; env != "" {
		if url := os.Getenv(env); url != "" {
			min, _ := config.ParseVerdict(cfg.Slack.MinVerdict)
			notifiers = append(notifiers, action.NewSlackActor(url, min, cfg.Slack.RemindEvery))
		} else {
			logger.Warn("slack configured but env var is empty", "env", env)
		}
	}

	cluster := newClusterActor(cfg, logger)

	scope := "central"
	if perNode {
		scope = "per-node"
	}
	attrs := []any{
		"version", version,
		"mode", cfg.Mode,
		"interval", cfg.Interval.String(),
		"slurm_source", jobs.Name(),
		"gpu_source", gpus.Name(),
		"scope", scope,
		"window", cfg.Thresholds.Window.String(),
		"util_threshold_pct", cfg.Thresholds.UtilPct,
	}
	if perNode {
		// The node name only matters when this daemon judges its own node.
		attrs = append(attrs, "node", localNode)
	}
	logger.Info("starting", attrs...)
	if !enforce {
		logger.Info("observe mode: no jobs will be cancelled and no nodes drained")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	r := &reaper{
		jobs: jobs, gpus: gpus, engine: engine,
		notifiers: notifiers, cluster: cluster,
		metrics: m, logger: logger,
		perNode: perNode, localNode: localNode,
		jobsFiltered: isNodeFiltered(jobs),
		now:          time.Now,
	}

	srv := startMetricsServer(cfg.MetricsAddr, reg, r.ready(3*cfg.Interval), logger)
	defer shutdown(srv, logger)

	ticker := time.NewTicker(cfg.Interval)
	defer ticker.Stop()

	var lastErr error
	for n := 1; ; n++ {
		if lastErr = r.cycle(ctx); lastErr != nil {
			logger.Error("cycle failed", "err", lastErr)
		}
		if *cycles > 0 && n >= *cycles {
			return lastErr
		}
		select {
		case <-ctx.Done():
			logger.Info("shutting down")
			return nil
		case <-ticker.C:
		}
	}
}

type reaper struct {
	jobs      slurm.Source
	gpus      gpu.Source
	engine    *policy.Engine
	notifiers []action.Actor
	cluster   action.Actor
	metrics   *metrics.Metrics
	logger    *slog.Logger

	// perNode is true when the GPU source reads only the local node. The
	// daemon then considers only jobs on localNode, and the evidence rule
	// below means it judges only single-node jobs.
	perNode   bool
	localNode string
	// jobsFiltered is true when the job source itself was scoped to
	// localNode (squeue --nodelist), so any GPU job it returns should be on
	// this node.
	jobsFiltered bool

	now         func() time.Time
	lastSuccess atomic.Int64 // unix nanoseconds
}

func (r *reaper) cycle(ctx context.Context) error {
	start := r.now()
	defer func() { r.metrics.ScrapeDuration.Observe(time.Since(start).Seconds()) }()

	running, err := r.jobs.RunningJobs(ctx)
	if err != nil {
		r.metrics.ScrapeErrors.WithLabelValues(r.jobs.Name()).Inc()
		return fmt.Errorf("list jobs: %w", err)
	}

	active := make(map[string]bool, len(running))
	for _, j := range running {
		active[j.JobID] = true
	}
	r.engine.Forget(active)
	for _, a := range r.notifiers {
		if fg, ok := a.(action.Forgetter); ok {
			fg.Forget(active)
		}
	}

	// A GPU sample identifies a node, not a job. Where a node hosts exactly one
	// GPU job, attribution is unambiguous. Where it hosts several, every job on
	// it would inherit its co-tenants' idle GPUs — and a busy job sharing a node
	// with an idle one would be reported as wasting resources it never held.
	//
	// Rather than guess, skip shared nodes. Correct attribution needs the Slurm
	// cgroup hierarchy to map GPU PIDs back to job IDs, which is deliberately
	// out of scope here; a false alert on a healthy job costs more trust than a
	// missed finding on a shared node costs GPU-hours. Counted over all running
	// jobs, before any per-node filtering, so a co-tenant is never missed.
	gpuJobsPerNode := map[string]int{}
	for _, j := range running {
		if j.GPUCount == 0 {
			continue
		}
		for _, n := range j.Nodes {
			gpuJobsPerNode[n]++
		}
	}

	var inScope []slurm.Job
	nodeSet := map[string]bool{}
	sawLocal := false
	for _, j := range running {
		if j.GPUCount == 0 {
			continue
		}
		if r.perNode && !contains(j.Nodes, r.localNode) {
			continue
		}
		sawLocal = true
		inScope = append(inScope, j)
		for _, n := range j.Nodes {
			nodeSet[n] = true
		}
	}
	if r.jobsFiltered && !sawLocal && len(gpuJobsPerNode) > 0 {
		// squeue --nodelist should only return jobs on this node. If GPU jobs
		// came back but none lists this node, the names disagree (an FQDN, a
		// pod name) and this daemon would silently judge nothing forever.
		r.logger.Warn("GPU jobs returned but none is on this node; is gpu.node_name the Slurm NodeName?",
			"node", r.localNode)
	}
	nodes := make([]string, 0, len(nodeSet))
	for n := range nodeSet {
		nodes = append(nodes, n)
	}
	sort.Strings(nodes)

	samples, err := r.gpus.Sample(ctx, nodes)
	var partial *gpu.PartialError
	switch {
	case errors.As(err, &partial):
		// Jobs on the failed nodes get no evidence this cycle; the rule below
		// skips them and resets their history.
		r.metrics.ScrapeErrors.WithLabelValues(r.gpus.Name()).Add(float64(len(partial.Failed)))
		r.logger.Warn("some nodes not sampled", "err", partial.Error())
	case err != nil:
		r.metrics.ScrapeErrors.WithLabelValues(r.gpus.Name()).Inc()
		// A collection failure must not be read as idleness, so no job is
		// observed this cycle. The engine resets a job's breach clock only
		// when the resulting hole between observations exceeds
		// max_sample_gap. With the defaults (interval 2m, gap 3m) one failed
		// cycle leaves a 4m hole and does reset. Where max_sample_gap is at
		// least twice the interval, a single failed cycle stays inside it and
		// is tolerated by design: the observations either side of it are
		// still contiguous evidence, and nothing escalates on the missing one.
		return fmt.Errorf("sample gpus: %w", err)
	}

	now := r.now()
	byNode := map[string][]gpu.Sample{}
	for _, s := range samples {
		byNode[s.NodeName] = append(byNode[s.NodeName], s)
	}

	var findings []policy.Finding
	withoutSamples := 0
	for _, j := range inScope {
		shared := false
		for _, n := range j.Nodes {
			if gpuJobsPerNode[n] > 1 {
				shared = true
				break
			}
		}
		if shared {
			r.logger.Debug("skipping job on shared node; GPU attribution ambiguous",
				"job_id", j.JobID, "nodes", j.Nodes)
			r.metrics.SkippedShared.Inc()
			r.engine.Drop(j.JobID)
			continue
		}

		// Judge a job only on evidence from every one of its nodes. In a
		// per-node deployment that means only single-node jobs on this node:
		// a multi-node job judged on one node's GPUs could be cancelled while
		// its other nodes are busy.
		var js []gpu.Sample
		var missing []string
		for _, n := range j.Nodes {
			if len(byNode[n]) == 0 {
				missing = append(missing, n)
			}
			js = append(js, byNode[n]...)
		}
		if len(missing) == len(j.Nodes) {
			withoutSamples++
			r.engine.Drop(j.JobID)
			continue
		}
		if len(missing) > 0 {
			r.logger.Debug("skipping job without telemetry for every node",
				"job_id", j.JobID, "missing", missing)
			r.metrics.SkippedPartial.Inc()
			r.engine.Drop(j.JobID)
			continue
		}
		if conflict := attributionConflict(j, js); conflict != "" {
			r.logger.Warn("skipping job: telemetry attributes its GPU to another job",
				"job_id", j.JobID, "detail", conflict)
			r.metrics.SkippedMapping.Inc()
			r.engine.Drop(j.JobID)
			continue
		}

		r.engine.Observe(j, js, now)
		findings = append(findings, r.engine.Evaluate(j, now))
	}

	r.metrics.ObserveCycle(findings, withoutSamples)

	flagged := 0
	for _, f := range findings {
		if f.Verdict >= policy.Alert {
			flagged++
		}
	}
	r.logger.Info("cycle complete",
		"gpu_jobs_in_scope", len(inScope),
		"evaluated", len(findings),
		"without_samples", withoutSamples,
		"at_or_above_alert", flagged)

	for _, f := range findings {
		if f.RecoveredAfterDrain {
			r.logger.Warn("job recovered after its nodes were drained; gpu-reaper does not resume nodes",
				"job_id", f.Job.JobID, "nodes", f.Job.Nodes,
				"hint", "scontrol update NodeName=<node> State=RESUME")
		}
		if f.Verdict < policy.Alert {
			continue
		}
		for _, a := range r.notifiers {
			r.handle(ctx, a, f)
		}
		out, err := r.handle(ctx, r.cluster, f)
		// Cancel is gated on this confirmation, so it is fed back only when
		// the drain actually ran (or, in observe mode, was dry-run). A failed
		// drain is not confirmed, is retried next cycle, and blocks Cancel.
		if err == nil && f.Verdict == policy.Drain && !f.DrainConfirmed && out != action.Noop {
			r.engine.ConfirmDrain(f.Job, now)
		}
	}

	r.lastSuccess.Store(now.UnixNano())
	r.metrics.LastSuccess.Set(float64(now.Unix()))
	return nil
}

// newEngine builds the policy engine from config: thresholds, the effective
// stages (defaults when none are set), exemptions, and the operator's
// allow_cancel_signatures. The last one matters most: without the AllowCancel
// call the engine keeps its built-in default, and an operator who emptied the
// list to turn cancellation off would still have idle jobs cancelled.
func newEngine(cfg *config.Config) (*policy.Engine, error) {
	stages, err := cfg.PolicyStages()
	if err != nil {
		return nil, err
	}
	engine, err := policy.New(cfg.PolicyThresholds(), stages, cfg.Exemptions)
	if err != nil {
		return nil, err
	}
	if err := engine.AllowCancel(cfg.CancelSignatures()...); err != nil {
		return nil, err
	}
	return engine, nil
}

// newClusterActor wires the destructive path. In observe mode the controller
// is NoopController and Enforce is false: two independent gates, so a mistake
// in either one alone cancels nothing.
func newClusterActor(cfg *config.Config, logger *slog.Logger) *action.ClusterActor {
	enforce := cfg.Mode == "enforce"
	var controller slurm.Controller = slurm.NoopController{}
	if enforce {
		controller = slurm.NewCLIController(cfg.Slurm.ScancelPath, cfg.Slurm.ScontrolPath)
	}
	return &action.ClusterActor{Controller: controller, Enforce: enforce, Logger: logger}
}

func isNodeFiltered(s slurm.Source) bool {
	sq, ok := s.(*slurm.SqueueSource)
	return ok && sq.NodeList != ""
}

func (r *reaper) handle(ctx context.Context, a action.Actor, f policy.Finding) (action.Outcome, error) {
	out, err := a.Handle(ctx, f)
	if err != nil {
		r.metrics.ActionErrors.WithLabelValues(a.Name()).Inc()
		r.logger.Error("action failed", "actor", a.Name(), "job_id", f.Job.JobID, "err", err)
		return out, err
	}
	if out != action.Noop {
		r.metrics.ActionsTotal.WithLabelValues(a.Name(), f.Verdict.String(), out.String()).Inc()
	}
	return out, nil
}

// attributionConflict reports when a sample the daemon is about to attribute
// to job j is labelled by the telemetry backend as belonging to other jobs
// only. It can only stop a judgement: samples with no attribution pass.
func attributionConflict(j slurm.Job, samples []gpu.Sample) string {
	for _, s := range samples {
		if len(s.Jobs) > 0 && !contains(s.Jobs, j.JobID) {
			return fmt.Sprintf("%s GPU %d labelled hpc_job=%v", s.NodeName, s.GPUIndex, s.Jobs)
		}
	}
	return ""
}

// ready reports whether a cycle has succeeded within maxAge. /healthz stays a
// liveness check; /readyz is what should page someone, because a daemon whose
// every cycle fails produces no findings and looks exactly like a healthy
// cluster.
func (r *reaper) ready(maxAge time.Duration) func() (bool, string) {
	return func() (bool, string) {
		last := r.lastSuccess.Load()
		if last == 0 {
			return false, "no successful cycle yet"
		}
		age := r.now().Sub(time.Unix(0, last))
		if age > maxAge {
			return false, fmt.Sprintf("last successful cycle %s ago (limit %s)", age.Round(time.Second), maxAge)
		}
		return true, "ok"
	}
}

func contains(xs []string, x string) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}

func buildSlurmSource(cfg *config.Config, perNode bool, localNode string, m *metrics.Metrics, logger *slog.Logger) (slurm.Source, error) {
	reject := func(source string) slurm.RejectFunc {
		return func(reason string) {
			m.SourceRejected.WithLabelValues(source).Inc()
			logger.Warn("rejected job record", "source", source, "reason", reason)
		}
	}
	switch cfg.Slurm.Source {
	case "squeue", "":
		s := slurm.NewSqueueSource(cfg.Slurm.SqueuePath)
		if perNode {
			s.NodeList = localNode
		}
		s.OnReject = reject(s.Name())
		return s, nil
	case "rest":
		if cfg.Slurm.RESTURL == "" {
			return nil, errors.New("slurm.rest_url required when source is rest")
		}
		token := ""
		if cfg.Slurm.TokenEnv != "" {
			token = os.Getenv(cfg.Slurm.TokenEnv)
		}
		s := slurm.NewRESTSource(cfg.Slurm.RESTURL, cfg.Slurm.RESTVer, token, cfg.Slurm.TokenFile, cfg.Slurm.Username)
		s.OnReject = reject(s.Name())
		return s, nil
	}
	return nil, fmt.Errorf("unknown slurm.source %q", cfg.Slurm.Source)
}

// buildGPUSource returns the source and whether it is per-node (reads only
// the local node's GPUs).
func buildGPUSource(cfg *config.Config, node string) (gpu.Source, bool, error) {
	switch cfg.GPU.Source {
	case "nvidia-smi", "":
		s := gpu.NewSMISource(node)
		if cfg.GPU.NvidiaSMIPath != "" {
			// Absolute in enforce mode (config.Validate), so PATH cannot
			// substitute the binary whose output decides "idle".
			s.Binary = cfg.GPU.NvidiaSMIPath
		}
		return s, true, nil
	case "dcgm":
		s := gpu.NewDCGMSource(cfg.GPU.DCGM.URL, node, cfg.GPU.DCGM.Timeout, cfg.GPU.DCGM.MaxAge)
		return s, !s.Central(), nil
	case "simulator":
		return gpu.NewSimSource(node, cfg.GPU.Sim.GPUs,
			gpu.Scenario(cfg.GPU.Sim.Scenario), cfg.GPU.Sim.Seed), false, nil
	}
	return nil, false, fmt.Errorf("unknown gpu.source %q", cfg.GPU.Source)
}

func newLogger(format string, w io.Writer) *slog.Logger {
	if format == "json" {
		return slog.New(slog.NewJSONHandler(w, nil))
	}
	return slog.New(slog.NewTextHandler(w, nil))
}

func startMetricsServer(addr string, reg *prometheus.Registry, ready func() (bool, string), logger *slog.Logger) *http.Server {
	srv := &http.Server{Addr: addr, Handler: newMux(reg, ready), ReadHeaderTimeout: 5 * time.Second}
	go func() {
		logger.Info("metrics listening", "addr", addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("metrics server", "err", err)
		}
	}()
	return srv
}

func newMux(reg *prometheus.Registry, ready func() (bool, string)) *http.ServeMux {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{}))
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, "ok")
	})
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, _ *http.Request) {
		ok, msg := ready()
		if !ok {
			w.WriteHeader(http.StatusServiceUnavailable)
		}
		fmt.Fprintln(w, msg)
	})
	return mux
}

func shutdown(srv *http.Server, logger *slog.Logger) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		logger.Error("metrics shutdown", "err", err)
	}
}
