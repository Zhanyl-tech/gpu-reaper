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
	"log/slog"
	"net/http"
	"os"
	"os/signal"
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
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "gpu-reaper: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		configPath = flag.String("config", "", "path to config file")
		once       = flag.Bool("once", false, "run a single cycle and exit")
		showVer    = flag.Bool("version", false, "print version and exit")
	)
	flag.Parse()

	if *showVer {
		fmt.Println(version)
		return nil
	}

	cfg, err := config.Load(*configPath)
	if err != nil {
		return err
	}

	logger := newLogger(cfg.LogFormat)
	enforce := cfg.Mode == "enforce"

	jobs, err := buildSlurmSource(cfg)
	if err != nil {
		return err
	}
	gpus, err := buildGPUSource(cfg)
	if err != nil {
		return err
	}
	defer gpus.Close()

	stages, err := cfg.PolicyStages()
	if err != nil {
		return err
	}
	engine, err := policy.New(cfg.PolicyThresholds(), stages, cfg.Exemptions)
	if err != nil {
		return err
	}

	reg := prometheus.NewRegistry()
	m := metrics.New(reg)

	actors := []action.Actor{action.LogActor{Logger: logger}}

	var controller slurm.Controller = slurm.NoopController{}
	if enforce {
		controller = slurm.NewCLIController()
	}
	actors = append(actors, &action.ClusterActor{
		Controller: controller, Enforce: enforce, Logger: logger,
	})

	if env := cfg.Slack.WebhookEnv; env != "" {
		if url := os.Getenv(env); url != "" {
			min, _ := config.ParseVerdict(cfg.Slack.MinVerdict)
			actors = append(actors, action.NewSlackActor(url, min))
		} else {
			logger.Warn("slack configured but env var is empty", "env", env)
		}
	}

	logger.Info("starting",
		"version", version,
		"mode", cfg.Mode,
		"interval", cfg.Interval.String(),
		"slurm_source", jobs.Name(),
		"gpu_source", gpus.Name(),
		"window", cfg.Thresholds.Window.String(),
		"util_threshold_pct", cfg.Thresholds.UtilPct,
	)
	if !enforce {
		logger.Info("observe mode: no jobs will be cancelled and no nodes drained")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	srv := startMetricsServer(cfg.MetricsAddr, reg, logger)
	defer shutdown(srv, logger)

	r := &reaper{
		jobs: jobs, gpus: gpus, engine: engine,
		actors: actors, metrics: m, logger: logger,
	}

	if *once {
		return r.cycle(ctx)
	}

	ticker := time.NewTicker(cfg.Interval)
	defer ticker.Stop()

	if err := r.cycle(ctx); err != nil {
		logger.Error("cycle failed", "err", err)
	}
	for {
		select {
		case <-ctx.Done():
			logger.Info("shutting down")
			return nil
		case <-ticker.C:
			if err := r.cycle(ctx); err != nil {
				logger.Error("cycle failed", "err", err)
			}
		}
	}
}

type reaper struct {
	jobs    slurm.Source
	gpus    gpu.Source
	engine  *policy.Engine
	actors  []action.Actor
	metrics *metrics.Metrics
	logger  *slog.Logger
}

func (r *reaper) cycle(ctx context.Context) error {
	start := time.Now()
	defer func() { r.metrics.ScrapeDuration.Observe(time.Since(start).Seconds()) }()

	running, err := r.jobs.RunningJobs(ctx)
	if err != nil {
		r.metrics.ScrapeErrors.WithLabelValues(r.jobs.Name()).Inc()
		return fmt.Errorf("list jobs: %w", err)
	}

	active := make(map[string]bool, len(running))
	var nodes []string
	for _, j := range running {
		active[j.JobID] = true
		nodes = append(nodes, j.Nodes...)
	}
	r.engine.Forget(active)

	samples, err := r.gpus.Sample(ctx, nodes)
	if err != nil {
		r.metrics.ScrapeErrors.WithLabelValues(r.gpus.Name()).Inc()
		// A collection failure must not be read as idleness. Returning here
		// leaves the window with a gap, which the policy engine detects and
		// refuses to escalate on.
		return fmt.Errorf("sample gpus: %w", err)
	}

	now := time.Now()
	byNode := map[string][]gpu.Sample{}
	for _, s := range samples {
		byNode[s.NodeName] = append(byNode[s.NodeName], s)
	}

	// A GPU sample identifies a node, not a job. Where a node hosts exactly one
	// GPU job, attribution is unambiguous. Where it hosts several, every job on
	// it would inherit its co-tenants' idle GPUs — and a busy job sharing a node
	// with an idle one would be reported as wasting resources it never held.
	//
	// Rather than guess, skip shared nodes. Correct attribution needs the Slurm
	// cgroup hierarchy to map GPU PIDs back to job IDs, which is deliberately
	// out of scope here; a false alert on a healthy job costs more trust than a
	// missed finding on a shared node costs GPU-hours.
	gpuJobsPerNode := map[string]int{}
	for _, j := range running {
		if j.GPUCount == 0 {
			continue
		}
		for _, n := range j.Nodes {
			gpuJobsPerNode[n]++
		}
	}

	var findings []policy.Finding
	for _, j := range running {
		if j.GPUCount == 0 {
			continue
		}

		var js []gpu.Sample
		shared := false
		for _, n := range j.Nodes {
			if gpuJobsPerNode[n] > 1 {
				shared = true
				break
			}
			js = append(js, byNode[n]...)
		}
		if shared {
			r.logger.Debug("skipping job on shared node; GPU attribution ambiguous",
				"job_id", j.JobID, "nodes", j.Nodes)
			r.metrics.SkippedShared.Inc()
			continue
		}

		r.engine.Observe(j.JobID, js, now)
		findings = append(findings, r.engine.Evaluate(j, now))
	}

	r.metrics.ObserveCycle(findings)

	for _, f := range findings {
		if f.Verdict < policy.Alert {
			continue
		}
		for _, a := range r.actors {
			if err := a.Handle(ctx, f); err != nil {
				r.metrics.ActionErrors.WithLabelValues(a.Name()).Inc()
				r.logger.Error("action failed", "actor", a.Name(), "job_id", f.Job.JobID, "err", err)
				continue
			}
			r.metrics.ActionsTotal.WithLabelValues(a.Name(), f.Verdict.String(), "true").Inc()
		}
	}
	return nil
}

func buildSlurmSource(cfg *config.Config) (slurm.Source, error) {
	switch cfg.Slurm.Source {
	case "squeue", "":
		return slurm.NewSqueueSource(), nil
	case "rest":
		if cfg.Slurm.RESTURL == "" {
			return nil, errors.New("slurm.rest_url required when source is rest")
		}
		return slurm.NewRESTSource(
			cfg.Slurm.RESTURL, cfg.Slurm.RESTVer,
			os.Getenv(cfg.Slurm.TokenEnv), cfg.Slurm.Username,
		), nil
	}
	return nil, fmt.Errorf("unknown slurm.source %q", cfg.Slurm.Source)
}

func buildGPUSource(cfg *config.Config) (gpu.Source, error) {
	node := cfg.GPU.NodeName
	if node == "" {
		node, _ = os.Hostname()
	}
	switch cfg.GPU.Source {
	case "nvidia-smi", "":
		return gpu.NewSMISource(node), nil
	case "simulator":
		return gpu.NewSimSource(node, cfg.GPU.Sim.GPUs,
			gpu.Scenario(cfg.GPU.Sim.Scenario), cfg.GPU.Sim.Seed), nil
	}
	return nil, fmt.Errorf("unknown gpu.source %q", cfg.GPU.Source)
}

func newLogger(format string) *slog.Logger {
	if format == "json" {
		return slog.New(slog.NewJSONHandler(os.Stdout, nil))
	}
	return slog.New(slog.NewTextHandler(os.Stdout, nil))
}

func startMetricsServer(addr string, reg *prometheus.Registry, logger *slog.Logger) *http.Server {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{}))
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, "ok")
	})

	srv := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	go func() {
		logger.Info("metrics listening", "addr", addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("metrics server", "err", err)
		}
	}()
	return srv
}

func shutdown(srv *http.Server, logger *slog.Logger) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		logger.Error("metrics shutdown", "err", err)
	}
}
