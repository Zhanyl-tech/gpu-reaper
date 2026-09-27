// Package config loads and validates runtime configuration.
//
// Decoding is strict: an unknown key is an error, not a silent no-op. A typo
// such as `user:` for `users:` used to load cleanly and leave the exemption
// empty, so the jobs an operator meant to protect were not protected.
package config

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/Zhanyl-tech/gpu-reaper/internal/policy"
)

type Config struct {
	// Mode is "observe" or "enforce". Observe is the default and must stay the
	// default: a tool that cancels jobs the moment it is installed does not get
	// installed twice.
	Mode string `yaml:"mode"`

	Interval    time.Duration `yaml:"interval"`
	MetricsAddr string        `yaml:"metrics_addr"`
	LogFormat   string        `yaml:"log_format"` // text | json

	Slurm struct {
		Source   string `yaml:"source"` // squeue | rest
		RESTURL  string `yaml:"rest_url"`
		RESTVer  string `yaml:"rest_version"`
		Username string `yaml:"username"`
		// TokenEnv names an env var holding a JWT, read once at startup.
		// TokenFile is re-read on every request, so rotation needs no restart.
		TokenEnv  string `yaml:"token_env"`
		TokenFile string `yaml:"token_file"`

		// Binary paths. In enforce mode every binary on the destructive path
		// must be absolute, so a writable directory earlier in PATH cannot
		// shadow it: scancel and scontrol act, and squeue decides which jobs
		// exist and where (required when source is squeue).
		SqueuePath   string `yaml:"squeue_path"`
		ScancelPath  string `yaml:"scancel_path"`
		ScontrolPath string `yaml:"scontrol_path"`
	} `yaml:"slurm"`

	GPU struct {
		Source string `yaml:"source"` // nvidia-smi | dcgm | simulator
		// NodeName is this host's Slurm NodeName. It defaults to the OS
		// hostname, which is wrong wherever the two differ (an FQDN, or a
		// Kubernetes pod whose hostname is not the NodeName).
		NodeName string `yaml:"node_name"`
		// NvidiaSMIPath is the nvidia-smi binary. Its output decides whether a
		// GPU is idle, so in enforce mode with source nvidia-smi it must be
		// absolute: a shadowing binary reporting every GPU idle would get a
		// busy job drained and cancelled.
		NvidiaSMIPath string `yaml:"nvidia_smi_path"`
		DCGM          struct {
			// URL of dcgm-exporter's metrics endpoint. "{node}" makes the
			// source central (one scrape per Slurm node).
			URL     string        `yaml:"url"`
			Timeout time.Duration `yaml:"timeout"`
			MaxAge  time.Duration `yaml:"max_age"`
		} `yaml:"dcgm"`
		Sim struct {
			GPUs     int    `yaml:"gpus"`
			Scenario string `yaml:"scenario"`
			Seed     int64  `yaml:"seed"`
		} `yaml:"simulator"`
	} `yaml:"gpu"`

	Thresholds struct {
		UtilPct         float64       `yaml:"util_pct"`
		MemHeldFraction float64       `yaml:"mem_held_fraction"`
		IdlePowerWatts  float64       `yaml:"idle_power_watts"`
		Window          time.Duration `yaml:"window"`
		Warmup          time.Duration `yaml:"warmup"`
		MinSamples      int           `yaml:"min_samples"`
		MaxSampleGap    time.Duration `yaml:"max_sample_gap"`
	} `yaml:"thresholds"`

	Stages []struct {
		After   time.Duration `yaml:"after"`
		Verdict string        `yaml:"verdict"`
	} `yaml:"stages"`

	// AllowCancelSignatures lists the signatures a cancel stage applies to.
	// Default [idle]. hung may be added deliberately; starved and unknown
	// are rejected.
	AllowCancelSignatures []string `yaml:"allow_cancel_signatures"`

	Exemptions policy.Exemptions `yaml:"exemptions"`

	Slack struct {
		WebhookEnv  string        `yaml:"webhook_env"`
		MinVerdict  string        `yaml:"min_verdict"`
		RemindEvery time.Duration `yaml:"remind_every"`
	} `yaml:"slack"`
}

func Default() *Config {
	c := &Config{
		Mode:        "observe",
		Interval:    2 * time.Minute,
		MetricsAddr: ":9835",
		LogFormat:   "text",
	}
	c.Slurm.Source = "squeue"
	c.Slurm.RESTVer = "v0.0.42"
	c.Slurm.SqueuePath = "squeue"
	c.Slurm.ScancelPath = "scancel"
	c.Slurm.ScontrolPath = "scontrol"
	c.GPU.Source = "nvidia-smi"
	c.GPU.NvidiaSMIPath = "nvidia-smi"
	c.GPU.DCGM.Timeout = 5 * time.Second
	c.GPU.DCGM.MaxAge = 2 * time.Minute
	c.GPU.Sim.GPUs = 8
	c.GPU.Sim.Scenario = "hung"

	t := policy.DefaultThresholds()
	c.Thresholds.UtilPct = t.UtilPct
	c.Thresholds.MemHeldFraction = t.MemHeldFraction
	c.Thresholds.IdlePowerWatts = t.IdlePowerWatts
	c.Thresholds.Window = t.Window
	c.Thresholds.Warmup = t.Warmup
	c.Thresholds.MinSamples = t.MinSamples
	c.Thresholds.MaxSampleGap = t.MaxSampleGap

	for _, s := range policy.DefaultCancelSignatures() {
		c.AllowCancelSignatures = append(c.AllowCancelSignatures, string(s))
	}

	c.Slack.MinVerdict = "alert"
	c.Slack.RemindEvery = 4 * time.Hour
	return c
}

func Load(path string) (*Config, error) {
	c := Default()
	if path == "" {
		return c, c.Validate()
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	if err := Decode(raw, c); err != nil {
		return nil, err
	}
	return c, c.Validate()
}

// Decode strictly decodes YAML onto c, which should hold defaults.
func Decode(raw []byte, c *Config) error {
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	dec.KnownFields(true)
	if err := dec.Decode(c); err != nil && !errors.Is(err, io.EOF) {
		return fmt.Errorf("parse config: %w", err)
	}
	return nil
}

// Validate rejects configurations that would run but do something other than
// what the operator plausibly meant. Each rule states the failure it prevents.
func (c *Config) Validate() error {
	switch c.Mode {
	case "observe", "enforce":
	default:
		return fmt.Errorf("mode must be observe or enforce, got %q", c.Mode)
	}
	switch c.LogFormat {
	case "text", "json":
	default:
		return fmt.Errorf("log_format must be text or json, got %q", c.LogFormat)
	}
	if c.MetricsAddr == "" {
		return errors.New("metrics_addr must be set")
	}
	if c.Interval <= 0 {
		return errors.New("interval must be positive")
	}

	th := c.Thresholds
	if th.UtilPct <= 0 || th.UtilPct > 100 {
		return fmt.Errorf("thresholds.util_pct must be in (0, 100], got %g", th.UtilPct)
	}
	if th.MemHeldFraction < 0 || th.MemHeldFraction > 1 {
		return fmt.Errorf("thresholds.mem_held_fraction must be in [0, 1], got %g", th.MemHeldFraction)
	}
	if th.IdlePowerWatts < 0 {
		return fmt.Errorf("thresholds.idle_power_watts must not be negative")
	}
	if th.Window <= 0 {
		return errors.New("thresholds.window must be positive")
	}
	if th.Warmup < 0 {
		return errors.New("thresholds.warmup must not be negative")
	}
	if th.MinSamples < 1 {
		return errors.New("thresholds.min_samples must be at least 1")
	}
	// interval < max_sample_gap: otherwise every ordinary gap between two
	// cycles looks like a collector fault and nothing is ever judged.
	// max_sample_gap < window: otherwise one snapshot "covers" the window.
	if !(c.Interval < th.MaxSampleGap && th.MaxSampleGap < th.Window) {
		return fmt.Errorf("need interval (%s) < thresholds.max_sample_gap (%s) < thresholds.window (%s)",
			c.Interval, th.MaxSampleGap, th.Window)
	}
	// min_samples counts cycles. If that many cycles do not fit in the window,
	// the daemon would sit silent forever looking healthy.
	if time.Duration(th.MinSamples)*c.Interval > th.Window {
		return fmt.Errorf("thresholds.min_samples (%d) x interval (%s) exceeds thresholds.window (%s); no breach could ever be confirmed",
			th.MinSamples, c.Interval, th.Window)
	}

	for i, s := range c.Stages {
		v, err := ParseVerdict(s.Verdict)
		if err != nil {
			return fmt.Errorf("stages[%d]: %w", i, err)
		}
		if v < policy.Alert {
			return fmt.Errorf("stages[%d]: verdict must be alert, drain or cancel, got %q", i, s.Verdict)
		}
		if s.After < 0 {
			return fmt.Errorf("stages[%d]: after must not be negative", i)
		}
	}
	// Dwell is measured from the first breaching observation, so it is already
	// about one window long when a breach is first confirmed. A drain or cancel
	// stage shorter than the window would fire on essentially the first
	// confirmation, one cycle after the first alert.
	//
	// This checks the stages the engine will actually run, not only the ones in
	// the file. With `stages:` omitted the defaults apply (drain after 60m), and
	// checking only the file let `window: 90m` load and drain one cycle after
	// the first alert.
	stages, err := c.PolicyStages()
	if err != nil {
		return err
	}
	for i, s := range stages {
		if s.Verdict >= policy.Drain && s.After < th.Window {
			where := fmt.Sprintf("stages[%d]", i)
			if len(c.Stages) == 0 {
				where = "default stages (stages: not set)"
			}
			return fmt.Errorf("%s: %s after %s is shorter than thresholds.window (%s); destructive stages must be at least one window",
				where, s.Verdict, s.After, th.Window)
		}
	}
	for _, s := range c.AllowCancelSignatures {
		if s != string(policy.SigIdle) && s != string(policy.SigHung) {
			return fmt.Errorf("allow_cancel_signatures: %q can never be cancelled (allowed: idle, hung)", s)
		}
	}

	if min, err := ParseVerdict(c.Slack.MinVerdict); err != nil {
		return fmt.Errorf("slack.min_verdict: %w", err)
	} else if min < policy.Alert {
		return fmt.Errorf("slack.min_verdict must be alert, drain or cancel")
	}
	if c.Slack.RemindEvery < 0 {
		return errors.New("slack.remind_every must not be negative")
	}

	switch c.Slurm.Source {
	case "squeue":
	case "rest":
		if c.Slurm.RESTURL == "" {
			return errors.New("slurm.rest_url required when source is rest")
		}
		if c.Slurm.TokenEnv != "" && c.Slurm.TokenFile != "" {
			return errors.New("set slurm.token_env or slurm.token_file, not both")
		}
	default:
		return fmt.Errorf("slurm.source must be squeue or rest, got %q", c.Slurm.Source)
	}
	if c.Mode == "enforce" {
		// Every binary whose output or effect reaches a drain or cancel. A
		// shadowed squeue can place any job on an idle node; a shadowed
		// nvidia-smi can report a busy node idle. Paths of sources not in use
		// are not checked.
		type binPath struct{ key, path string }
		paths := []binPath{
			{"slurm.scancel_path", c.Slurm.ScancelPath},
			{"slurm.scontrol_path", c.Slurm.ScontrolPath},
		}
		if c.Slurm.Source == "squeue" {
			paths = append(paths, binPath{"slurm.squeue_path", c.Slurm.SqueuePath})
		}
		if c.GPU.Source == "nvidia-smi" {
			paths = append(paths, binPath{"gpu.nvidia_smi_path", c.GPU.NvidiaSMIPath})
		}
		for _, p := range paths {
			if !filepath.IsAbs(p.path) {
				return fmt.Errorf("%s must be an absolute path in enforce mode, got %q", p.key, p.path)
			}
		}
	}

	switch c.GPU.Source {
	case "nvidia-smi", "simulator":
	case "dcgm":
		u, err := url.Parse(strings.ReplaceAll(c.GPU.DCGM.URL, "{node}", "node"))
		if c.GPU.DCGM.URL == "" || err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
			return fmt.Errorf("gpu.dcgm.url must be an http(s) URL, got %q", c.GPU.DCGM.URL)
		}
		if c.GPU.DCGM.Timeout <= 0 {
			return errors.New("gpu.dcgm.timeout must be positive")
		}
		if c.GPU.DCGM.Timeout >= c.Interval {
			return fmt.Errorf("gpu.dcgm.timeout (%s) must be shorter than interval (%s)", c.GPU.DCGM.Timeout, c.Interval)
		}
	default:
		return fmt.Errorf("gpu.source must be nvidia-smi, dcgm or simulator, got %q", c.GPU.Source)
	}

	return c.Exemptions.Compile()
}

// PolicyThresholds converts config into the policy type.
func (c *Config) PolicyThresholds() policy.Thresholds {
	return policy.Thresholds{
		UtilPct:         c.Thresholds.UtilPct,
		MemHeldFraction: c.Thresholds.MemHeldFraction,
		IdlePowerWatts:  c.Thresholds.IdlePowerWatts,
		Window:          c.Thresholds.Window,
		Warmup:          c.Thresholds.Warmup,
		MinSamples:      c.Thresholds.MinSamples,
		MaxSampleGap:    c.Thresholds.MaxSampleGap,
	}
}

// PolicyStages converts config stages, falling back to defaults when unset.
// Validate checks what this returns, so the defaults are held to the same
// rules as stages written in the file.
func (c *Config) PolicyStages() ([]policy.Stage, error) {
	if len(c.Stages) == 0 {
		return policy.DefaultStages(), nil
	}
	out := make([]policy.Stage, 0, len(c.Stages))
	for _, s := range c.Stages {
		v, err := ParseVerdict(s.Verdict)
		if err != nil {
			return nil, err
		}
		out = append(out, policy.Stage{After: s.After, Verdict: v})
	}
	return out, nil
}

// CancelSignatures converts allow_cancel_signatures.
func (c *Config) CancelSignatures() []policy.Signature {
	out := make([]policy.Signature, 0, len(c.AllowCancelSignatures))
	for _, s := range c.AllowCancelSignatures {
		out = append(out, policy.Signature(s))
	}
	return out
}

func ParseVerdict(s string) (policy.Verdict, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "healthy":
		return policy.Healthy, nil
	case "watching":
		return policy.Watching, nil
	case "alert":
		return policy.Alert, nil
	case "drain":
		return policy.Drain, nil
	case "cancel":
		return policy.Cancel, nil
	}
	return policy.Healthy, fmt.Errorf("unknown verdict %q (want alert, drain, or cancel)", s)
}
