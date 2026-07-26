// Package config loads and validates runtime configuration.
package config

import (
	"fmt"
	"os"
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
		TokenEnv string `yaml:"token_env"`
	} `yaml:"slurm"`

	GPU struct {
		Source   string `yaml:"source"` // nvidia-smi | simulator
		NodeName string `yaml:"node_name"`
		Sim      struct {
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

	Exemptions policy.Exemptions `yaml:"exemptions"`

	Slack struct {
		WebhookEnv string `yaml:"webhook_env"`
		MinVerdict string `yaml:"min_verdict"`
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
	c.GPU.Source = "nvidia-smi"
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

	c.Slack.MinVerdict = "alert"
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
	if err := yaml.Unmarshal(raw, c); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	return c, c.Validate()
}

func (c *Config) Validate() error {
	switch c.Mode {
	case "observe", "enforce":
	default:
		return fmt.Errorf("mode must be observe or enforce, got %q", c.Mode)
	}
	if c.Interval <= 0 {
		return fmt.Errorf("interval must be positive")
	}
	// Sampling slower than the window means a verdict can never be supported by
	// enough samples, and the daemon would sit silent forever looking healthy.
	if c.Thresholds.Window > 0 && c.Interval > c.Thresholds.Window {
		return fmt.Errorf("interval (%s) must not exceed thresholds.window (%s)",
			c.Interval, c.Thresholds.Window)
	}
	for _, s := range c.Stages {
		if _, err := ParseVerdict(s.Verdict); err != nil {
			return err
		}
	}
	if _, err := ParseVerdict(c.Slack.MinVerdict); err != nil {
		return fmt.Errorf("slack.min_verdict: %w", err)
	}
	return c.Exemptions.Compile()
}

// Thresholds converts config into the policy type.
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
