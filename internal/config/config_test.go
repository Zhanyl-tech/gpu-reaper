package config

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/Zhanyl-tech/gpu-reaper/internal/policy"
)

func load(t *testing.T, yaml string) (*Config, error) {
	t.Helper()
	p := filepath.Join(t.TempDir(), "c.yaml")
	if err := os.WriteFile(p, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	return Load(p)
}

func TestDefaultsValidate(t *testing.T) {
	c, err := Load("")
	if err != nil {
		t.Fatalf("defaults must validate: %v", err)
	}
	if c.Mode != "observe" {
		t.Fatalf("default mode must be observe, got %q", c.Mode)
	}
	if !reflect.DeepEqual(c.AllowCancelSignatures, []string{"idle"}) {
		t.Fatalf("default cancel signatures = %v", c.AllowCancelSignatures)
	}
}

// The audit's case: a typo'd key loaded with no error and an empty exemption.
func TestUnknownKeysAreRejected(t *testing.T) {
	for _, y := range []string{
		"exemptions: {user: [alice], partition: [interactive]}\n",
		"thresholds: {util: 10}\n",
		"stage: [{after: 0s, verdict: alert}]\n",
	} {
		if _, err := load(t, y); err == nil || !strings.Contains(err.Error(), "not found") {
			t.Errorf("%q: want an unknown-field error, got %v", y, err)
		}
	}
}

// name_pattern was silently ignored before: the exemption struct had no yaml
// tags, so yaml.v3 expected "namepattern".
func TestNamePatternIsDecoded(t *testing.T) {
	c, err := load(t, "exemptions:\n  users: [ci-bot]\n  name_pattern: \"^(debug|jupyter)-\"\n")
	if err != nil {
		t.Fatal(err)
	}
	if c.Exemptions.NamePattern != "^(debug|jupyter)-" || !reflect.DeepEqual(c.Exemptions.Users, []string{"ci-bot"}) {
		t.Fatalf("exemptions = %+v", c.Exemptions)
	}
}

func TestValidationRules(t *testing.T) {
	cases := []struct {
		name, yaml, want string
	}{
		{"typo mode", "mode: Enforce\n", "mode must be"},
		{"util over 100", "thresholds: {util_pct: 150}\n", "util_pct"},
		{"util zero", "thresholds: {util_pct: 0}\n", "util_pct"},
		{"mem fraction", "thresholds: {mem_held_fraction: 1.5}\n", "mem_held_fraction"},
		{"negative power", "thresholds: {idle_power_watts: -1}\n", "idle_power_watts"},
		{"negative warmup", "thresholds: {warmup: -1m}\n", "warmup"},
		{"min samples zero", "thresholds: {min_samples: 0}\n", "min_samples"},
		// The audit's case: every ordinary cycle gap looked like a collector
		// fault, so the daemon silently never escalated.
		{"interval above gap", "interval: 5m\nthresholds: {max_sample_gap: 3m}\n", "max_sample_gap"},
		{"gap equals interval", "interval: 3m\n", "max_sample_gap"},
		{"gap at window", "thresholds: {max_sample_gap: 20m}\n", "max_sample_gap"},
		{"interval above window", "interval: 30m\n", "max_sample_gap"},
		{"min samples do not fit", "thresholds: {min_samples: 11}\n", "min_samples"},
		{"drain shorter than window", "stages: [{after: 0s, verdict: alert}, {after: 5m, verdict: drain}]\n", "at least one window"},
		{"cancel shorter than window", "stages: [{after: 10m, verdict: cancel}]\n", "at least one window"},
		{"stage verdict healthy", "stages: [{after: 0s, verdict: healthy}]\n", "must be alert"},
		{"stage verdict typo", "stages: [{after: 0s, verdict: alrt}]\n", "unknown verdict"},
		{"negative stage", "stages: [{after: -1m, verdict: alert}]\n", "negative"},
		{"cancel starved", "allow_cancel_signatures: [idle, starved]\n", "can never be cancelled"},
		{"slack min healthy", "slack: {min_verdict: healthy}\n", "min_verdict"},
		{"bad log format", "log_format: xml\n", "log_format"},
		{"unknown slurm source", "slurm: {source: sacct}\n", "slurm.source"},
		{"rest without url", "slurm: {source: rest}\n", "rest_url"},
		{"rest two tokens", "slurm: {source: rest, rest_url: 'http://x', token_env: A, token_file: /f}\n", "not both"},
		{"enforce relative scancel", "mode: enforce\n", "slurm.scancel_path must be an absolute path"},
		{"enforce relative scontrol", "mode: enforce\nslurm: {scancel_path: /usr/bin/scancel}\n", "slurm.scontrol_path must be an absolute path"},
		// squeue decides which jobs exist and where; nvidia-smi decides what
		// is idle. A shadowed copy of either can steer a drain or cancel.
		{"enforce relative squeue", "mode: enforce\nslurm: {scancel_path: /usr/bin/scancel, scontrol_path: /usr/bin/scontrol}\n", "slurm.squeue_path must be an absolute path"},
		{"enforce relative nvidia-smi", "mode: enforce\nslurm: {squeue_path: /usr/bin/squeue, scancel_path: /usr/bin/scancel, scontrol_path: /usr/bin/scontrol}\n", "gpu.nvidia_smi_path must be an absolute path"},
		// The defaults drain after 60m. Only stages written in the file used
		// to be checked, so a 90m window loaded and drained one cycle after
		// the first alert.
		{"window above default drain", "thresholds: {window: 90m}\n", "default stages"},
		{"unknown gpu source", "gpu: {source: nvml}\n", "gpu.source"},
		{"dcgm no url", "gpu: {source: dcgm}\n", "gpu.dcgm.url"},
		{"dcgm bad scheme", "gpu: {source: dcgm, dcgm: {url: 'file:///etc/passwd'}}\n", "gpu.dcgm.url"},
		{"dcgm slow timeout", "gpu: {source: dcgm, dcgm: {url: 'http://{node}:9400/metrics', timeout: 5m}}\n", "timeout"},
		{"bad name pattern", "exemptions: {name_pattern: '([unclosed'}\n", "name_pattern"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := load(t, tc.yaml)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("want error containing %q, got %v", tc.want, err)
			}
		})
	}
}

func TestEnforceWithAbsolutePathsValidates(t *testing.T) {
	c, err := load(t, "mode: enforce\n"+
		"slurm: {squeue_path: /usr/bin/squeue, scancel_path: /usr/bin/scancel, scontrol_path: /usr/bin/scontrol}\n"+
		"gpu: {nvidia_smi_path: /usr/bin/nvidia-smi}\n")
	if err != nil {
		t.Fatal(err)
	}
	if c.Mode != "enforce" || c.GPU.NvidiaSMIPath != "/usr/bin/nvidia-smi" {
		t.Fatalf("not decoded: mode=%q nvidia_smi_path=%q", c.Mode, c.GPU.NvidiaSMIPath)
	}
}

// Paths of sources that are not in use are not checked: slurmrestd needs no
// squeue binary, and dcgm-exporter needs no nvidia-smi.
func TestEnforceChecksOnlyTheBinariesInUse(t *testing.T) {
	_, err := load(t, "mode: enforce\n"+
		"slurm: {source: rest, rest_url: 'http://slurmrestd:6820', scancel_path: /usr/bin/scancel, scontrol_path: /usr/bin/scontrol}\n"+
		"gpu: {source: dcgm, dcgm: {url: 'http://{node}:9400/metrics'}}\n")
	if err != nil {
		t.Fatal(err)
	}
}

// Observe mode keeps the PATH lookups: nothing it runs can change the cluster.
func TestObserveModeAllowsBareBinaryNames(t *testing.T) {
	c, err := load(t, "mode: observe\n")
	if err != nil {
		t.Fatal(err)
	}
	if c.Slurm.SqueuePath != "squeue" || c.GPU.NvidiaSMIPath != "nvidia-smi" {
		t.Fatalf("defaults: squeue_path=%q nvidia_smi_path=%q", c.Slurm.SqueuePath, c.GPU.NvidiaSMIPath)
	}
}

// The default stages are held to the same rule as written ones. A window
// within the default drain time still loads with no stages set; a longer one
// needs explicit stages, and loads once they are at least one window.
func TestDefaultStagesAreCheckedAgainstTheWindow(t *testing.T) {
	c, err := load(t, "thresholds: {window: 60m}\n")
	if err != nil {
		t.Fatalf("window equal to the default drain stage should load: %v", err)
	}
	if st, _ := c.PolicyStages(); !reflect.DeepEqual(st, policy.DefaultStages()) {
		t.Fatalf("no stages set should mean the defaults, got %v", st)
	}
	_, err = load(t, "thresholds: {window: 90m}\n")
	if err == nil || !strings.Contains(err.Error(), "default stages") || !strings.Contains(err.Error(), "drain after 1h0m0s") {
		t.Fatalf("want the default drain stage rejected against a 90m window, got %v", err)
	}
	if _, err := load(t, "thresholds: {window: 90m}\nstages: [{after: 0s, verdict: alert}, {after: 3h, verdict: drain}]\n"); err != nil {
		t.Fatalf("explicit stages at least one window long should load: %v", err)
	}
}

func TestAllowCancelSignaturesReplacesDefault(t *testing.T) {
	c, err := load(t, "allow_cancel_signatures: [hung, idle]\n")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(c.CancelSignatures(), []policy.Signature{policy.SigHung, policy.SigIdle}) {
		t.Fatalf("got %v", c.CancelSignatures())
	}
	c, err = load(t, "allow_cancel_signatures: []\n")
	if err != nil || len(c.CancelSignatures()) != 0 {
		t.Fatalf("an empty list means nothing is cancel-eligible: %v %v", c.CancelSignatures(), err)
	}
}

func TestDCGMCentralURLValidates(t *testing.T) {
	c, err := load(t, "gpu: {source: dcgm, dcgm: {url: 'http://{node}:9400/metrics'}}\n")
	if err != nil {
		t.Fatal(err)
	}
	if c.GPU.DCGM.Timeout != 5*time.Second || c.GPU.DCGM.MaxAge != 2*time.Minute {
		t.Fatalf("dcgm defaults = %+v", c.GPU.DCGM)
	}
}

// The shipped example is what operators copy. It must validate, and no drain
// or cancel stage in it may be shorter than the window (it used to drain
// after 30s and carry the demo's simulator block and exemption).
func TestExampleConfigIsAProductionConfig(t *testing.T) {
	c, err := Load(filepath.Join("..", "..", "config.example.yaml"))
	if err != nil {
		t.Fatalf("config.example.yaml: %v", err)
	}
	if c.Mode != "observe" {
		t.Fatalf("example must ship in observe mode, got %q", c.Mode)
	}
	if c.GPU.Source == "simulator" {
		t.Fatal("example must not use the simulator")
	}
	stages, err := c.PolicyStages()
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range stages {
		if s.Verdict >= policy.Drain && s.After < c.Thresholds.Window {
			t.Fatalf("stage %s after %s is below the window %s", s.Verdict, s.After, c.Thresholds.Window)
		}
		if s.Verdict == policy.Cancel {
			t.Fatal("example must not enable cancel")
		}
	}
	if len(c.Exemptions.Users) != 0 {
		t.Fatalf("example must not carry demo exemptions: %v", c.Exemptions.Users)
	}
	if !reflect.DeepEqual(c.AllowCancelSignatures, []string{"idle"}) {
		t.Fatalf("example cancel signatures = %v", c.AllowCancelSignatures)
	}
}

func TestDemoConfigValidates(t *testing.T) {
	c, err := Load(filepath.Join("..", "..", "demo", "config.yaml"))
	if err != nil {
		t.Fatalf("demo/config.yaml: %v", err)
	}
	if c.Mode != "observe" {
		t.Fatal("the demo must never enforce")
	}
}

func TestParseVerdict(t *testing.T) {
	for in, want := range map[string]policy.Verdict{"alert": policy.Alert, " Drain ": policy.Drain, "CANCEL": policy.Cancel} {
		if got, err := ParseVerdict(in); err != nil || got != want {
			t.Errorf("ParseVerdict(%q) = %v, %v", in, got, err)
		}
	}
}
