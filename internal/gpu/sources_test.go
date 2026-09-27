package gpu

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"testing"
)

// fakeSMI writes a stand-in nvidia-smi that prints a fixture for the device
// query and another for the compute-apps query. appsExit != 0 makes the
// process query fail, the way it can on a node where it is not permitted.
func fakeSMI(t *testing.T, gpuFixture, appsFixture string, appsExit int) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("fake binaries are shell scripts")
	}
	abs := func(p string) string {
		a, err := filepath.Abs(filepath.Join("testdata", p))
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	script := "#!/bin/sh\ncase \"$*\" in\n" +
		"  *--query-gpu=*) cat '" + abs(gpuFixture) + "' ;;\n" +
		"  *--query-compute-apps=*)\n"
	if appsExit != 0 {
		script += "    echo 'Failed to query compute apps' >&2; exit " + strconv.Itoa(appsExit) + " ;;\n"
	} else {
		script += "    cat '" + abs(appsFixture) + "' ;;\n"
	}
	script += "  *) echo \"unexpected args: $*\" >&2; exit 2 ;;\nesac\n"
	path := filepath.Join(t.TempDir(), "nvidia-smi")
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func smiSource(bin string) *SMISource {
	s := NewSMISource("gpu001")
	s.Binary = bin
	return s
}

func TestSMIParsesReadableValues(t *testing.T) {
	src := smiSource(fakeSMI(t, "smi_ok.csv", "smi_apps.csv", 0))
	got, err := src.Sample(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("want 2 samples, got %d", len(got))
	}
	g0, g1 := got[0], got[1]
	if !g0.Judgeable() || !g0.PowerKnown || !g0.PIDsKnown {
		t.Fatalf("gpu0 should be fully readable: %+v", g0)
	}
	if g0.SMUtilPct != 97 || g0.PowerWatts != 612.5 || len(g0.PIDs) != 2 {
		t.Fatalf("gpu0 values wrong: %+v", g0)
	}
	if g1.SMUtilPct != 0 || !g1.UtilKnown || len(g1.PIDs) != 0 || !g1.PIDsKnown {
		t.Fatalf("gpu1 is a known-idle device with a known-empty process list: %+v", g1)
	}
	if f := g0.MemUsedFraction(); f < 0.84 || f > 0.85 {
		t.Fatalf("gpu0 mem fraction = %.3f", f)
	}
}

// The audit's critical finding: [N/A] fields used to parse as 0, which made
// absent telemetry look like an idle GPU. Every unreadable field must now be
// unknown, and an unknown utilization or memory makes the sample unjudgeable.
func TestSMIUnreadableFieldsAreUnknownNotZero(t *testing.T) {
	cases := []struct {
		fixture        string
		wantJudgeable  []bool
		wantPowerKnown []bool
	}{
		{"smi_na.csv", []bool{false, false, false, false, false, false, false, false},
			[]bool{false, false, false, false, false, false, false, false}},
		{"smi_not_supported.csv", []bool{false, true}, []bool{true, false}},
		{"smi_unknown_error.csv", []bool{false}, []bool{true}},
	}
	for _, tc := range cases {
		t.Run(tc.fixture, func(t *testing.T) {
			src := smiSource(fakeSMI(t, tc.fixture, "smi_apps_empty.csv", 0))
			got, err := src.Sample(context.Background(), nil)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(tc.wantJudgeable) {
				t.Fatalf("want %d samples, got %d", len(tc.wantJudgeable), len(got))
			}
			for i, s := range got {
				if s.Judgeable() != tc.wantJudgeable[i] {
					t.Errorf("gpu%d judgeable=%t, want %t (note %q)", i, s.Judgeable(), tc.wantJudgeable[i], s.Note)
				}
				if s.PowerKnown != tc.wantPowerKnown[i] {
					t.Errorf("gpu%d powerKnown=%t, want %t", i, s.PowerKnown, tc.wantPowerKnown[i])
				}
				if !s.UtilKnown && s.SMUtilPct != 0 {
					t.Errorf("gpu%d: unknown utilization must not carry a value", i)
				}
				if !s.Judgeable() && s.Note == "" {
					t.Errorf("gpu%d: an unjudgeable sample should say why", i)
				}
			}
		})
	}
}

func TestSMIRefusesMIGEnabledGPUs(t *testing.T) {
	src := smiSource(fakeSMI(t, "smi_mig.csv", "smi_apps_empty.csv", 0))
	got, err := src.Sample(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Fatalf("want 3 samples, got %d", len(got))
	}
	// Row 0 is what the manual leads one to expect: MIG enabled and
	// utilization [N/A]. It is unjudgeable on utilization alone, so it cannot
	// show that the MIG flag does anything.
	if !got[0].MIG || got[0].Judgeable() {
		t.Fatalf("MIG-enabled GPU must not be judgeable: %+v", got[0])
	}
	// [N/A] for mig.mode.current means "not MIG-capable", which is fine.
	if got[1].MIG || !got[1].Judgeable() {
		t.Fatalf("GPU without MIG support should be judgeable: %+v", got[1])
	}
	// Row 2 is MIG enabled with readable utilization and memory. Only the MIG
	// flag stands between it and a judgement, so dropping that guard from
	// Judgeable fails here.
	if !got[2].UtilKnown || !got[2].MemKnown {
		t.Fatalf("fixture row 2 should read cleanly: %+v", got[2])
	}
	if !got[2].MIG || got[2].Judgeable() {
		t.Fatalf("a MIG-enabled GPU must not be judgeable even with readable utilization: %+v", got[2])
	}
}

func TestSMIFailedProcessQueryMakesPIDsUnknown(t *testing.T) {
	src := smiSource(fakeSMI(t, "smi_ok.csv", "", 3))
	got, err := src.Sample(context.Background(), nil)
	if err != nil {
		t.Fatalf("a failed process query must not fail collection: %v", err)
	}
	for _, s := range got {
		if s.PIDsKnown {
			t.Fatalf("process list must be unknown when the query failed: %+v", s)
		}
	}
}

func TestSMIMalformedRowFailsTheWholeSample(t *testing.T) {
	// Dropping one unparseable row would let a job be judged on the GPUs that
	// happened to parse.
	src := smiSource(fakeSMI(t, "smi_bad_columns.csv", "smi_apps_empty.csv", 0))
	if _, err := src.Sample(context.Background(), nil); err == nil {
		t.Fatal("a row with the wrong column count must fail the sample")
	}
}

func TestSMIMissingBinaryIsUnavailable(t *testing.T) {
	src := smiSource(filepath.Join(t.TempDir(), "does-not-exist"))
	_, err := src.Sample(context.Background(), nil)
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("want ErrUnavailable, got %v", err)
	}
}

func TestParseReading(t *testing.T) {
	for _, in := range []string{"[N/A]", "N/A", "[Not Supported]", "[Unknown Error]",
		"[Insufficient Permissions]", "", "  ", "abc", "NaN", "-1", "101"} {
		if _, ok := parseReading(in, 0, 100); ok {
			t.Errorf("parseReading(%q) should be unknown", in)
		}
	}
	if v, ok := parseReading(" 42.5 ", 0, 100); !ok || v != 42.5 {
		t.Errorf("parseReading(42.5) = %v, %v", v, ok)
	}
	if v, ok := parseReading("0", 0, 100); !ok || v != 0 {
		t.Errorf("a real zero is a known zero: %v, %v", v, ok)
	}
}

func TestParseComputeAppsRejectsGarbage(t *testing.T) {
	if _, err := parseComputeApps([]byte("GPU-x, [N/A]\n")); err == nil {
		t.Fatal("an unparseable PID must make the process list unknown")
	}
	m, err := parseComputeApps([]byte("\n"))
	if err != nil || len(m) != 0 {
		t.Fatalf("empty output is a known-empty list: %v %v", m, err)
	}
}

func TestSimulatorMarksEveryFieldKnown(t *testing.T) {
	for _, sc := range []Scenario{ScenarioHealthy, ScenarioIdle, ScenarioHung, ScenarioStarved, ScenarioFlaky} {
		s := NewSimSource("n", 2, sc, 1)
		got, err := s.Sample(context.Background(), []string{"a", "b", "a"})
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 4 {
			t.Fatalf("%s: want 2 nodes x 2 GPUs, got %d", sc, len(got))
		}
		for _, smp := range got {
			if !smp.Judgeable() || !smp.PowerKnown || !smp.PIDsKnown {
				t.Fatalf("%s: simulator samples should be fully known: %+v", sc, smp)
			}
			if !smp.Timestamp.Equal(got[0].Timestamp) {
				t.Fatalf("%s: one cycle should share one timestamp", sc)
			}
		}
	}
}

func TestSimulatorUnreadableScenario(t *testing.T) {
	got, err := NewSimSource("n", 2, ScenarioUnreadable, 1).Sample(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range got {
		if s.Judgeable() || s.PowerKnown {
			t.Fatalf("unreadable scenario must produce unjudgeable samples: %+v", s)
		}
	}
}

func TestSimulatorRejectsUnknownScenario(t *testing.T) {
	if _, err := NewSimSource("n", 1, "bogus", 1).Sample(context.Background(), nil); err == nil {
		t.Fatal("unknown scenario should error")
	}
}

func TestPartialErrorMessageIsSorted(t *testing.T) {
	e := &PartialError{Failed: map[string]error{"b": errors.New("x"), "a": errors.New("y")}}
	if got := e.Error(); got != "2 node(s) not sampled: a: y; b: x" {
		t.Fatalf("got %q", got)
	}
}
