package slurm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestParseGPUCount(t *testing.T) {
	cases := []struct {
		tres string
		want int
	}{
		{"cpu=32,mem=256G,node=2,billing=32,gres/gpu=8", 8},
		{"cpu=8,mem=64G,gres/gpu:a100=4", 4},
		// Slurm emits both the typed and untyped forms; counting both would
		// report twice the GPUs actually allocated and double every waste figure.
		{"cpu=64,gres/gpu=16,gres/gpu:h100=16", 16},
		{"cpu=64,gres/gpu:h100=16,gres/gpu=16", 16},
		{"cpu=4,mem=16G", 0},
		{"", 0},
		{"gres/gpu:a100=2,gres/gpu:v100=3", 5},
		// Typed-only names with uppercase, hyphens and dots used to parse as 0.
		{"gres/gpu:A100=4", 4},
		{"gres/gpu:a100-sxm4-80gb=4", 4},
		{"gres/gpu:1g.10gb=2", 2},
		// gpumem/gpuutil TRES are not GPU counts.
		{"cpu=8,gres/gpumem=80G,gres/gpuutil=0,gres/gpu=1", 1},
		{"gres/gpumem=80G", 0},
	}
	for _, tc := range cases {
		if got := ParseGPUCount(tc.tres); got != tc.want {
			t.Errorf("ParseGPUCount(%q) = %d, want %d", tc.tres, got, tc.want)
		}
	}
}

func TestExpandNodeList(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"gpu001", []string{"gpu001"}},
		{"gpu[001-004]", []string{"gpu001", "gpu002", "gpu003", "gpu004"}},
		{"gpu[001-003,007]", []string{"gpu001", "gpu002", "gpu003", "gpu007"}},
		{"gpu[01-02],cpu[5-6]", []string{"gpu01", "gpu02", "cpu5", "cpu6"}},
		{"node1,node2", []string{"node1", "node2"}},
		{"", nil},
		{"(null)", nil},
		{"None assigned", nil},
		// Zero padding must be preserved: gpu007 is a different host from gpu7.
		{"gpu[007-009]", []string{"gpu007", "gpu008", "gpu009"}},
		{"gpu[9-10]", []string{"gpu9", "gpu10"}},
		// Forms the audit found returned unexpanded.
		{"gpu[01-02]-ib", []string{"gpu01-ib", "gpu02-ib"}},
		{"rack[1-2]-gpu[01-02]", []string{"rack1-gpu01", "rack1-gpu02", "rack2-gpu01", "rack2-gpu02"}},
		{"gpu[01-02].cluster", []string{"gpu01.cluster", "gpu02.cluster"}},
		{"gpu-2-1", []string{"gpu-2-1"}},
	}
	for _, tc := range cases {
		got, err := ExpandNodeList(tc.in)
		if err != nil {
			t.Errorf("ExpandNodeList(%q): %v", tc.in, err)
			continue
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("ExpandNodeList(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

func TestExpandNodeListRejectsMalformed(t *testing.T) {
	for _, in := range []string{
		"gpu[009-003]", // inverted
		"gpu[01-02",    // unbalanced
		"gpu01-02]",    // unbalanced
		"gpu[[1-2]]",   // nested
		"gpu[a-b]",     // not numeric
		"gpu[1-]",      // open range
		"gpu[]",        // empty
		"gpu 01",       // space
		"gpu[1-2]$x",   // bad literal
		"a,,b",         // empty host
		"gpu[0-99999999]",
		"gpu[0-100000]", // one past the cap
		// hi-lo+1 overflowed to MinInt64 and passed the cap, then the loop
		// allocated until memory ran out.
		"gpu[0-9223372036854775807]",
		// Many spans, each under the cap, in one bracket: about 1e8 strings
		// were built before the total was checked.
		"gpu[" + strings.Repeat("1-99999,", 999) + "1-99999]",
		// Two brackets whose product is 2.5e9: make() was sized by the product
		// before any check.
		"a[1-50000]b[1-50000]",
		// Several parts, each under the cap, together over it.
		"a[1-60000],b[1-60000]",
	} {
		if got, err := expandWithin(t, in); err == nil {
			t.Errorf("ExpandNodeList(%q) = %d names, want an error", truncate(in, 40), len(got))
		}
	}
}

// The size cap is exact, and a span ending at MaxInt64 terminates (n++ used to
// wrap there, so the loop never ended).
func TestExpandNodeListSizeCapIsExact(t *testing.T) {
	got, err := expandWithin(t, "gpu[1-100000]")
	if err != nil || len(got) != maxExpandedNodes {
		t.Fatalf("exactly the cap should expand: %d names, %v", len(got), err)
	}
	got, err = expandWithin(t, "gpu[9223372036854775807]")
	if err != nil || !reflect.DeepEqual(got, []string{"gpu9223372036854775807"}) {
		t.Fatalf("got %v, %v", got, err)
	}
}

// expandWithin fails the whole test binary if ExpandNodeList does not return
// promptly. A regression here allocates without bound, so waiting for go
// test's own timeout would mean waiting for the machine to run out of memory.
func expandWithin(t *testing.T, in string) ([]string, error) {
	t.Helper()
	const deadline = 20 * time.Second
	timer := time.AfterFunc(deadline, func() {
		panic("ExpandNodeList(" + truncate(in, 40) + ") did not return within " + deadline.String())
	})
	defer timer.Stop()
	return ExpandNodeList(in)
}

// ── squeue parsing ─────────────────────────────────────────────────────────

const goodLine = "100241|alice|research|gpu|normal|RUNNING|2026-09-26T08:00:00|gpu[001-002]|cpu=64,gres/gpu=16|llama-70b-sft"

func TestParseSqueueLine(t *testing.T) {
	jobs := ParseSqueue(goodLine+"\n", nil)
	if len(jobs) != 1 {
		t.Fatalf("want 1 job, got %d", len(jobs))
	}
	j := jobs[0]
	want := Job{
		JobID: "100241", Name: "llama-70b-sft", User: "alice", Account: "research",
		Partition: "gpu", QOS: "normal", State: "RUNNING",
		StartTime: time.Date(2026, 9, 26, 8, 0, 0, 0, time.UTC),
		Nodes:     []string{"gpu001", "gpu002"}, GPUCount: 16,
	}
	if !reflect.DeepEqual(j, want) {
		t.Fatalf("got %+v\nwant %+v", j, want)
	}
}

// The audit's finding: a job named "eval|ci-bot" parsed as user ci-bot with 0
// GPUs, so any user could hide a job or borrow an exemption.
func TestPipeInJobNameCannotShiftFields(t *testing.T) {
	line := "7|mallory|research|gpu|normal|RUNNING|2026-09-26T08:00:00|gpu003|cpu=8,gres/gpu=4|eval|ci-bot|x"
	jobs := ParseSqueue(line, nil)
	if len(jobs) != 1 {
		t.Fatalf("want 1 job, got %d", len(jobs))
	}
	j := jobs[0]
	if j.User != "mallory" || j.GPUCount != 4 || j.Name != "eval|ci-bot|x" {
		t.Fatalf("fields shifted: %+v", j)
	}
}

func TestParseSqueueRejectsBadLines(t *testing.T) {
	cases := map[string]string{
		"too few fields":  "1|alice|research|gpu|normal|RUNNING|2026-09-26T08:00:00|gpu001",
		"bad job id":      "abc|alice|research|gpu|normal|RUNNING|2026-09-26T08:00:00|gpu001|gres/gpu=1|n",
		"not running":     "1|alice|research|gpu|normal|PENDING|2026-09-26T08:00:00|gpu001|gres/gpu=1|n",
		"custom time fmt": "1|alice|research|gpu|normal|RUNNING|Sat Sep 26 08:00|gpu001|gres/gpu=1|n",
		"unknown start":   "1|alice|research|gpu|normal|RUNNING|Unknown|gpu001|gres/gpu=1|n",
		"bad hostlist":    "1|alice|research|gpu|normal|RUNNING|2026-09-26T08:00:00|gpu[01-02|gres/gpu=1|n",
	}
	for name, line := range cases {
		t.Run(name, func(t *testing.T) {
			var reasons []string
			jobs := ParseSqueue(line, func(r string) { reasons = append(reasons, r) })
			if len(jobs) != 0 {
				t.Fatalf("line should be rejected, got %+v", jobs)
			}
			if len(reasons) != 1 {
				t.Fatalf("rejection should be reported once, got %v", reasons)
			}
		})
	}
}

func TestParseSqueueDropsDuplicateJobIDs(t *testing.T) {
	out := goodLine + "\n" +
		"100241|root|admin|gpu|normal|RUNNING|2026-09-26T08:00:00|gpu009|gres/gpu=8|forged\n" +
		"100242|bob|infra|gpu|normal|RUNNING|2026-09-26T08:00:00|gpu003|gres/gpu=4|ok\n"
	var reasons []string
	jobs := ParseSqueue(out, func(r string) { reasons = append(reasons, r) })
	if len(jobs) != 1 || jobs[0].JobID != "100242" {
		t.Fatalf("both copies of a duplicated job must be dropped, got %+v", jobs)
	}
	if len(reasons) != 2 {
		t.Fatalf("each dropped copy should be reported, got %v", reasons)
	}
}

func TestArrayAndHetJobIDsAccepted(t *testing.T) {
	for _, id := range []string{"123_4", "123+0", "123_4+1"} {
		line := id + "|a|b|gpu|n|RUNNING|2026-09-26T08:00:00|gpu001|gres/gpu=1|n"
		if jobs := ParseSqueue(line, nil); len(jobs) != 1 {
			t.Errorf("job id %q should parse", id)
		}
	}
}

// ── squeue time zone (the audit's high-severity finding) ────────────────────

// fakeSqueue writes a stand-in that behaves like real squeue in the one way
// that matters here: it prints StartTime in local time, honouring TZ, in the
// format SLURM_TIME_FORMAT selects. It records its arguments.
func fakeSqueue(t *testing.T) (bin, argsFile string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("fake binaries are shell scripts")
	}
	dir := t.TempDir()
	argsFile = filepath.Join(dir, "args")
	script := `#!/bin/sh
fmt='%Y-%m-%dT%H:%M:%S'
if [ -n "$SLURM_TIME_FORMAT" ] && [ "$SLURM_TIME_FORMAT" != standard ]; then fmt="$SLURM_TIME_FORMAT"; fi
start=$(date -d "@$FAKE_START_EPOCH" +"$fmt" 2>/dev/null || date -r "$FAKE_START_EPOCH" +"$fmt")
echo "$*" > "` + argsFile + `"
printf '%s\n' "42|alice|research|gpu|normal|RUNNING|$start|gpu001|cpu=8,gres/gpu=4|train"
`
	bin = filepath.Join(dir, "squeue")
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return bin, argsFile
}

func TestSqueueStartTimeIsCorrectInAnyHostTimeZone(t *testing.T) {
	bin, _ := fakeSqueue(t)
	start := time.Date(2026, 9, 26, 14, 0, 0, 0, time.UTC)
	t.Setenv("FAKE_START_EPOCH", strconv.FormatInt(start.Unix(), 10))

	for _, tz := range []string{"America/New_York", "Asia/Tokyo", "America/Los_Angeles", "UTC"} {
		t.Run(tz, func(t *testing.T) {
			t.Setenv("TZ", tz)
			t.Setenv("SLURM_TIME_FORMAT", "")

			// Guard against a vacuous pass: confirm the fake really prints
			// local time when nothing overrides TZ.
			if tz != "UTC" {
				raw, err := exec.Command(bin).Output()
				if err != nil {
					t.Fatal(err)
				}
				if strings.Contains(string(raw), "2026-09-26T14:00:00") {
					t.Skipf("this platform's date(1) ignored TZ=%s; the test would prove nothing", tz)
				}
			}

			src := NewSqueueSource(bin)
			jobs, err := src.RunningJobs(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			if len(jobs) != 1 {
				t.Fatalf("want 1 job, got %d", len(jobs))
			}
			if !jobs[0].StartTime.Equal(start) {
				t.Fatalf("host TZ=%s: parsed start %s, want %s", tz, jobs[0].StartTime, start)
			}
		})
	}
}

func TestSqueueCustomTimeFormatIsOverridden(t *testing.T) {
	bin, _ := fakeSqueue(t)
	start := time.Date(2026, 9, 26, 14, 0, 0, 0, time.UTC)
	t.Setenv("FAKE_START_EPOCH", strconv.FormatInt(start.Unix(), 10))
	t.Setenv("SLURM_TIME_FORMAT", "%a %b %d %H:%M")

	var rejected []string
	src := NewSqueueSource(bin)
	src.OnReject = func(r string) { rejected = append(rejected, r) }
	jobs, err := src.RunningJobs(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(jobs) != 1 || !jobs[0].StartTime.Equal(start) {
		t.Fatalf("a site SLURM_TIME_FORMAT must not break parsing: jobs=%+v rejected=%v", jobs, rejected)
	}
}

func TestSqueueNodeListIsPassed(t *testing.T) {
	bin, argsFile := fakeSqueue(t)
	t.Setenv("FAKE_START_EPOCH", "1790000000")
	src := NewSqueueSource(bin)
	src.NodeList = "gpu001"
	if _, err := src.RunningJobs(context.Background()); err != nil {
		t.Fatal(err)
	}
	args, _ := os.ReadFile(argsFile)
	if !strings.Contains(string(args), "--nodelist=gpu001") {
		t.Fatalf("per-node daemon must scope squeue to its node, args: %s", args)
	}
	if !strings.Contains(string(args), "--Format="+squeueFormat) {
		t.Fatalf("format missing, args: %s", args)
	}
}

// ── REST ───────────────────────────────────────────────────────────────────

func TestNormalizeStateAcrossAPIVersions(t *testing.T) {
	if got := normalizeState("RUNNING"); got != "RUNNING" {
		t.Errorf("string form: got %q", got)
	}
	if got := normalizeState([]any{"RUNNING", "CONFIGURING"}); got != "RUNNING" {
		t.Errorf("list form: got %q", got)
	}
	if got := normalizeState(nil); got != "" {
		t.Errorf("nil should yield empty, got %q", got)
	}
}

func TestNormalizeTimeAcrossAPIVersions(t *testing.T) {
	if got := normalizeTime(float64(1700000000)); got.Unix() != 1700000000 {
		t.Errorf("numeric form: got %v", got)
	}
	obj := map[string]any{"set": true, "infinite": false, "number": float64(1700000000)}
	if got := normalizeTime(obj); got.Unix() != 1700000000 {
		t.Errorf("object form: got %v", got)
	}
	for name, v := range map[string]any{
		"nil":      nil,
		"zero":     float64(0),
		"unset":    map[string]any{"set": false, "infinite": false, "number": float64(0)},
		"infinite": map[string]any{"set": true, "infinite": true, "number": float64(0)},
	} {
		// set=false with number 0 used to become 1970-01-01, i.e. a job 56
		// years old that skipped warmup.
		if got := normalizeTime(v); !got.IsZero() {
			t.Errorf("%s should yield zero time, got %v", name, got)
		}
	}
}

// Response bodies in the shapes the parser supports: job_state as a string
// (older API versions) and as a list with start_time as an object (newer).
const restOld = `{"jobs":[{"job_id":11,"name":"a","user_name":"alice","account":"r","partition":"gpu","qos":"normal","job_state":"RUNNING","start_time":1790000000,"nodes":"gpu[001-002]","tres_alloc_str":"cpu=8,gres/gpu=16"},
{"job_id":12,"name":"b","user_name":"bob","job_state":"PENDING","start_time":0,"nodes":"","tres_alloc_str":""}]}`

const restNew = `{"jobs":[{"job_id":21,"name":"c","user_name":"carol","account":"r","partition":"gpu","qos":"normal","job_state":["RUNNING"],"start_time":{"set":true,"infinite":false,"number":1790000000},"nodes":"gpu003","tres_alloc_str":"gres/gpu:h100=8"},
{"job_id":22,"name":"d","user_name":"dave","job_state":["RUNNING"],"start_time":{"set":true,"infinite":false,"number":1790000000},"nodes":"gpu[01-02","tres_alloc_str":"gres/gpu=1"}]}`

func TestRESTSourceParsesBothShapes(t *testing.T) {
	var gotToken string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotToken = r.Header.Get("X-SLURM-USER-TOKEN")
		switch r.URL.Path {
		case "/slurm/v0.0.40/jobs":
			_, _ = w.Write([]byte(restOld))
		case "/slurm/v0.0.42/jobs":
			_, _ = w.Write([]byte(restNew))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	old := NewRESTSource(srv.URL, "v0.0.40", "tok", "", "slurm")
	jobs, err := old.RunningJobs(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(jobs) != 1 || jobs[0].JobID != "11" || jobs[0].GPUCount != 16 || len(jobs[0].Nodes) != 2 {
		t.Fatalf("old shape: %+v", jobs)
	}
	if gotToken != "tok" {
		t.Fatalf("token header = %q", gotToken)
	}

	var rejected []string
	nw := NewRESTSource(srv.URL, "v0.0.42", "tok", "", "slurm")
	nw.OnReject = func(r string) { rejected = append(rejected, r) }
	jobs, err = nw.RunningJobs(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(jobs) != 1 || jobs[0].JobID != "21" || jobs[0].GPUCount != 8 || jobs[0].StartTime.Unix() != 1790000000 {
		t.Fatalf("new shape: %+v", jobs)
	}
	if len(rejected) != 1 {
		t.Fatalf("the job with a malformed hostlist should be rejected and reported: %v", rejected)
	}
}

func TestRESTTokenFileIsReReadEachRequest(t *testing.T) {
	var seen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = append(seen, r.Header.Get("X-SLURM-USER-TOKEN"))
		_, _ = w.Write([]byte(`{"jobs":[]}`))
	}))
	defer srv.Close()

	path := filepath.Join(t.TempDir(), "jwt")
	_ = os.WriteFile(path, []byte("first\n"), 0o600)
	src := NewRESTSource(srv.URL, "v0.0.42", "", path, "slurm")
	if _, err := src.RunningJobs(context.Background()); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(path, []byte("second\n"), 0o600)
	if _, err := src.RunningJobs(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(seen, []string{"first", "second"}) {
		t.Fatalf("rotated token not picked up: %v", seen)
	}
}

func TestRESTNon200IsAnError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "expired", http.StatusUnauthorized)
	}))
	defer srv.Close()
	if _, err := NewRESTSource(srv.URL, "v0.0.42", "t", "", "u").RunningJobs(context.Background()); err == nil {
		t.Fatal("401 must be an error")
	}
}

// ── Controller ─────────────────────────────────────────────────────────────

// recorder writes a fake binary that appends its name and arguments to a log
// and exits with the given code.
func recorder(t *testing.T, dir, name, log string, exit int) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("fake binaries are shell scripts")
	}
	p := filepath.Join(dir, name)
	script := "#!/bin/sh\necho \"" + name + " $*\" >> '" + log + "'\nexit " + strconv.Itoa(exit) + "\n"
	if err := os.WriteFile(p, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

func readLog(t *testing.T, path string) []string {
	t.Helper()
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		t.Fatal(err)
	}
	return strings.Split(strings.TrimSpace(string(b)), "\n")
}

func TestCancelRecordsReasonThenCancels(t *testing.T) {
	dir := t.TempDir()
	log := filepath.Join(dir, "calls")
	c := NewCLIController(recorder(t, dir, "scancel", log, 0), recorder(t, dir, "scontrol", log, 0))
	if err := c.Cancel(context.Background(), "4242", "gpu-reaper: idle, 3.0 GPU-hours"); err != nil {
		t.Fatal(err)
	}
	got := readLog(t, log)
	want := []string{
		"scontrol update JobId=4242 AdminComment=gpu-reaper: idle, 3.0 GPU-hours",
		"scancel --verbose 4242",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("calls:\n%v\nwant:\n%v", got, want)
	}
}

func TestCancelIsNotAttemptedWhenReasonCannotBeRecorded(t *testing.T) {
	dir := t.TempDir()
	log := filepath.Join(dir, "calls")
	c := NewCLIController(recorder(t, dir, "scancel", log, 0), recorder(t, dir, "scontrol", log, 1))
	err := c.Cancel(context.Background(), "4242", "gpu-reaper: idle")
	if err == nil || !strings.Contains(err.Error(), "NOT cancelled") {
		t.Fatalf("want a not-cancelled error, got %v", err)
	}
	for _, call := range readLog(t, log) {
		if strings.HasPrefix(call, "scancel") {
			t.Fatalf("scancel ran although the reason was not recorded: %v", readLog(t, log))
		}
	}
}

func TestDrainArgv(t *testing.T) {
	dir := t.TempDir()
	log := filepath.Join(dir, "calls")
	c := NewCLIController(recorder(t, dir, "scancel", log, 0), recorder(t, dir, "scontrol", log, 0))
	if err := c.Drain(context.Background(), "gpu001", "gpu-reaper: hung job 7"); err != nil {
		t.Fatal(err)
	}
	if got := readLog(t, log); !reflect.DeepEqual(got, []string{"scontrol update NodeName=gpu001 State=DRAIN Reason=gpu-reaper: hung job 7"}) {
		t.Fatalf("calls: %v", got)
	}
}

func TestControllerRefusesUnsafeArguments(t *testing.T) {
	dir := t.TempDir()
	log := filepath.Join(dir, "calls")
	c := NewCLIController(recorder(t, dir, "scancel", log, 0), recorder(t, dir, "scontrol", log, 0))
	ctx := context.Background()
	if err := c.Cancel(ctx, "1 --user=root", "r"); err == nil {
		t.Error("job id with spaces must be refused")
	}
	if err := c.Cancel(ctx, "1", "line\nbreak"); err == nil {
		t.Error("reason with a newline must be refused")
	}
	if err := c.Drain(ctx, "gpu001 State=RESUME", "r"); err == nil {
		t.Error("node with spaces must be refused")
	}
	if err := c.Drain(ctx, "-gpu", "r"); err == nil {
		t.Error("node starting with '-' must be refused")
	}
	if calls := readLog(t, log); len(calls) != 0 {
		t.Fatalf("nothing should have been executed: %v", calls)
	}
}

func TestNoopControllerDoesNothing(t *testing.T) {
	var c Controller = NoopController{}
	if c.Cancel(context.Background(), "1", "r") != nil || c.Drain(context.Background(), "n", "r") != nil {
		t.Fatal("noop controller must not fail")
	}
}

// ── Fuzz ───────────────────────────────────────────────────────────────────

func FuzzExpandNodeList(f *testing.F) {
	// The last two seeds are the overflow cases; CI runs only the seeds, so
	// they are here as well as in TestExpandNodeListRejectsMalformed.
	for _, s := range []string{"gpu[001-004,007]", "rack[1-2]-gpu[01-02]", "a,b", "gpu[01-02].cluster", "x[", "[]",
		"gpu[0-9223372036854775807]", "gpu[9223372036854775807]"} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		nodes, err := ExpandNodeList(s)
		if err != nil {
			return
		}
		if len(nodes) > maxExpandedNodes {
			t.Fatalf("expansion exceeded cap: %d", len(nodes))
		}
		// Expansion only concatenates checked literals and digits, so no
		// output can carry a bracket, comma, space or '=' into an argv.
		for _, n := range nodes {
			if n == "" || !literalRe.MatchString(n) {
				t.Fatalf("unsafe node name %q from %q", n, s)
			}
		}
	})
}

func FuzzParseGPUCount(f *testing.F) {
	for _, s := range []string{"gres/gpu=8", "gres/gpu:a100=4,gres/gpu=4", "gres/gpumem=1G"} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		if n := ParseGPUCount(s); n < 0 {
			t.Fatalf("negative GPU count %d for %q", n, s)
		}
	})
}

func FuzzParseSqueue(f *testing.F) {
	f.Add(goodLine)
	f.Add("7|mallory|r|gpu|n|RUNNING|2026-09-26T08:00:00|gpu003|gres/gpu=4|a|b|c")
	f.Fuzz(func(t *testing.T, s string) {
		for _, j := range ParseSqueue(s, nil) {
			if j.State != "RUNNING" || j.StartTime.IsZero() || !jobIDRe.MatchString(j.JobID) {
				t.Fatalf("invalid job accepted: %+v", j)
			}
			// No free-text field other than Name may contain the delimiter.
			for _, v := range []string{j.User, j.Account, j.Partition, j.QOS} {
				if strings.Contains(v, "|") {
					t.Fatalf("delimiter leaked into a structured field: %+v", j)
				}
			}
		}
	})
}
