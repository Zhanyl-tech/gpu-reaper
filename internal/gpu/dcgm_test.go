package gpu

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// All DCGM tests run against the synthetic fixtures in testdata/ (see
// testdata/README.md). None of this has been run against a real exporter.

var scrapeTime = time.Date(2026, 9, 26, 12, 0, 0, 0, time.UTC)

func parseFixture(t *testing.T, name string, maxAge time.Duration) []Sample {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	got, err := ParseDCGM(f, "gpu001", scrapeTime, maxAge)
	if err != nil {
		t.Fatalf("ParseDCGM(%s): %v", name, err)
	}
	return got
}

func TestDCGMParsesDefaultCounters(t *testing.T) {
	got := parseFixture(t, "dcgm_basic.prom", 0)
	if len(got) != 2 {
		t.Fatalf("want 2 GPUs, got %d", len(got))
	}
	g0, g1 := got[0], got[1]
	if g0.GPUIndex != 0 || g0.UUID != "GPU-aaaaaaaa-0000-0000-0000-000000000000" || g0.NodeName != "gpu001" {
		t.Fatalf("gpu0 identity wrong: %+v", g0)
	}
	if !g0.Judgeable() || !g0.PowerKnown {
		t.Fatalf("gpu0 should be readable: %+v", g0)
	}
	// max(GPU_UTIL 97, GR_ENGINE_ACTIVE 0.91*100) = 97
	if g0.SMUtilPct != 97 {
		t.Fatalf("gpu0 util = %v, want 97", g0.SMUtilPct)
	}
	// used 69000 / (69000 + 11000 free + 500 reserved) MiB
	if f := g0.MemUsedFraction(); f < 0.857 || f > 0.858 {
		t.Fatalf("gpu0 mem fraction = %.4f", f)
	}
	if g1.SMUtilPct != 0 || !g1.UtilKnown || g1.PowerWatts != 71.25 {
		t.Fatalf("gpu1 values wrong: %+v", g1)
	}
	for _, s := range got {
		if s.PIDsKnown {
			t.Fatal("dcgm-exporter exposes no process data; PIDs must stay unknown")
		}
		if !s.Timestamp.Equal(scrapeTime) {
			t.Fatalf("untimestamped series take the scrape time, got %v", s.Timestamp)
		}
	}
}

// The DCGM path must fail safe exactly like the fixed nvidia-smi path: a
// value dcgm-exporter skipped (blank or non-OK in DCGM) is unknown, not 0.
func TestDCGMMissingUtilizationIsUnknownNotIdle(t *testing.T) {
	got := parseFixture(t, "dcgm_missing_util.prom", 0)
	g1 := got[1]
	if g1.UtilKnown || g1.Judgeable() {
		t.Fatalf("GPU with no activity series must be unjudgeable: %+v", g1)
	}
	if !g1.MemKnown || !g1.PowerKnown {
		t.Fatalf("memory and power were present and should be known: %+v", g1)
	}
	if !got[0].Judgeable() {
		t.Fatal("the other GPU is unaffected")
	}
}

func TestDCGMRejectsBadValues(t *testing.T) {
	got := parseFixture(t, "dcgm_bad_values.prom", 0)
	g0, g1 := got[0], got[1]
	if g0.UtilKnown {
		t.Errorf("utilization of 150%% is out of range and must be unknown: %+v", g0)
	}
	if g0.PowerKnown {
		t.Errorf("+Inf power must be unknown")
	}
	if g1.UtilKnown {
		t.Errorf("NaN utilization must be unknown")
	}
	if g1.MemKnown {
		t.Errorf("negative framebuffer use must make memory unknown")
	}
	if !g1.PowerKnown {
		t.Errorf("a valid power reading should survive its siblings being bad")
	}
}

func TestDCGMMIGGPUsAreNotJudgeable(t *testing.T) {
	got := parseFixture(t, "dcgm_mig.prom", 0)
	if len(got) != 1 || !got[0].MIG || got[0].Judgeable() {
		t.Fatalf("GPU with MIG instance labels must be refused: %+v", got)
	}
}

func TestDCGMStaleTimestampedValuesAreUnknown(t *testing.T) {
	// Fixture: utilization stamped 1790000000000 ms, the rest 10 minutes later.
	at := time.UnixMilli(1790000600000)
	f, _ := os.Open(filepath.Join("testdata", "dcgm_timestamps.prom"))
	defer f.Close()
	got, err := ParseDCGM(f, "gpu001", at, 2*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	g := got[0]
	if g.UtilKnown {
		t.Fatalf("a utilization value 10m older than max_age must be unknown: %+v", g)
	}
	if !g.MemKnown || !g.PowerKnown {
		t.Fatalf("fresh series should stay known: %+v", g)
	}
	if !g.Timestamp.Equal(at) {
		t.Fatalf("sample time should be the newest series timestamp, got %v", g.Timestamp)
	}

	// Scraped far later, every series is stale.
	f2, _ := os.Open(filepath.Join("testdata", "dcgm_timestamps.prom"))
	defer f2.Close()
	got, err = ParseDCGM(f2, "gpu001", at.Add(time.Hour), 2*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if got[0].UtilKnown || got[0].MemKnown || got[0].PowerKnown {
		t.Fatalf("an hour-old page must be entirely unknown: %+v", got[0])
	}
}

func TestDCGMGroupsHPCJobClonesByGPU(t *testing.T) {
	got := parseFixture(t, "dcgm_hpc_job.prom", 0)
	if len(got) != 2 {
		t.Fatalf("clones of one GPU must collapse to one sample, got %d", len(got))
	}
	if !reflect.DeepEqual(got[0].Jobs, []string{"4242", "4243"}) {
		t.Fatalf("gpu0 jobs = %v", got[0].Jobs)
	}
	if !reflect.DeepEqual(got[1].Jobs, []string{"4242"}) {
		t.Fatalf("gpu1 jobs = %v", got[1].Jobs)
	}
	if !got[0].Judgeable() || got[0].SMUtilPct != 0 {
		t.Fatalf("identical clones should read normally: %+v", got[0])
	}
}

func TestDCGMDisagreeingClonesAreUnknown(t *testing.T) {
	got := parseFixture(t, "dcgm_conflict.prom", 0)
	if got[0].UtilKnown {
		t.Fatalf("clones of one GPU reporting 0 and 88 must be unknown, not either value: %+v", got[0])
	}
}

func TestDCGMKubernetesLabelsAreCarried(t *testing.T) {
	got := parseFixture(t, "dcgm_k8s.prom", 0)
	if !reflect.DeepEqual(got[0].Pods, []string{"slurm/gpu-2-1"}) {
		t.Fatalf("pods = %v", got[0].Pods)
	}
	if got[0].PowerKnown {
		t.Fatal("no power series in this fixture; power must be unknown")
	}
}

func TestDCGMProfilingOnlyUsesTheLargestActivity(t *testing.T) {
	got := parseFixture(t, "dcgm_prof_only.prom", 0)
	// max(GR_ENGINE 0.42, SM_ACTIVE 0.05) * 100
	if !got[0].UtilKnown || got[0].SMUtilPct < 41.99 || got[0].SMUtilPct > 42.01 {
		t.Fatalf("util = %v, want 42", got[0].SMUtilPct)
	}
}

func TestDCGMMalformedPagesFail(t *testing.T) {
	for _, name := range []string{"dcgm_malformed.prom", "dcgm_no_gpu_label.prom"} {
		f, _ := os.Open(filepath.Join("testdata", name))
		_, err := ParseDCGM(f, "gpu001", scrapeTime, 0)
		f.Close()
		if err == nil {
			t.Errorf("%s: a page that cannot be fully read must fail, not half-parse", name)
		}
	}
}

func serveFixture(t *testing.T, byNode map[string]string) (*httptest.Server, *int32) {
	t.Helper()
	var hits int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		node := strings.TrimSuffix(r.URL.Path, "/metrics")
		node = strings.TrimPrefix(node, "/")
		fixture, ok := byNode[node]
		if !ok {
			http.Error(w, "no exporter here", http.StatusServiceUnavailable)
			return
		}
		b, err := os.ReadFile(filepath.Join("testdata", fixture))
		if err != nil {
			t.Error(err)
		}
		_, _ = w.Write(b)
	}))
	t.Cleanup(srv.Close)
	return srv, &hits
}

func TestDCGMCentralModeScrapesEachNode(t *testing.T) {
	srv, hits := serveFixture(t, map[string]string{"gpu001": "dcgm_basic.prom", "gpu002": "dcgm_basic.prom"})
	src := NewDCGMSource(srv.URL+"/{node}/metrics", "", 2*time.Second, 0)
	if !src.Central() {
		t.Fatal("a {node} URL is central")
	}
	got, err := src.Sample(context.Background(), []string{"gpu002", "gpu001", "gpu001"})
	if err != nil {
		t.Fatal(err)
	}
	if atomic.LoadInt32(hits) != 2 {
		t.Fatalf("want one scrape per distinct node, got %d", *hits)
	}
	if len(got) != 4 || got[0].NodeName != "gpu001" || got[3].NodeName != "gpu002" {
		t.Fatalf("samples should be attributed to the node scraped: %+v", got)
	}
}

func TestDCGMCentralModeReportsPartialFailure(t *testing.T) {
	srv, _ := serveFixture(t, map[string]string{"gpu001": "dcgm_basic.prom"})
	src := NewDCGMSource(srv.URL+"/{node}/metrics", "", 2*time.Second, 0)
	got, err := src.Sample(context.Background(), []string{"gpu001", "gpu002", "bad/name"})
	var pe *PartialError
	if !errors.As(err, &pe) {
		t.Fatalf("want *PartialError, got %v", err)
	}
	if _, ok := pe.Failed["gpu002"]; !ok {
		t.Fatalf("gpu002 should be reported failed: %v", pe)
	}
	if _, ok := pe.Failed["bad/name"]; !ok {
		t.Fatalf("a node name unsafe for a URL must be refused, not scraped: %v", pe)
	}
	for _, s := range got {
		if s.NodeName != "gpu001" {
			t.Fatalf("only gpu001 should have samples: %+v", s)
		}
	}
}

func TestDCGMEveryEndpointFailingIsAnError(t *testing.T) {
	srv, _ := serveFixture(t, nil)
	src := NewDCGMSource(srv.URL+"/{node}/metrics", "", 2*time.Second, 0)
	_, err := src.Sample(context.Background(), []string{"gpu001"})
	var pe *PartialError
	if err == nil || errors.As(err, &pe) {
		t.Fatalf("all nodes failing must be a hard error, got %v", err)
	}
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("want ErrUnavailable, got %v", err)
	}
}

func TestDCGMPerNodeModeUsesConfiguredNodeName(t *testing.T) {
	srv, _ := serveFixture(t, map[string]string{"": "dcgm_basic.prom"})
	src := NewDCGMSource(srv.URL+"/metrics", "gpu-2-1", 2*time.Second, 0)
	if src.Central() {
		t.Fatal("a fixed URL is per-node")
	}
	got, err := src.Sample(context.Background(), []string{"ignored"})
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range got {
		if s.NodeName != "gpu-2-1" {
			t.Fatalf("per-node samples take the configured node name: %+v", s)
		}
	}
}

func TestDCGMNon200IsAnError(t *testing.T) {
	srv, _ := serveFixture(t, nil)
	src := NewDCGMSource(srv.URL+"/metrics", "gpu001", 2*time.Second, 0)
	if _, err := src.Sample(context.Background(), nil); err == nil {
		t.Fatal("a 503 from the exporter must fail the sample")
	}
}
