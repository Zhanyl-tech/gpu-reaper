package gpu

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"
)

// ── dcgm-exporter ──────────────────────────────────────────────────────────
//
// DCGMSource reads NVIDIA dcgm-exporter's Prometheus text endpoint.
//
// NOT VERIFIED ON REAL HARDWARE. Every behaviour here is tested only against
// recorded text fixtures written to match dcgm-exporter's documented output.
// The metric and label names were checked against these upstream sources
// (dcgm-exporter main at commit fafd151, 2026-09-18):
//
//   - etc/default-counters.csv lists DCGM_FI_DEV_GPU_UTIL "GPU utilization
//     (in %)", DCGM_FI_DEV_FB_USED / _FB_FREE / _FB_RESERVED "(in MiB)",
//     DCGM_FI_DEV_POWER_USAGE "Power draw (in W)", and
//     DCGM_FI_PROF_GR_ENGINE_ACTIVE "Ratio of time the graphics engine is
//     active". DCGM_FI_PROF_SM_ACTIVE ("The ratio of cycles an SM has at least
//     1 warp assigned") is listed but commented out by default.
//   - internal/pkg/rendermetrics/render_metrics.go labels GPU metrics with
//     gpu, UUID, pci_bus_id, device, modelName, hostname, and GPU_I_PROFILE /
//     GPU_I_ID for MIG instances. It sets no sample timestamps.
//   - internal/pkg/collector/gpu_collector.go (toString) skips values whose
//     DCGM status is not OK and DCGM "blank" values, so an unreadable field is
//     an absent series rather than a number.
//   - internal/pkg/transformation/const.go names the pod, namespace, container
//     and hpc_job labels; hpc.go clones a GPU's series once per mapped job,
//     each clone carrying one hpc_job value.
//   - https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html
//     gives the default listen address :9400 and path /metrics.
//
// dcgm-exporter exposes no per-process data. Process residency is therefore
// always unknown with this backend, and because both actionable signatures
// (idle: "no processes"; hung: "processes resident") need that evidence, a
// DCGM-backed finding can reach Alert but is classified starved or unknown and
// never escalates past it.
type DCGMSource struct {
	// URL is the metrics endpoint. If it contains "{node}", the source is
	// central: it scrapes one endpoint per Slurm node, substituting the node
	// name. Otherwise it scrapes exactly this URL and attributes every GPU to
	// NodeName (a per-node deployment).
	URL      string
	NodeName string
	Timeout  time.Duration
	// MaxAge bounds how old a sample may be when the exposition carries
	// timestamps. dcgm-exporter itself sets none, so this applies only when
	// something in between (a proxy, a recorded fixture) adds them.
	MaxAge time.Duration
	// Parallel bounds concurrent scrapes in central mode.
	Parallel int
	Client   *http.Client

	now func() time.Time
}

// NewDCGMSource builds a source for the given endpoint (see DCGMSource.URL).
func NewDCGMSource(url, nodeName string, timeout, maxAge time.Duration) *DCGMSource {
	return &DCGMSource{
		URL: url, NodeName: nodeName, Timeout: timeout, MaxAge: maxAge,
		Parallel: 16,
		Client:   &http.Client{Timeout: timeout},
		now:      time.Now,
	}
}

func (d *DCGMSource) Name() string { return "dcgm-exporter" }
func (d *DCGMSource) Close() error { return nil }

// Central reports whether the source scrapes one endpoint per node.
func (d *DCGMSource) Central() bool { return strings.Contains(d.URL, "{node}") }

// Metric names read by this source.
const (
	dcgmGPUUtil        = "DCGM_FI_DEV_GPU_UTIL"          // percent
	dcgmGREngineActive = "DCGM_FI_PROF_GR_ENGINE_ACTIVE" // ratio 0..1
	dcgmSMActive       = "DCGM_FI_PROF_SM_ACTIVE"        // ratio 0..1
	dcgmFBUsed         = "DCGM_FI_DEV_FB_USED"           // MiB
	dcgmFBFree         = "DCGM_FI_DEV_FB_FREE"           // MiB
	dcgmFBReserved     = "DCGM_FI_DEV_FB_RESERVED"       // MiB
	dcgmPower          = "DCGM_FI_DEV_POWER_USAGE"       // W
)

// Node names are substituted into a URL, so only plain hostname characters are
// accepted. A crafted node list must not be able to steer the scrape.
var safeNodeName = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*$`)

func (d *DCGMSource) Sample(ctx context.Context, nodes []string) ([]Sample, error) {
	if !d.Central() {
		out, err := d.scrape(ctx, d.URL, d.NodeName)
		if err != nil {
			return nil, fmt.Errorf("%w: %v", ErrUnavailable, err)
		}
		return out, nil
	}

	targets := dedupe(nodes)
	if len(targets) == 0 {
		return nil, nil
	}
	par := d.Parallel
	if par < 1 {
		par = 1
	}

	var (
		mu     sync.Mutex
		wg     sync.WaitGroup
		out    []Sample
		failed = map[string]error{}
		sem    = make(chan struct{}, par)
	)
	for _, node := range targets {
		if !safeNodeName.MatchString(node) {
			mu.Lock()
			failed[node] = errors.New("node name has characters not allowed in a URL host")
			mu.Unlock()
			continue
		}
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			s, err := d.scrape(ctx, strings.ReplaceAll(d.URL, "{node}", node), node)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				failed[node] = err
				return
			}
			out = append(out, s...)
		}()
	}
	wg.Wait()

	if len(failed) == len(targets) {
		return nil, fmt.Errorf("%w: every endpoint failed: %v", ErrUnavailable, &PartialError{Failed: failed})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].NodeName != out[j].NodeName {
			return out[i].NodeName < out[j].NodeName
		}
		return out[i].GPUIndex < out[j].GPUIndex
	})
	if len(failed) > 0 {
		return out, &PartialError{Failed: failed}
	}
	return out, nil
}

func (d *DCGMSource) scrape(ctx context.Context, url, node string) ([]Sample, error) {
	ctx, cancel := context.WithTimeout(ctx, d.Timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Accept", "text/plain;version=0.0.4")
	resp, err := d.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("scrape: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("scrape returned %s", resp.Status)
	}
	// 16 MiB is far above any plausible exporter page and stops a broken or
	// hostile endpoint from exhausting memory.
	body := io.LimitReader(resp.Body, 16<<20)
	return ParseDCGM(body, node, d.now(), d.MaxAge)
}

// dcgmReading accumulates every series of one metric for one GPU. Several
// series per GPU are normal: dcgm-exporter clones a GPU's series once per
// hpc_job, and once per pod when several pods share a GPU.
type dcgmReading struct {
	values []float64
	valid  bool // every series was finite, in range, and fresh
}

type dcgmGPU struct {
	index    int
	uuid     string
	mig      bool
	readings map[string]*dcgmReading
	jobs     map[string]bool
	pods     map[string]bool
	newest   time.Time
	notes    []string
}

// ParseDCGM converts one dcgm-exporter exposition into samples for one node.
// scrapedAt stamps samples whose series carry no timestamp; maxAge (if > 0)
// marks a timestamped value older than that as unknown.
//
// Any parse error fails the whole page: a half-read page could drop the busy
// GPU and keep the idle ones.
func ParseDCGM(r io.Reader, node string, scrapedAt time.Time, maxAge time.Duration) ([]Sample, error) {
	parser := expfmt.NewTextParser(model.UTF8Validation)
	families, err := parser.TextToMetricFamilies(r)
	if err != nil {
		return nil, fmt.Errorf("parse exposition: %w", err)
	}

	gpus := map[int]*dcgmGPU{}
	for _, name := range []string{dcgmGPUUtil, dcgmGREngineActive, dcgmSMActive, dcgmFBUsed, dcgmFBFree, dcgmFBReserved, dcgmPower} {
		fam, ok := families[name]
		if !ok {
			continue
		}
		for _, m := range fam.GetMetric() {
			labels := map[string]string{}
			for _, lp := range m.GetLabel() {
				labels[lp.GetName()] = lp.GetValue()
			}
			idx, err := strconv.Atoi(labels["gpu"])
			if err != nil || idx < 0 {
				// A series without a usable gpu label cannot be attributed to
				// a device. Fail rather than guess which GPU it belongs to.
				return nil, fmt.Errorf("%s: series without a valid gpu label", name)
			}
			g := gpus[idx]
			if g == nil {
				g = &dcgmGPU{index: idx, readings: map[string]*dcgmReading{},
					jobs: map[string]bool{}, pods: map[string]bool{}}
				gpus[idx] = g
			}
			if u := labels["UUID"]; u != "" {
				g.uuid = u
			}
			if labels["GPU_I_ID"] != "" || labels["GPU_I_PROFILE"] != "" {
				g.mig = true
			}
			if j := labels["hpc_job"]; j != "" {
				g.jobs[j] = true
			}
			if p := labels["pod"]; p != "" {
				g.pods[labels["namespace"]+"/"+p] = true
			}

			v, ok := metricValue(m)
			at := scrapedAt
			if ts := m.GetTimestampMs(); ts != 0 {
				at = time.UnixMilli(ts)
				if maxAge > 0 && (scrapedAt.Sub(at) > maxAge || at.Sub(scrapedAt) > maxAge) {
					ok = false
					g.notes = append(g.notes, name+" stale")
				}
			}
			if at.After(g.newest) {
				g.newest = at
			}
			if ok && !inRange(name, v) {
				ok = false
				g.notes = append(g.notes, fmt.Sprintf("%s out of range (%g)", name, v))
			}

			rd := g.readings[name]
			if rd == nil {
				rd = &dcgmReading{valid: true}
				g.readings[name] = rd
			}
			rd.values = append(rd.values, v)
			rd.valid = rd.valid && ok
		}
	}

	idxs := make([]int, 0, len(gpus))
	for i := range gpus {
		idxs = append(idxs, i)
	}
	sort.Ints(idxs)

	out := make([]Sample, 0, len(idxs))
	for _, i := range idxs {
		out = append(out, gpus[i].sample(node))
	}
	return out, nil
}

// sample reduces one GPU's readings to a Sample. A metric counts as read only
// if every series for it was valid and they agree; clones of one GPU reading
// that disagree mean something is wrong, and wrong is unknown.
func (g *dcgmGPU) sample(node string) Sample {
	s := Sample{
		Timestamp: g.newest,
		NodeName:  node,
		GPUIndex:  g.index,
		UUID:      g.uuid,
		MIG:       g.mig,
		Jobs:      sortedKeys(g.jobs),
		Pods:      sortedKeys(g.pods),
	}
	notes := append([]string(nil), g.notes...)

	get := func(name string) (v float64, present, ok bool) {
		rd := g.readings[name]
		if rd == nil {
			return 0, false, false
		}
		if !rd.valid || !agree(rd.values) {
			return 0, true, false
		}
		return rd.values[0], true, true
	}

	// Utilization: the largest of whichever activity readings exist, each
	// scaled to percent. Taking the maximum can only make a GPU look busier,
	// which is the safe direction for a tool that acts on low readings. A
	// reading that is present but unusable makes the whole figure unknown
	// rather than letting a lower sibling stand in for it.
	var util float64
	var have, broken bool
	for _, m := range []struct {
		name  string
		scale float64
	}{{dcgmGPUUtil, 1}, {dcgmGREngineActive, 100}, {dcgmSMActive, 100}} {
		v, present, ok := get(m.name)
		switch {
		case !present:
		case !ok:
			broken = true
			notes = append(notes, m.name+" unusable")
		default:
			have = true
			util = math.Max(util, v*m.scale)
		}
	}
	if have && !broken {
		s.SMUtilPct, s.UtilKnown = util, true
	} else if !have && !broken {
		notes = append(notes, "no utilization metric")
	}

	used, uPresent, uOK := get(dcgmFBUsed)
	free, fPresent, fOK := get(dcgmFBFree)
	reserved, rPresent, rOK := get(dcgmFBReserved)
	switch {
	case uOK && fOK && (!rPresent || rOK):
		total := used + free
		if rPresent {
			total += reserved
		}
		if total > 0 {
			s.MemUsedBytes = uint64(used * 1024 * 1024)
			s.MemTotalBytes = uint64(total * 1024 * 1024)
			s.MemKnown = true
		} else {
			notes = append(notes, "framebuffer total is zero")
		}
	default:
		notes = append(notes, fmt.Sprintf("framebuffer used/free present=%t/%t", uPresent, fPresent))
	}

	if v, _, ok := get(dcgmPower); ok {
		s.PowerWatts, s.PowerKnown = v, true
	}

	// No process data exists in dcgm-exporter output. PIDsKnown stays false.
	if s.MIG {
		notes = append(notes, "MIG instance labels present")
	}
	if len(notes) > 0 {
		s.Note = strings.Join(notes, "; ")
	}
	return s
}

func metricValue(m *dto.Metric) (float64, bool) {
	var v float64
	switch {
	case m.Gauge != nil:
		v = m.GetGauge().GetValue()
	case m.Untyped != nil:
		v = m.GetUntyped().GetValue()
	case m.Counter != nil:
		v = m.GetCounter().GetValue()
	default:
		return 0, false
	}
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, false
	}
	return v, true
}

// inRange rejects values outside what the metric can physically report. The
// bounds are plausibility limits, not tuning: a utilization above 100%, a
// ratio above 1, or a multi-petabyte framebuffer is a broken reading.
func inRange(name string, v float64) bool {
	if v < 0 {
		return false
	}
	switch name {
	case dcgmGPUUtil:
		return v <= 100
	case dcgmGREngineActive, dcgmSMActive:
		return v <= 1
	case dcgmFBUsed, dcgmFBFree, dcgmFBReserved:
		return v < 1<<24 // MiB; 16 TiB
	case dcgmPower:
		return v < 1e5
	}
	return true
}

func agree(vs []float64) bool {
	if len(vs) == 0 {
		return false
	}
	for _, v := range vs[1:] {
		if v != vs[0] {
			return false
		}
	}
	return len(vs) > 0
}

func sortedKeys(m map[string]bool) []string {
	if len(m) == 0 {
		return nil
	}
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
