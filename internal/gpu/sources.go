package gpu

import (
	"context"
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// ── nvidia-smi ─────────────────────────────────────────────────────────────

// SMISource reads telemetry by shelling out to nvidia-smi.
//
// Not the fastest option — DCGM is — but it needs nothing installed beyond the
// driver, which makes it the backend that works on every GPU node without a
// deployment project first. Query cost is a few milliseconds per node.
type SMISource struct {
	Binary   string
	NodeName string
	Timeout  time.Duration
}

func NewSMISource(nodeName string) *SMISource {
	return &SMISource{Binary: "nvidia-smi", NodeName: nodeName, Timeout: 10 * time.Second}
}

func (s *SMISource) Name() string { return "nvidia-smi" }
func (s *SMISource) Close() error { return nil }

const smiQuery = "index,utilization.gpu,memory.used,memory.total,power.draw"

func (s *SMISource) Sample(ctx context.Context, _ []string) ([]Sample, error) {
	if _, err := exec.LookPath(s.Binary); err != nil {
		return nil, fmt.Errorf("%w: %s not on PATH", ErrUnavailable, s.Binary)
	}

	ctx, cancel := context.WithTimeout(ctx, s.Timeout)
	defer cancel()

	out, err := exec.CommandContext(ctx, s.Binary,
		"--query-gpu="+smiQuery,
		"--format=csv,noheader,nounits",
	).Output()
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi query: %w", err)
	}

	pids, err := s.computePIDs(ctx)
	if err != nil {
		// Process enumeration is a nice-to-have. Losing it degrades the hung vs
		// idle distinction but should not stop collection.
		pids = map[int][]int{}
	}

	now := time.Now()
	var samples []Sample
	r := csv.NewReader(strings.NewReader(string(out)))
	r.TrimLeadingSpace = true
	rows, err := r.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("parse nvidia-smi csv: %w", err)
	}

	for _, row := range rows {
		if len(row) < 5 {
			continue
		}
		idx := atoiSafe(row[0])
		samples = append(samples, Sample{
			Timestamp:     now,
			NodeName:      s.NodeName,
			GPUIndex:      idx,
			SMUtilPct:     atofSafe(row[1]),
			MemUsedBytes:  uint64(atofSafe(row[2]) * 1024 * 1024),
			MemTotalBytes: uint64(atofSafe(row[3]) * 1024 * 1024),
			PowerWatts:    atofSafe(row[4]),
			PIDs:          pids[idx],
		})
	}
	return samples, nil
}

func (s *SMISource) computePIDs(ctx context.Context) (map[int][]int, error) {
	out, err := exec.CommandContext(ctx, s.Binary,
		"--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil, err
	}
	// gpu_uuid does not map to index without a second lookup; for the purposes
	// of "are there processes at all" a flat list per index-0 bucket is enough
	// when only one GPU is visible, so resolve indices explicitly instead.
	uuidToIdx, err := s.uuidIndex(ctx)
	if err != nil {
		return nil, err
	}

	res := map[int][]int{}
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		parts := strings.Split(line, ",")
		if len(parts) < 2 {
			continue
		}
		uuid := strings.TrimSpace(parts[0])
		pid := atoiSafe(parts[1])
		if idx, ok := uuidToIdx[uuid]; ok && pid > 0 {
			res[idx] = append(res[idx], pid)
		}
	}
	return res, nil
}

func (s *SMISource) uuidIndex(ctx context.Context) (map[string]int, error) {
	out, err := exec.CommandContext(ctx, s.Binary,
		"--query-gpu=uuid,index", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil, err
	}
	res := map[string]int{}
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		parts := strings.Split(line, ",")
		if len(parts) < 2 {
			continue
		}
		res[strings.TrimSpace(parts[0])] = atoiSafe(parts[1])
	}
	return res, nil
}

// ── Simulator ──────────────────────────────────────────────────────────────

// Scenario names a synthetic GPU behaviour.
type Scenario string

const (
	ScenarioHealthy Scenario = "healthy" // saturated, as a good run looks
	ScenarioIdle    Scenario = "idle"    // allocation held, nothing running
	ScenarioHung    Scenario = "hung"    // memory held, zero compute
	ScenarioStarved Scenario = "starved" // low compute, dataloader-bound
	ScenarioFlaky   Scenario = "flaky"   // mostly idle with occasional bursts
)

// SimSource generates telemetry without hardware.
//
// This is what makes the project reviewable: anyone can clone the repo on a
// laptop, run `make demo`, and watch the policy engine escalate a fake hung job
// through alert and drain. Reviewers who cannot run your code do not read it.
type SimSource struct {
	NodeName  string
	GPUCount  int
	Scenario  Scenario
	rng       *rand.Rand
	tick      int
}

func NewSimSource(node string, gpus int, sc Scenario, seed int64) *SimSource {
	return &SimSource{NodeName: node, GPUCount: gpus, Scenario: sc, rng: rand.New(rand.NewSource(seed))}
}

func (s *SimSource) Name() string { return "simulator/" + string(s.Scenario) }
func (s *SimSource) Close() error { return nil }

// Sample emits telemetry for every node it is asked about, so one simulator
// stands in for the whole fleet. Falls back to its configured node name when
// the caller passes none.
func (s *SimSource) Sample(_ context.Context, nodes []string) ([]Sample, error) {
	targets := dedupe(nodes)
	if len(targets) == 0 {
		targets = []string{s.NodeName}
	}

	s.tick++
	out := make([]Sample, 0, len(targets)*s.GPUCount)
	for _, node := range targets {
		samples, err := s.forNode(node)
		if err != nil {
			return nil, err
		}
		out = append(out, samples...)
	}
	return out, nil
}

func (s *SimSource) forNode(node string) ([]Sample, error) {
	const totalMem = 80 * 1024 * 1024 * 1024
	now := time.Now()

	out := make([]Sample, 0, s.GPUCount)
	for i := 0; i < s.GPUCount; i++ {
		var util, memFrac, power float64
		var procs int

		switch s.Scenario {
		case ScenarioHealthy:
			util = clamp(92+s.rng.NormFloat64()*6, 0, 100)
			memFrac, power, procs = 0.86, 380+s.rng.NormFloat64()*30, 1
		case ScenarioIdle:
			util = 0
			memFrac, power, procs = 0, 24+s.rng.NormFloat64()*3, 0
		case ScenarioHung:
			util = 0
			memFrac, power, procs = 0.74, 88+s.rng.NormFloat64()*5, 2
		case ScenarioStarved:
			util = clamp(7+s.rng.NormFloat64()*3, 0, 100)
			memFrac, power, procs = 0.81, 130+s.rng.NormFloat64()*15, 1
		case ScenarioFlaky:
			// A burst every ~12 ticks — exercises the peak-clears-the-finding path.
			if s.tick%12 == 0 {
				util, memFrac, power, procs = 95, 0.8, 360, 1
			} else {
				util, memFrac, power, procs = 1, 0.8, 90, 1
			}
		default:
			return nil, fmt.Errorf("unknown scenario %q", s.Scenario)
		}

		out = append(out, Sample{
			Timestamp:     now,
			NodeName:      node,
			GPUIndex:      i,
			SMUtilPct:     util,
			MemUsedBytes:  uint64(memFrac * totalMem),
			MemTotalBytes: totalMem,
			PowerWatts:    math.Max(0, power),
			PIDs:          makePIDs(procs, i),
		})
	}
	return out, nil
}

func dedupe(in []string) []string {
	seen := make(map[string]bool, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		if s != "" && !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}

func makePIDs(n, gpu int) []int {
	if n == 0 {
		return nil
	}
	pids := make([]int, n)
	for i := range pids {
		pids[i] = 40000 + gpu*10 + i
	}
	return pids
}

func clamp(v, lo, hi float64) float64 { return math.Max(lo, math.Min(hi, v)) }

func atoiSafe(s string) int {
	n, _ := strconv.Atoi(strings.TrimSpace(s))
	return n
}

func atofSafe(s string) float64 {
	s = strings.TrimSpace(s)
	if s == "" || strings.EqualFold(s, "[N/A]") || strings.EqualFold(s, "N/A") {
		return 0
	}
	f, _ := strconv.ParseFloat(s, 64)
	return f
}
