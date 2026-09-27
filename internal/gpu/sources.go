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
// Not the richest option (DCGM exposes more), but it needs nothing installed
// beyond the driver, which makes it the backend that works on every GPU node
// without a deployment project first. Each Sample runs nvidia-smi twice: one
// device query and one compute-process query. Its latency has not been
// measured here, because no GPU was available to measure it on.
//
// It is a per-node backend: it reads the local GPUs only and ignores the node
// list it is given.
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

// smiQuery asks for everything in one invocation, including the UUID (to join
// the process list) and the MIG mode. mig.mode.current is the property NVIDIA's
// MIG user guide queries with --query-gpu; the nvidia-smi manual documents its
// values as NA, Enabled or Disabled. A driver too old to know the property
// makes nvidia-smi exit non-zero, which fails the cycle rather than guessing.
const smiQuery = "index,uuid,utilization.gpu,memory.used,memory.total,power.draw,mig.mode.current"

const smiColumns = 7

func (s *SMISource) Sample(ctx context.Context, _ []string) ([]Sample, error) {
	if _, err := exec.LookPath(s.Binary); err != nil {
		return nil, fmt.Errorf("%w: %s not found", ErrUnavailable, s.Binary)
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

	// Process enumeration failing does not stop collection, but it does make
	// every sample's process list unknown. Unknown is not "no processes": it
	// can never satisfy the idle signature.
	pids, pidErr := s.computePIDs(ctx)

	return parseSMI(out, s.NodeName, time.Now(), pids, pidErr == nil)
}

// parseSMI turns nvidia-smi CSV into samples. A malformed row fails the whole
// call: silently dropping one GPU would let the engine judge a job on the
// GPUs that happened to parse, and the missing one might be the busy one.
func parseSMI(out []byte, node string, now time.Time, pids map[string][]int, pidsKnown bool) ([]Sample, error) {
	r := csv.NewReader(strings.NewReader(string(out)))
	r.TrimLeadingSpace = true
	r.FieldsPerRecord = -1
	rows, err := r.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("parse nvidia-smi csv: %w", err)
	}

	samples := make([]Sample, 0, len(rows))
	for i, row := range rows {
		if len(row) != smiColumns {
			return nil, fmt.Errorf("nvidia-smi row %d: want %d columns, got %d", i+1, smiColumns, len(row))
		}
		idx, err := strconv.Atoi(strings.TrimSpace(row[0]))
		if err != nil {
			return nil, fmt.Errorf("nvidia-smi row %d: bad index %q", i+1, row[0])
		}
		uuid := strings.TrimSpace(row[1])

		smp := Sample{
			Timestamp: now,
			NodeName:  node,
			GPUIndex:  idx,
			UUID:      uuid,
		}
		var notes []string

		if v, ok := parseReading(row[2], 0, 100); ok {
			smp.SMUtilPct, smp.UtilKnown = v, true
		} else {
			notes = append(notes, "utilization.gpu="+strings.TrimSpace(row[2]))
		}

		used, okU := parseReading(row[3], 0, math.MaxUint32)
		total, okT := parseReading(row[4], 0, math.MaxUint32)
		if okU && okT && total > 0 && used <= total {
			smp.MemUsedBytes = uint64(used * 1024 * 1024)
			smp.MemTotalBytes = uint64(total * 1024 * 1024)
			smp.MemKnown = true
		} else {
			notes = append(notes, fmt.Sprintf("memory.used=%s memory.total=%s",
				strings.TrimSpace(row[3]), strings.TrimSpace(row[4])))
		}

		if v, ok := parseReading(row[5], 0, 1e5); ok {
			smp.PowerWatts, smp.PowerKnown = v, true
		} else {
			notes = append(notes, "power.draw="+strings.TrimSpace(row[5]))
		}

		// "Disabled" and N/A (a GPU without MIG support) are the only values
		// that mean utilization is device-level and meaningful. Anything else,
		// including an error string, is treated as MIG so the GPU is not judged.
		switch mig := strings.TrimSpace(row[6]); {
		case strings.EqualFold(mig, "Disabled"), isNA(mig):
		default:
			smp.MIG = true
			notes = append(notes, "mig.mode.current="+mig)
		}

		if pidsKnown {
			smp.PIDsKnown = true
			smp.PIDs = pids[uuid]
		} else {
			notes = append(notes, "compute-apps query failed")
		}
		smp.Note = strings.Join(notes, "; ")
		samples = append(samples, smp)
	}
	return samples, nil
}

func (s *SMISource) computePIDs(ctx context.Context) (map[string][]int, error) {
	out, err := exec.CommandContext(ctx, s.Binary,
		"--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil, err
	}
	return parseComputeApps(out)
}

// parseComputeApps maps GPU UUID to resident PIDs. Any line it cannot parse
// makes the whole list unknown rather than silently shorter.
func parseComputeApps(out []byte) (map[string][]int, error) {
	res := map[string][]int{}
	text := strings.TrimSpace(string(out))
	if text == "" {
		return res, nil
	}
	for _, line := range strings.Split(text, "\n") {
		parts := strings.Split(line, ",")
		if len(parts) != 2 {
			return nil, fmt.Errorf("compute-apps line %q: want 2 fields", line)
		}
		pid, err := strconv.Atoi(strings.TrimSpace(parts[1]))
		if err != nil || pid <= 0 {
			return nil, fmt.Errorf("compute-apps line %q: bad pid", line)
		}
		uuid := strings.TrimSpace(parts[0])
		res[uuid] = append(res[uuid], pid)
	}
	return res, nil
}

// parseReading parses one numeric nvidia-smi field. It returns ok=false for
// anything that is not a finite number inside [lo, hi]: "[N/A]", "N/A",
// "[Not Supported]", "[Unknown Error]", "[Insufficient Permissions]", an
// empty field, or garbage. Callers treat ok=false as unknown, never as zero.
func parseReading(s string, lo, hi float64) (float64, bool) {
	s = strings.TrimSpace(s)
	if s == "" || strings.HasPrefix(s, "[") || isNA(s) {
		return 0, false
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || math.IsNaN(f) || math.IsInf(f, 0) || f < lo || f > hi {
		return 0, false
	}
	return f, true
}

func isNA(s string) bool {
	s = strings.TrimSpace(s)
	return strings.EqualFold(s, "[N/A]") || strings.EqualFold(s, "N/A") || strings.EqualFold(s, "NA")
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
	// ScenarioUnreadable reports utilization and power as unreadable, the way
	// nvidia-smi prints [N/A]. It exists to show the engine refusing to judge.
	ScenarioUnreadable Scenario = "unreadable"
)

// SimSource generates telemetry without hardware.
//
// This is what makes the project reviewable: anyone can clone the repo on a
// laptop, run `make demo`, and watch the policy engine escalate a fake hung job
// through alert and drain.
type SimSource struct {
	NodeName string
	GPUCount int
	Scenario Scenario
	rng      *rand.Rand
	tick     int
}

func NewSimSource(node string, gpus int, sc Scenario, seed int64) *SimSource {
	return &SimSource{NodeName: node, GPUCount: gpus, Scenario: sc, rng: rand.New(rand.NewSource(seed))}
}

func (s *SimSource) Name() string { return "simulator/" + string(s.Scenario) }
func (s *SimSource) Close() error { return nil }

// Sample emits telemetry for every node it is asked about, so one simulator
// stands in for the whole fleet (a central backend). Falls back to its
// configured node name when the caller passes none.
func (s *SimSource) Sample(_ context.Context, nodes []string) ([]Sample, error) {
	targets := dedupe(nodes)
	if len(targets) == 0 {
		targets = []string{s.NodeName}
	}

	s.tick++
	// One timestamp per cycle, the way one nvidia-smi invocation stamps all
	// of a node's GPUs at once.
	now := time.Now()
	out := make([]Sample, 0, len(targets)*s.GPUCount)
	for _, node := range targets {
		samples, err := s.forNode(node, now)
		if err != nil {
			return nil, err
		}
		out = append(out, samples...)
	}
	return out, nil
}

func (s *SimSource) forNode(node string, now time.Time) ([]Sample, error) {
	const totalMem = 80 * 1024 * 1024 * 1024

	out := make([]Sample, 0, s.GPUCount)
	for i := 0; i < s.GPUCount; i++ {
		var util, memFrac, power float64
		var procs int
		utilKnown, powerKnown := true, true

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
			// Narrow spread: with 8 GPUs per node sampled every cycle, a wider
			// one regularly produces a sample above the 15% threshold, and by
			// the peak rule that job is (correctly) alive, not starved.
			util = clamp(7+s.rng.NormFloat64()*1.5, 1, 100)
			memFrac, power, procs = 0.81, 130+s.rng.NormFloat64()*15, 1
		case ScenarioFlaky:
			// A burst every ~12 ticks — exercises the peak-clears-the-finding
			// path when the window spans a burst.
			if s.tick%12 == 0 {
				util, memFrac, power, procs = 95, 0.8, 360, 1
			} else {
				util, memFrac, power, procs = 1, 0.8, 90, 1
			}
		case ScenarioUnreadable:
			memFrac, procs = 0.5, 1
			utilKnown, powerKnown = false, false
		default:
			return nil, fmt.Errorf("unknown scenario %q", s.Scenario)
		}

		smp := Sample{
			Timestamp:     now,
			NodeName:      node,
			GPUIndex:      i,
			MemUsedBytes:  uint64(memFrac * totalMem),
			MemTotalBytes: totalMem,
			PIDs:          makePIDs(procs, i),
			MemKnown:      true,
			PIDsKnown:     true,
		}
		if utilKnown {
			smp.SMUtilPct, smp.UtilKnown = util, true
		} else {
			smp.Note = "simulated [N/A] utilization"
		}
		if powerKnown {
			smp.PowerWatts, smp.PowerKnown = math.Max(0, power), true
		}
		out = append(out, smp)
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
