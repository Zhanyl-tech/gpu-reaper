package slurm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// ── squeue ─────────────────────────────────────────────────────────────────

// SqueueSource reads running jobs by shelling out to squeue.
//
// Uses an explicit --Format with fixed widths rather than the default output,
// because the default is localised, column-ordered by config, and truncates.
type SqueueSource struct {
	Binary  string
	Timeout time.Duration
}

func NewSqueueSource() *SqueueSource {
	return &SqueueSource{Binary: "squeue", Timeout: 15 * time.Second}
}

func (s *SqueueSource) Name() string { return "squeue" }

// Pipe-delimited so job names containing spaces survive parsing.
const squeueFormat = "JobID:|,Name:|,UserName:|,Account:|,Partition:|,QOS:|,State:|,StartTime:|,NodeList:|,tres-alloc:"

func (s *SqueueSource) RunningJobs(ctx context.Context) ([]Job, error) {
	ctx, cancel := context.WithTimeout(ctx, s.Timeout)
	defer cancel()

	out, err := exec.CommandContext(ctx, s.Binary,
		"--noheader", "--states=RUNNING", "--array",
		"--Format="+squeueFormat,
	).Output()
	if err != nil {
		return nil, fmt.Errorf("squeue: %w", err)
	}

	var jobs []Job
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		f := strings.Split(line, "|")
		if len(f) < 10 {
			continue
		}
		for i := range f {
			f[i] = strings.TrimSpace(f[i])
		}
		start, _ := time.Parse("2006-01-02T15:04:05", f[7])
		jobs = append(jobs, Job{
			JobID:     f[0],
			Name:      f[1],
			User:      f[2],
			Account:   f[3],
			Partition: f[4],
			QOS:       f[5],
			State:     f[6],
			StartTime: start,
			Nodes:     ExpandNodeList(f[8]),
			GPUCount:  ParseGPUCount(f[9]),
		})
	}
	return jobs, nil
}

// ── REST (slurmrestd) ──────────────────────────────────────────────────────

// RESTSource reads jobs from slurmrestd. Preferred where available: structured,
// versioned, and no output parsing.
type RESTSource struct {
	BaseURL   string
	Version   string // e.g. "v0.0.42"
	Token     string
	Username  string
	Client    *http.Client
}

func NewRESTSource(baseURL, version, token, username string) *RESTSource {
	return &RESTSource{
		BaseURL: strings.TrimRight(baseURL, "/"), Version: version,
		Token: token, Username: username,
		Client: &http.Client{Timeout: 20 * time.Second},
	}
}

func (r *RESTSource) Name() string { return "slurmrestd/" + r.Version }

type restJobsResponse struct {
	Jobs []struct {
		JobID     json.Number `json:"job_id"`
		Name      string      `json:"name"`
		UserName  string      `json:"user_name"`
		Account   string      `json:"account"`
		Partition string      `json:"partition"`
		QOS       string      `json:"qos"`
		JobState  any         `json:"job_state"`
		StartTime any         `json:"start_time"`
		Nodes     string      `json:"nodes"`
		TRESAlloc string      `json:"tres_alloc_str"`
	} `json:"jobs"`
}

func (r *RESTSource) RunningJobs(ctx context.Context) ([]Job, error) {
	url := fmt.Sprintf("%s/slurm/%s/jobs", r.BaseURL, r.Version)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("X-SLURM-USER-NAME", r.Username)
	req.Header.Set("X-SLURM-USER-TOKEN", r.Token)

	resp, err := r.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("slurmrestd: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("slurmrestd returned %s", resp.Status)
	}

	var parsed restJobsResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, fmt.Errorf("decode slurmrestd response: %w", err)
	}

	var jobs []Job
	for _, j := range parsed.Jobs {
		state := normalizeState(j.JobState)
		if state != "RUNNING" {
			continue
		}
		jobs = append(jobs, Job{
			JobID:     j.JobID.String(),
			Name:      j.Name,
			User:      j.UserName,
			Account:   j.Account,
			Partition: j.Partition,
			QOS:       j.QOS,
			State:     state,
			StartTime: normalizeTime(j.StartTime),
			Nodes:     ExpandNodeList(j.Nodes),
			GPUCount:  ParseGPUCount(j.TRESAlloc),
		})
	}
	return jobs, nil
}

// job_state is a bare string on older API versions and a list on newer ones.
func normalizeState(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case []any:
		if len(t) > 0 {
			if s, ok := t[0].(string); ok {
				return s
			}
		}
	}
	return ""
}

// start_time is a Unix int on older versions and {set,infinite,number} on newer.
func normalizeTime(v any) time.Time {
	switch t := v.(type) {
	case float64:
		return time.Unix(int64(t), 0)
	case map[string]any:
		if n, ok := t["number"].(float64); ok {
			return time.Unix(int64(n), 0)
		}
	}
	return time.Time{}
}

// ── Controller ─────────────────────────────────────────────────────────────

// CLIController performs actions via scancel and scontrol.
type CLIController struct {
	ScancelBinary  string
	ScontrolBinary string
	Timeout        time.Duration
}

func NewCLIController() *CLIController {
	return &CLIController{ScancelBinary: "scancel", ScontrolBinary: "scontrol", Timeout: 20 * time.Second}
}

func (c *CLIController) Name() string { return "scancel/scontrol" }

func (c *CLIController) Cancel(ctx context.Context, jobID, reason string) error {
	ctx, cancel := context.WithTimeout(ctx, c.Timeout)
	defer cancel()
	// Recorded in accounting so the user can find out what happened.
	out, err := exec.CommandContext(ctx, c.ScancelBinary,
		"--full", "--verbose", jobID).CombinedOutput()
	if err != nil {
		return fmt.Errorf("scancel %s: %w: %s", jobID, err, strings.TrimSpace(string(out)))
	}
	return nil
}

func (c *CLIController) Drain(ctx context.Context, node, reason string) error {
	ctx, cancel := context.WithTimeout(ctx, c.Timeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, c.ScontrolBinary, "update",
		"NodeName="+node, "State=DRAIN", "Reason="+reason).CombinedOutput()
	if err != nil {
		return fmt.Errorf("scontrol drain %s: %w: %s", node, err, strings.TrimSpace(string(out)))
	}
	return nil
}

// NoopController logs intent without touching the cluster. The default.
type NoopController struct{}

func (NoopController) Name() string                             { return "dry-run" }
func (NoopController) Cancel(context.Context, string, string) error { return nil }
func (NoopController) Drain(context.Context, string, string) error  { return nil }

// ── Parsing helpers ────────────────────────────────────────────────────────

var gresGPU = regexp.MustCompile(`gres/gpu(?::[a-z0-9_]+)?=(\d+)`)

// ParseGPUCount extracts total GPUs from a TRES string such as
// "cpu=32,mem=256G,node=2,billing=32,gres/gpu=8" or "gres/gpu:a100=4".
//
// Typed and untyped entries can both appear; sum only the untyped total when
// present, since Slurm emits both and double counting would inflate waste
// figures by 2x.
func ParseGPUCount(tres string) int {
	matches := gresGPU.FindAllStringSubmatch(tres, -1)
	if len(matches) == 0 {
		return 0
	}
	for _, m := range matches {
		// The untyped "gres/gpu=" form is the authoritative total.
		if strings.Contains(m[0], "gres/gpu=") {
			n, _ := strconv.Atoi(m[1])
			return n
		}
	}
	total := 0
	for _, m := range matches {
		n, _ := strconv.Atoi(m[1])
		total += n
	}
	return total
}

var rangeRe = regexp.MustCompile(`^([a-zA-Z0-9\-_]*?)\[([0-9,\-]+)\]$`)

// ExpandNodeList expands Slurm's compact hostlist syntax:
// "gpu[001-004,007]" -> gpu001 gpu002 gpu003 gpu004 gpu007.
func ExpandNodeList(list string) []string {
	list = strings.TrimSpace(list)
	if list == "" || list == "(null)" || list == "None assigned" {
		return nil
	}

	var out []string
	for _, part := range splitTopLevel(list) {
		m := rangeRe.FindStringSubmatch(part)
		if m == nil {
			out = append(out, part)
			continue
		}
		prefix, body := m[1], m[2]
		for _, span := range strings.Split(body, ",") {
			lo, hi, width, ok := parseSpan(span)
			if !ok {
				continue
			}
			for n := lo; n <= hi; n++ {
				out = append(out, fmt.Sprintf("%s%0*d", prefix, width, n))
			}
		}
	}
	return out
}

func parseSpan(span string) (lo, hi, width int, ok bool) {
	if a, b, found := strings.Cut(span, "-"); found {
		l, e1 := strconv.Atoi(a)
		h, e2 := strconv.Atoi(b)
		if e1 != nil || e2 != nil || h < l {
			return 0, 0, 0, false
		}
		return l, h, len(a), true
	}
	n, err := strconv.Atoi(span)
	if err != nil {
		return 0, 0, 0, false
	}
	return n, n, len(span), true
}

// splitTopLevel splits on commas that are not inside brackets, so
// "a[1,2],b[3]" yields "a[1,2]" and "b[3]".
func splitTopLevel(s string) []string {
	var parts []string
	depth, start := 0, 0
	for i, r := range s {
		switch r {
		case '[':
			depth++
		case ']':
			depth--
		case ',':
			if depth == 0 {
				parts = append(parts, s[start:i])
				start = i + 1
			}
		}
	}
	return append(parts, s[start:])
}
