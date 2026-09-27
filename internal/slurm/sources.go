package slurm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// RejectFunc receives one record a source refused to turn into a Job, with a
// short reason. The daemon counts and logs these; a rejected record is never
// judged, because a record that cannot be parsed cannot be trusted.
type RejectFunc func(reason string)

// ── squeue ─────────────────────────────────────────────────────────────────

// SqueueSource reads running jobs by shelling out to squeue.
//
// It uses an explicit --Format with every field sized ":|" or ":" (no width),
// rather than the default output, because the default is column-ordered by
// config and truncates. SchedMD's opts.c and print.c give such fields width 0,
// which prints them in full.
type SqueueSource struct {
	Binary  string
	Timeout time.Duration
	// NodeList, when set, restricts squeue to jobs on these nodes
	// (--nodelist). A per-node daemon sets it to its own node so it does not
	// pull the whole cluster's job list every cycle.
	NodeList string
	// OnReject, if set, is called for every line that fails validation.
	OnReject RejectFunc
}

func NewSqueueSource(binary string) *SqueueSource {
	if binary == "" {
		binary = "squeue"
	}
	return &SqueueSource{Binary: binary, Timeout: 15 * time.Second}
}

func (s *SqueueSource) Name() string { return "squeue" }

// squeueFormat puts Name last. Job names are the one free-text field a user
// controls, and they may contain '|'. With Name last, SplitN into exactly
// squeueFields parts leaves any '|' inside the name, instead of shifting every
// field after it (which previously let a job named "x|ci-bot" parse as user
// ci-bot and hide from, or be exempted by, the reaper).
const squeueFormat = "JobID:|,UserName:|,Account:|,Partition:|,QOS:|,State:|,StartTime:|,NodeList:|,tres-alloc:|,Name:"

const squeueFields = 10

// squeueEnv pins how squeue prints times. Slurm formats StartTime with
// localtime_r and "%FT%T" (no offset) unless SLURM_TIME_FORMAT says otherwise
// (src/common/parse_time.c, slurm_make_time_str). Setting TZ=UTC for the child
// makes the output UTC whatever the host zone is, and SLURM_TIME_FORMAT=standard
// ("year-month-dateThour:minute:second" per the squeue man page) overrides any
// site or user custom format. os/exec uses the last value of a duplicated key,
// so these win over anything inherited.
var squeueEnv = []string{"TZ=UTC", "SLURM_TIME_FORMAT=standard"}

func (s *SqueueSource) RunningJobs(ctx context.Context) ([]Job, error) {
	ctx, cancel := context.WithTimeout(ctx, s.Timeout)
	defer cancel()

	args := []string{"--noheader", "--states=RUNNING", "--array", "--Format=" + squeueFormat}
	if s.NodeList != "" {
		args = append(args, "--nodelist="+s.NodeList)
	}
	cmd := exec.CommandContext(ctx, s.Binary, args...)
	cmd.Env = append(os.Environ(), squeueEnv...)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("squeue: %w", err)
	}
	return ParseSqueue(string(out), s.OnReject), nil
}

// ParseSqueue parses squeue output in squeueFormat. Lines that fail validation
// are passed to reject (if non-nil) and dropped. Job IDs that appear more than
// once are dropped entirely: two records for one job means at least one was
// forged or mangled, and we cannot tell which.
func ParseSqueue(out string, reject RejectFunc) []Job {
	if reject == nil {
		reject = func(string) {}
	}
	var jobs []Job
	count := map[string]int{}
	for _, line := range strings.Split(out, "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		j, err := parseSqueueLine(line)
		if err != nil {
			reject(err.Error())
			continue
		}
		count[j.JobID]++
		jobs = append(jobs, j)
	}
	kept := jobs[:0]
	for _, j := range jobs {
		if count[j.JobID] > 1 {
			reject(fmt.Sprintf("job %s appears %d times; dropping every copy", j.JobID, count[j.JobID]))
			continue
		}
		kept = append(kept, j)
	}
	return kept
}

// jobIDRe accepts a plain job ID, an array task (123_4) and a heterogeneous
// component (123+0).
var jobIDRe = regexp.MustCompile(`^[0-9]+(_[0-9]+)?(\+[0-9]+)?$`)

func parseSqueueLine(line string) (Job, error) {
	f := strings.SplitN(line, "|", squeueFields)
	if len(f) != squeueFields {
		return Job{}, fmt.Errorf("want %d fields, got %d", squeueFields, len(f))
	}
	for i := range f {
		f[i] = strings.TrimSpace(f[i])
	}
	if !jobIDRe.MatchString(f[0]) {
		return Job{}, fmt.Errorf("invalid job id %q", truncate(f[0], 32))
	}
	if f[5] != "RUNNING" {
		return Job{}, fmt.Errorf("job %s: state %q, want RUNNING", f[0], truncate(f[5], 32))
	}
	// Parsed as UTC because squeueEnv forces the child to print UTC. A value
	// that does not parse is a rejected record, never a zero start time: a
	// zero start time once meant "age 0", which silently exempted the job.
	start, err := time.ParseInLocation("2006-01-02T15:04:05", f[6], time.UTC)
	if err != nil {
		return Job{}, fmt.Errorf("job %s: unparseable start time %q", f[0], truncate(f[6], 32))
	}
	nodes, err := ExpandNodeList(f[7])
	if err != nil {
		return Job{}, fmt.Errorf("job %s: %v", f[0], err)
	}
	return Job{
		JobID:     f[0],
		User:      f[1],
		Account:   f[2],
		Partition: f[3],
		QOS:       f[4],
		State:     f[5],
		StartTime: start,
		Nodes:     nodes,
		GPUCount:  ParseGPUCount(f[8]),
		Name:      f[9],
	}, nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// ── REST (slurmrestd) ──────────────────────────────────────────────────────

// RESTSource reads jobs from slurmrestd. Preferred where available: structured,
// versioned, and no text parsing, so job names cannot disturb other fields.
type RESTSource struct {
	BaseURL  string
	Version  string // e.g. "v0.0.42"
	Username string
	// Token is a fixed JWT. TokenFile, if set, is re-read on every request
	// instead, so a token rotated on disk is picked up without a restart.
	// Slurm's JWT documentation gives `scontrol token` a default lifespan of
	// 1800 seconds, so a token read once at startup stops working.
	Token     string
	TokenFile string
	Client    *http.Client
	OnReject  RejectFunc
}

func NewRESTSource(baseURL, version, token, tokenFile, username string) *RESTSource {
	return &RESTSource{
		BaseURL: strings.TrimRight(baseURL, "/"), Version: version,
		Token: token, TokenFile: tokenFile, Username: username,
		Client: &http.Client{Timeout: 20 * time.Second},
	}
}

func (r *RESTSource) Name() string { return "slurmrestd/" + r.Version }

type restJob struct {
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
}

type restJobsResponse struct {
	Jobs []restJob `json:"jobs"`
}

func (r *RESTSource) token() (string, error) {
	if r.TokenFile == "" {
		return r.Token, nil
	}
	b, err := os.ReadFile(r.TokenFile)
	if err != nil {
		return "", fmt.Errorf("read slurm token file: %w", err)
	}
	return strings.TrimSpace(string(b)), nil
}

func (r *RESTSource) RunningJobs(ctx context.Context) ([]Job, error) {
	tok, err := r.token()
	if err != nil {
		return nil, err
	}
	url := fmt.Sprintf("%s/slurm/%s/jobs", r.BaseURL, r.Version)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("X-SLURM-USER-NAME", r.Username)
	req.Header.Set("X-SLURM-USER-TOKEN", tok)

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
	return convertREST(parsed, r.OnReject), nil
}

func convertREST(parsed restJobsResponse, reject RejectFunc) []Job {
	if reject == nil {
		reject = func(string) {}
	}
	var jobs []Job
	for _, j := range parsed.Jobs {
		state := normalizeState(j.JobState)
		if state != "RUNNING" {
			continue
		}
		id := j.JobID.String()
		nodes, err := ExpandNodeList(j.Nodes)
		if err != nil {
			reject(fmt.Sprintf("job %s: %v", id, err))
			continue
		}
		jobs = append(jobs, Job{
			JobID:     id,
			Name:      j.Name,
			User:      j.UserName,
			Account:   j.Account,
			Partition: j.Partition,
			QOS:       j.QOS,
			State:     state,
			// A zero StartTime (unset or unparseable) is refused by the policy
			// engine with "start time unknown"; it is never read as age 0.
			StartTime: normalizeTime(j.StartTime),
			Nodes:     nodes,
			GPUCount:  ParseGPUCount(j.TRESAlloc),
		})
	}
	return jobs
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

// start_time is a Unix int on older versions and {set,infinite,number} on
// newer. An object with set=false, infinite=true, or a non-positive number
// has no usable start time and returns the zero time.
func normalizeTime(v any) time.Time {
	switch t := v.(type) {
	case float64:
		if t > 0 {
			return time.Unix(int64(t), 0)
		}
	case map[string]any:
		if set, ok := t["set"].(bool); ok && !set {
			return time.Time{}
		}
		if inf, ok := t["infinite"].(bool); ok && inf {
			return time.Time{}
		}
		if n, ok := t["number"].(float64); ok && n > 0 {
			return time.Unix(int64(n), 0)
		}
	}
	return time.Time{}
}

// ── Controller ─────────────────────────────────────────────────────────────

// CLIController performs actions via scancel and scontrol.
//
// Binaries should be absolute paths: the configuration validator requires that
// in enforce mode, so a writable directory earlier in PATH cannot substitute
// its own scancel.
type CLIController struct {
	ScancelBinary  string
	ScontrolBinary string
	Timeout        time.Duration
}

func NewCLIController(scancel, scontrol string) *CLIController {
	return &CLIController{ScancelBinary: scancel, ScontrolBinary: scontrol, Timeout: 20 * time.Second}
}

func (c *CLIController) Name() string { return "scancel/scontrol" }

// nodeNameRe matches the characters a Slurm node name expanded by
// ExpandNodeList can contain. Anything else is refused before exec.
var nodeNameRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*$`)

var errUnsafeArg = errors.New("refusing to run: argument failed validation")

// Cancel records reason as the job's AdminComment, then cancels it.
//
// The order is deliberate and fails safe: if the reason cannot be recorded,
// the job is not cancelled, so no job is ever killed without an answer to
// "what happened to job 12345" in its accounting record. scontrol documents
// AdminComment as settable only by a Slurm administrator; a deployment without
// that privilege cannot cancel through this controller.
func (c *CLIController) Cancel(ctx context.Context, jobID, reason string) error {
	if !jobIDRe.MatchString(jobID) || strings.ContainsAny(reason, "\n\r") {
		return fmt.Errorf("%w: job %q", errUnsafeArg, jobID)
	}
	ctx, cancel := context.WithTimeout(ctx, c.Timeout)
	defer cancel()

	out, err := exec.CommandContext(ctx, c.ScontrolBinary, "update",
		"JobId="+jobID, "AdminComment="+reason).CombinedOutput()
	if err != nil {
		return fmt.Errorf("record cancel reason for job %s (job NOT cancelled): %w: %s",
			jobID, err, strings.TrimSpace(string(out)))
	}
	out, err = exec.CommandContext(ctx, c.ScancelBinary, "--verbose", jobID).CombinedOutput()
	if err != nil {
		return fmt.Errorf("scancel %s: %w: %s", jobID, err, strings.TrimSpace(string(out)))
	}
	return nil
}

// Drain marks a node DRAIN with a reason. Slurm then leaves it DRAINED once
// its jobs finish, until someone runs `scontrol update NodeName=<n>
// State=RESUME`. This tool never does that; see the README.
func (c *CLIController) Drain(ctx context.Context, node, reason string) error {
	if !nodeNameRe.MatchString(node) || strings.ContainsAny(reason, "\n\r") {
		return fmt.Errorf("%w: node %q", errUnsafeArg, node)
	}
	ctx, cancel := context.WithTimeout(ctx, c.Timeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, c.ScontrolBinary, "update",
		"NodeName="+node, "State=DRAIN", "Reason="+reason).CombinedOutput()
	if err != nil {
		return fmt.Errorf("scontrol drain %s: %w: %s", node, err, strings.TrimSpace(string(out)))
	}
	return nil
}

// NoopController touches nothing. Observe mode wires it in, so even a bug that
// let a dry-run actor call the controller could not change the cluster.
type NoopController struct{}

func (NoopController) Name() string                                 { return "dry-run" }
func (NoopController) Cancel(context.Context, string, string) error { return nil }
func (NoopController) Drain(context.Context, string, string) error  { return nil }

// ── Parsing helpers ────────────────────────────────────────────────────────

// gresGPU matches "gres/gpu=N" and typed "gres/gpu:<type>=N", where the type
// may contain anything but '=' or ',' (uppercase, hyphens, dots: "A100",
// "a100-sxm4-80gb", "1g.10gb"). It does not match gres/gpumem or gres/gpuutil.
var gresGPU = regexp.MustCompile(`gres/gpu(?::[^=,]+)?=(\d+)`)

// ParseGPUCount extracts total GPUs from a TRES string such as
// "cpu=32,mem=256G,node=2,billing=32,gres/gpu=8" or "gres/gpu:a100=4".
//
// Typed and untyped entries can both appear; use only the untyped total when
// present, since counting both would double every waste figure.
func ParseGPUCount(tres string) int {
	matches := gresGPU.FindAllStringSubmatch(tres, -1)
	if len(matches) == 0 {
		return 0
	}
	for _, m := range matches {
		if strings.HasPrefix(m[0], "gres/gpu=") {
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

// maxExpandedNodes caps expansion so a hostile or broken list cannot make the
// daemon allocate without bound.
const maxExpandedNodes = 100000

// ExpandNodeList expands Slurm's compact hostlist syntax, including several
// bracket groups and text after them:
//
//	gpu[001-003,007]     -> gpu001 gpu002 gpu003 gpu007
//	gpu[01-02]-ib        -> gpu01-ib gpu02-ib
//	rack[1-2]-gpu[01-02] -> rack1-gpu01 rack1-gpu02 rack2-gpu01 rack2-gpu02
//	gpu[01-02].cluster   -> gpu01.cluster gpu02.cluster
//
// It returns an error rather than a literal for anything it cannot parse. A
// literal like "gpu[01-02]-ib" matches no real node, which used to drop the
// job from coverage and from the shared-node count without a trace.
func ExpandNodeList(list string) ([]string, error) {
	list = strings.TrimSpace(list)
	if list == "" || list == "(null)" || list == "None assigned" {
		return nil, nil
	}
	parts, err := splitTopLevel(list)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, part := range parts {
		names, err := expandPart(part)
		if err != nil {
			return nil, fmt.Errorf("hostlist %q: %w", truncate(list, 64), err)
		}
		if len(out)+len(names) > maxExpandedNodes {
			return nil, fmt.Errorf("hostlist %q expands to more than %d nodes", truncate(list, 64), maxExpandedNodes)
		}
		out = append(out, names...)
	}
	return out, nil
}

func expandPart(part string) ([]string, error) {
	if part == "" {
		return nil, errors.New("empty host")
	}
	results := []string{""}
	for len(part) > 0 {
		open := strings.IndexByte(part, '[')
		if open < 0 {
			if err := checkLiteral(part); err != nil {
				return nil, err
			}
			for i := range results {
				results[i] += part
			}
			break
		}
		lit := part[:open]
		if err := checkLiteral(lit); err != nil {
			return nil, err
		}
		close := strings.IndexByte(part[open:], ']')
		if close < 0 {
			return nil, errors.New("unbalanced '['")
		}
		close += open
		body := part[open+1 : close]
		var nums []string
		for _, span := range strings.Split(body, ",") {
			lo, hi, width, ok := parseSpan(span)
			if !ok {
				return nil, fmt.Errorf("bad range %q", span)
			}
			// Every size check happens before anything is allocated, and none
			// can overflow. parseSpan guarantees 0 <= lo <= hi, so hi-lo is
			// non-negative and fits; hi-lo+1 does not when the span is
			// 0-MaxInt64, which used to wrap negative and pass the cap. The
			// running total covers many spans in one bracket, each under the
			// cap on its own.
			if hi-lo >= maxExpandedNodes || len(nums) > maxExpandedNodes-(hi-lo+1) {
				return nil, fmt.Errorf("range %q expands to more than %d nodes", truncate(span, 32), maxExpandedNodes)
			}
			// Count up from lo rather than comparing n <= hi: at hi ==
			// MaxInt64, n++ wraps and n <= hi never becomes false.
			for k := 0; k <= hi-lo; k++ {
				nums = append(nums, fmt.Sprintf("%0*d", width, lo+k))
			}
		}
		// len(results) and len(nums) are each at most maxExpandedNodes, so the
		// product fits in an int; check it before make() sizes a slice by it.
		if len(results)*len(nums) > maxExpandedNodes {
			return nil, fmt.Errorf("expands to more than %d nodes", maxExpandedNodes)
		}
		next := make([]string, 0, len(results)*len(nums))
		for _, r := range results {
			for _, n := range nums {
				next = append(next, r+lit+n)
			}
		}
		results = next
		part = part[close+1:]
	}
	return results, nil
}

var literalRe = regexp.MustCompile(`^[A-Za-z0-9._-]*$`)

func checkLiteral(s string) error {
	if !literalRe.MatchString(s) {
		return fmt.Errorf("unexpected characters in %q", truncate(s, 32))
	}
	return nil
}

func parseSpan(span string) (lo, hi, width int, ok bool) {
	if a, b, found := strings.Cut(span, "-"); found {
		if a == "" || b == "" {
			return 0, 0, 0, false
		}
		l, e1 := strconv.Atoi(a)
		h, e2 := strconv.Atoi(b)
		if e1 != nil || e2 != nil || h < l || l < 0 {
			return 0, 0, 0, false
		}
		return l, h, len(a), true
	}
	if span == "" {
		return 0, 0, 0, false
	}
	n, err := strconv.Atoi(span)
	if err != nil || n < 0 {
		return 0, 0, 0, false
	}
	return n, n, len(span), true
}

// splitTopLevel splits on commas that are not inside brackets, so
// "a[1,2],b[3]" yields "a[1,2]" and "b[3]".
func splitTopLevel(s string) ([]string, error) {
	var parts []string
	depth, start := 0, 0
	for i, r := range s {
		switch r {
		case '[':
			depth++
			if depth > 1 {
				return nil, errors.New("nested '['")
			}
		case ']':
			depth--
			if depth < 0 {
				return nil, errors.New("unbalanced ']'")
			}
		case ',':
			if depth == 0 {
				parts = append(parts, s[start:i])
				start = i + 1
			}
		}
	}
	if depth != 0 {
		return nil, errors.New("unbalanced '['")
	}
	return append(parts, s[start:]), nil
}
