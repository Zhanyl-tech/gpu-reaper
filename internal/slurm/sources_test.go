package slurm

import (
	"reflect"
	"testing"
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
		{"cpu=4,mem=16G", 0},
		{"", 0},
		{"gres/gpu:a100=2,gres/gpu:v100=3", 5},
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
	}
	for _, tc := range cases {
		if got := ExpandNodeList(tc.in); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("ExpandNodeList(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

func TestExpandNodeListRejectsInvertedRange(t *testing.T) {
	if got := ExpandNodeList("gpu[009-003]"); len(got) != 0 {
		t.Errorf("inverted range should yield nothing, got %v", got)
	}
}

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
	if got := normalizeTime(nil); !got.IsZero() {
		t.Errorf("nil should yield zero time, got %v", got)
	}
}
