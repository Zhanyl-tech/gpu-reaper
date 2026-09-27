#!/usr/bin/env bash
# Run a bounded demo and assert what the README says it shows. CI runs this,
# so if the demo stops doing what the README describes, the build fails.
#
#   demo/check.sh [scenario]      scenario: healthy idle hung starved flaky unreadable
#
# Needs ./bin/gpu-reaper (make build). Writes only under ./bin/.
set -euo pipefail

cd "$(dirname "$0")/.."
scenario="${1:-hung}"
cycles="${CYCLES:-9}"
bin=./bin/gpu-reaper
cfg="bin/demo-$scenario.yaml"

[[ -x "$bin" ]] || { echo "build first: make build" >&2; exit 2; }

# Pin the fake squeue's notion of "now" for the whole run (see demo/bin/squeue).
FAKE_SQUEUE_ANCHOR="${FAKE_SQUEUE_ANCHOR:-$(date +%s)}"
export FAKE_SQUEUE_ANCHOR

# Same config as the demo, with the chosen scenario and a random metrics port
# so parallel runs do not collide.
sed -e "s/^\( *scenario:\).*/\1 $scenario/" \
	-e 's/^metrics_addr:.*/metrics_addr: "127.0.0.1:0"/' \
	demo/config.yaml >"$cfg"

out="$(PATH="$PWD/demo/bin:$PATH" "$bin" --config "$cfg" --cycles "$cycles" 2>&1)"

fail() {
	echo "FAIL [$scenario, TZ=${TZ:-unset}]: $*" >&2
	echo "---- output ----" >&2
	echo "$out" >&2
	exit 1
}
has() { grep -qE -- "$1" <<<"$out"; }
count() { grep -cE -- "$1" <<<"$out" || true; }

# Always true, whatever the scenario.
has 'mode=observe' || fail "not in observe mode"
has 'verdict=cancel' && fail "a cancel verdict appeared"
has 'would cancel' && fail "a cancel was attempted"
has 'job_id=100243' && fail "exempt user carol's job was reported"
has 'job_id=100244' && fail "CPU-only job was reported"
has 'msg="rejected job record"' && fail "the fake squeue output was rejected (time zone or format regression)"
has 'level=ERROR' && fail "an error was logged"

finding() { echo "msg=finding verdict=$1 signature=$2 job_id=$3"; }

case "$scenario" in
healthy | unreadable)
	has 'msg=finding' && fail "expected no findings"
	;;
hung | idle)
	for job in 100241 100242; do
		has "$(finding alert "$scenario" "$job")" || fail "job $job never alerted as $scenario"
		has "$(finding drain "$scenario" "$job")" || fail "job $job never reached drain"
	done
	# Drain is dry-run and issued once per breach, not every cycle.
	[[ "$(count 'msg="would drain" node=gpu001')" == 1 ]] || fail "gpu001 should be dry-run drained exactly once"
	[[ "$(count 'msg="would drain" node=gpu002')" == 1 ]] || fail "gpu002 should be dry-run drained exactly once"
	[[ "$(count 'msg="would drain" node=gpu003')" == 1 ]] || fail "gpu003 should be dry-run drained exactly once"
	# One snapshot is never a sustained breach: the first cycle must be silent.
	first_cycle="$(grep -m1 'msg="cycle complete"' <<<"$out" || true)"
	grep -q 'at_or_above_alert=0' <<<"$first_cycle" || fail "the first cycle flagged a job: $first_cycle"
	;;
starved)
	has "$(finding alert starved 100241)" || fail "starved job never alerted"
	has 'verdict=drain' && fail "starved must never pass alert"
	;;
flaky)
	# Between bursts the GPUs hold memory at 1% compute, which is starved and
	# capped at alert; a burst clears the finding. Whether an alert or a burst
	# lands in the run depends on how many cycles run, so only the cap is
	# asserted, plus the common checks above.
	has 'verdict=drain' && fail "flaky must never pass alert"
	has 'signature=(idle|hung)' && fail "flaky GPUs do some compute; they are never idle or hung"
	;;
*)
	echo "unknown scenario $scenario" >&2
	exit 2
	;;
esac

# Summary: the last verdict each job reached.
printf '  %-11s' "$scenario"
last="$(grep 'msg=finding' <<<"$out" | sed -E 's/.*verdict=([a-z]+) signature=([a-z]+) job_id=([0-9]+).*/\3 \1 \2/' | awk '{v[$1]=$2" ("$3")"} END{for (j in v) printf "job %s -> %s   ", j, v[j]}' || true)"
echo "${last:-no findings}"
