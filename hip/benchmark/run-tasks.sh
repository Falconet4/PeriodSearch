#!/bin/bash
# Run a binary over the 54-workunit benchmark set.
# Usage: ./run-tasks.sh <binary> <output-dir> [TASKS.tsv]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
if [ $# -lt 2 ]; then
	echo "usage: $0 <binary> <output-dir> [TASKS.tsv]" >&2
	exit 2
fi
BIN=$(readlink -f "$1")
OUT="$2"
MAN="${3:-$HERE/TASKS.tsv}"
mkdir -p "$OUT"
: > "$OUT/times.tsv"

tail -n +2 "$MAN" | while IFS=$'\t' read -r task input set; do
	[ -z "$task" ] && continue
	d="$OUT/work/$task"
	rm -rf "$d"
	mkdir -p "$d"
	cp "$HERE/inputs/$input" "$d/ps_in" || { echo "$task COPY_FAIL"; continue; }
	printf '<soft_link>ps_in</soft_link>\n' > "$d/period_search_in"
	(
		cd "$d" || exit 1
		if [ -n "${PS_FP32_TAIL:-}" ]; then
			PS_STANDALONE=1 PS_FP32_TAIL="$PS_FP32_TAIL" PS_WARP="${PS_WARP:-1}" \
				/usr/bin/time -f "%e" -o time.txt "$BIN" >/dev/null 2> app.log
		else
			PS_STANDALONE=1 PS_WARP="${PS_WARP:-1}" \
				/usr/bin/time -f "%e" -o time.txt "$BIN" >/dev/null 2> app.log
		fi
	)
	t=$(cat "$d/time.txt" 2>/dev/null || echo 0)
	cp "$d/period_search_out" "$OUT/$task.out" 2>/dev/null || true
	printf '%s\t%s\t%s\n' "$task" "$t" "$set" | tee -a "$OUT/times.tsv"
done

awk -F'\t' 'NF>=2 && $2+0>0 {s+=$2; n++} END {printf "TOTAL %.2f s over %d tasks\n", s, n}' "$OUT/times.tsv"
