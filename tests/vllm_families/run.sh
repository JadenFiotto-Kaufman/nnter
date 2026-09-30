#!/bin/bash
# Run the vLLM suite on a host with vLLM and a GPU, one process per family file
# (one engine per process; engines sharing a card overrun their memory fractions).
#   tests/vllm_families/run.sh [family ...]        # default: every tests/vllm_families/test_vllm_*.py
# Environment: PYTHON (the interpreter with vLLM and nnsight), plus whatever that host's vLLM needs.
# Each family's full output is kept in ${TMPDIR:-/tmp}/nnter-vllm-<family>.log.
cd "$(dirname "$0")/../.." || exit 1
families=("$@")
[ ${#families[@]} -eq 0 ] && families=($(ls tests/vllm_families/test_vllm_*.py | sed 's/.*test_vllm_\(.*\)\.py/\1/'))
status=0
for family in "${families[@]}"; do
    log="${TMPDIR:-/tmp}/nnter-vllm-$family.log"
    "${PYTHON:-python}" -m pytest -q -p no:cacheprovider "tests/vllm_families/test_vllm_$family.py" > "$log" 2>&1 || status=1
    echo "$family: $(grep -E '^(FAILED|ERROR) |passed|failed' "$log" | tail -5 | tr '\n' ' ')"
done
exit $status
