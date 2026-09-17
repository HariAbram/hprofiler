#!/bin/bash
# Integration regression matrix: builds tiny per-backend fixture programs
# (tests/fixtures/), profiles each through every relevant hprofiler backend,
# and runs `summary`, `efficiency`, and `critical-path` against the
# resulting trace -- asserting none of them crash. This is the "does the
# whole toolchain still work end-to-end" check referenced in
# DOCUMENTATION.md / the audit-fixes memory; unit tests in tests/*.py check
# the analysis math in isolation, this checks the real pipeline.
#
# Not all backends produce events on every machine (e.g. no working GPU
# driver, perf_event_paranoid blocking perf/likwid) -- that's expected and
# not a failure here; a *crash* (non-zero exit, traceback) is.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"       # tests/
REPO="$(cd "$HERE/.." && pwd)"
FIX="$HERE/fixtures"
OUT="$(mktemp -d)"
FAIL=0

log() { echo "$@"; }
ok()  { log "  OK   $1"; }
bad() { log "  FAIL $1"; FAIL=1; }

run_or_fail() {
    local desc="$1"; shift
    if "$@" > "$OUT/last.log" 2>&1; then
        ok "$desc"
    else
        bad "$desc (see $OUT/last.log)"
        tail -20 "$OUT/last.log"
    fi
}

cd "$REPO"

log "=== building fixtures (skips ones whose toolchain is missing) ==="
declare -A BUILD_CMD=(
    [cpu]="gcc -O2 -fno-omit-frame-pointer -rdynamic -o $FIX/likwid_mini $FIX/likwid_mini.c"
    [likwid]="gcc -O2 -fno-omit-frame-pointer -rdynamic -o $FIX/likwid_mini $FIX/likwid_mini.c"
    [openmp]="clang++ -O2 -fopenmp -fno-omit-frame-pointer -rdynamic -o $FIX/omp_mini $FIX/omp_mini.cpp"
    [opencl]="gcc -O2 -fno-omit-frame-pointer -rdynamic -o $FIX/cl_mini $FIX/cl_mini.c -lOpenCL"
    [mpi]="mpicc -O2 -fno-omit-frame-pointer -rdynamic -o $FIX/mpi_mini $FIX/mpi_mini.c"
    [cuda]="nvcc -O2 -o $FIX/cuda_mini $FIX/cuda_mini.cu"
    [rocm]="gcc -O2 -fno-omit-frame-pointer -rdynamic -o $FIX/hip_mini $FIX/hip_mini.c /usr/lib/x86_64-linux-gnu/libamdhip64.so.5"
)
declare -A PROG=(
    [cpu]="$FIX/likwid_mini"
    [likwid]="$FIX/likwid_mini"
    [openmp]="$FIX/omp_mini"
    [opencl]="$FIX/cl_mini"
    [mpi]="$FIX/mpi_mini"
    [cuda]="$FIX/cuda_mini"
    [rocm]="$FIX/hip_mini"
)

AVAILABLE=()
for backend in cpu likwid openmp opencl mpi cuda rocm; do
    if eval "${BUILD_CMD[$backend]}" > "$OUT/build_$backend.log" 2>&1; then
        ok "build $backend fixture"
        AVAILABLE+=("$backend")
    else
        log "  SKIP build $backend fixture (toolchain unavailable, see $OUT/build_$backend.log)"
    fi
done

log ""
log "=== profiling each + running summary/efficiency/critical-path ==="
for backend in "${AVAILABLE[@]}"; do
    trace="$OUT/${backend}.json"
    run_or_fail "hprofiler run --backend $backend" \
        python3 hprofiler run --backend "$backend" --no-ui -o "$trace" -- "${PROG[$backend]}"
    [ -f "$trace" ] || { bad "$backend produced no trace file"; continue; }
    run_or_fail "hprofiler summary ($backend)"       python3 hprofiler summary "$trace"
    run_or_fail "hprofiler efficiency ($backend)"     python3 hprofiler efficiency "$trace"
    run_or_fail "hprofiler critical-path ($backend)"  python3 hprofiler critical-path "$trace"
done

log ""
if [ "$FAIL" -eq 0 ]; then
    log "=== ALL OK (logs in $OUT) ==="
else
    log "=== FAILURES ABOVE (logs in $OUT) ==="
fi
exit "$FAIL"
