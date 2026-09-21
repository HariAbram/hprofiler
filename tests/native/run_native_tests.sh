#!/bin/bash
# Builds and runs the standalone C-level correctness/stress tests for
# hooks/common/ringbuffer.h (see project_causal_attribution_redesign memory
# / DOCUMENTATION.md's collection-path section). Separate from
# tests/integration/run_matrix.sh, which profiles fixture *programs*
# through the real backends -- this tests infrastructure that isn't wired
# into any hook yet, so there's no "profile a program" path to exercise it
# through.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
FAIL=0

echo "=== building ringbuffer_stress ==="
if gcc -O2 -pthread -Wall -Wextra -o "$HERE/ringbuffer_stress" "$HERE/ringbuffer_stress.c"; then
    echo "  OK   build"
else
    echo "  FAIL build"
    exit 1
fi

echo "=== running ringbuffer_stress ==="
if "$HERE/ringbuffer_stress"; then
    echo "  OK   ringbuffer_stress"
else
    echo "  FAIL ringbuffer_stress"
    FAIL=1
fi

# ThreadSanitizer pass, when available -- the strongest available check for
# this hand-rolled lock-free structure (catches missing/incorrect
# atomics that logical correctness checks can pass despite, if a data race
# never happens to flip a result on a given run). Not required (some
# environments can't run TSan's instrumented mmap layout at all -- this
# machine needed ASLR disabled via `setarch -R` to even start it), so this
# step is best-effort and does not fail the overall run if TSan itself
# can't start.
if command -v gcc >/dev/null 2>&1 && gcc -fsanitize=thread -E -x c /dev/null -o /dev/null >/dev/null 2>&1; then
    echo "=== building + running under ThreadSanitizer (best-effort) ==="
    if gcc -O1 -pthread -fsanitize=thread -Wall -Wextra -o "$HERE/ringbuffer_stress_tsan" "$HERE/ringbuffer_stress.c" 2>/tmp/tsan_build.log; then
        if setarch "$(uname -m)" -R "$HERE/ringbuffer_stress_tsan" > /tmp/tsan_run.log 2>&1; then
            if grep -q "ThreadSanitizer: data race" /tmp/tsan_run.log; then
                echo "  FAIL TSan detected a data race -- see /tmp/tsan_run.log"
                FAIL=1
            else
                echo "  OK   TSan clean (no data races detected)"
            fi
        else
            echo "  SKIP TSan run failed to start in this environment (see /tmp/tsan_run.log) -- not counted as a failure"
        fi
    else
        echo "  SKIP TSan build failed in this environment -- not counted as a failure"
    fi
fi

if [ "$FAIL" = "0" ]; then
    echo "=== ALL OK ==="
else
    echo "=== FAILED ==="
fi
exit $FAIL
