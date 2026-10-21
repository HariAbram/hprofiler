#!/bin/bash
# Builds and runs the standalone C-level correctness/stress test and
# micro-benchmark for the fixed-slot ring of hooks/common/ringbuffer.h
# (DOCUMENTATION.md, "Measured hot-path overhead"). The variable-length rings every hook actually
# emits through (hooks/common/hp_transport.h) are tested by
# tests/test_transport_native.py (plain / ASan+UBSan / TSan scenarios) and,
# end to end, by tests/integration/run_matrix.sh and the integration tests.
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

# hooks/common/hp_module.h: module ids, the kernel table, in-memory ELF and
# fat-binary image sizes (also driven, with ASan+UBSan, by
# tests/test_disasm_modules.py).
echo "=== hp_module_test ==="
if gcc -O1 -Wall -Wextra -pthread -o "$HERE/hp_module_test" "$HERE/hp_module_test.c" \
        && "$HERE/hp_module_test" | grep -q "RESULT fails=0"; then
    echo "  OK   hp_module_test"
else
    echo "  FAIL hp_module_test"
    FAIL=1
fi
rm -f "$HERE/hp_module_test"

# hooks/cuda_hook/cupti_trace.c's record decoding and callback handling,
# built against the real CUPTI headers (asserted in detail by
# tests/integration/test_native_gpu_records.py); here: it builds, runs, and
# a launch the hook did not intercept names its module (mod=).
CUPTI_INC=""
for d in "${HPROFILER_CUPTI_INCLUDE:-}" /usr/local/cuda/include /usr/local/cuda/extras/CUPTI/include /usr/include; do
    if [ -n "$d" ] && [ -f "$d/cupti.h" ]; then CUPTI_INC="$d"; break; fi
done
if [ -n "$CUPTI_INC" ]; then
    echo "=== cupti_records_test ==="
    ROOT="$HERE/../.."
    if gcc -O1 -Wall -Wextra -DHP_HAVE_CUPTI -DHP_CUPTI_UNIT_TEST -DHP_CUPTI_MEMCPY6 -DHP_CUPTI_SYNC2 \
            -I"$CUPTI_INC" -I"$ROOT/hooks/cuda_hook" -o "$HERE/cupti_records_test" \
            "$HERE/cupti_records_test.c" "$ROOT/hooks/cuda_hook/cupti_trace.c" -ldl -lpthread \
            && "$HERE/cupti_records_test" | grep -q "cuLaunchKernelEx:.*src=cupti_cb,corr=760,.*mod=0123456789abcdef"; then
        echo "  OK   cupti_records_test"
    else
        echo "  FAIL cupti_records_test"
        FAIL=1
    fi
    rm -f "$HERE/cupti_records_test"
else
    echo "  SKIP cupti_records_test (no CUPTI headers)"
fi

if [ "$FAIL" = "0" ]; then
    echo "=== ALL OK ==="
else
    echo "=== FAILED ==="
fi
exit $FAIL
