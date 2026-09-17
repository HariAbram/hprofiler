# hprofiler test suite

Two layers:

- **Unit tests** (`tests/test_*.py`, stdlib `unittest`, no new dependency):
  test the analysis math and algorithms in isolation against small synthetic
  traces with hand-computed expected answers -- not just "it runs".

  ```bash
  python3 -m unittest discover tests
  ```

  - `test_runner_stack_correlation.py` -- regression test for the cross-hook
    `stk:` correlation race fixed in `src/core/runner.py`
    (`_remember_recent_span`/`_find_recent_span`): two different LD_PRELOAD
    hooks emitting spans from the same OS tid on their own independent
    socket connections must not clobber each other's call-stack attachment.
  - `test_pop_efficiency.py` -- `src/analysis/pop_efficiency.py`: Load
    Balance / Communication Efficiency exact-formula checks, the
    self-calibrated alpha/beta latency-bandwidth fit (verified against a
    known synthetic model), the NCCL bus-bandwidth formula, computational
    scaling, and edge cases (empty trace, single rank).
  - `test_criticalpath.py` -- `src/analysis/criticalpath.py`: each
    dependency-edge builder (program order, GPU stream order, device sync,
    MPI/NCCL point-to-point and collective pairing, OpenMP barrier
    rendezvous, explicit span_id/parent_span_id correlation) checked in
    isolation, plus a hand-computed 2-backend (MPI+CUDA) known-answer case
    and a causality-enforcement regression test (see below).

- **Integration matrix** (`tests/integration/run_matrix.sh`): builds the
  small per-backend fixture programs in `tests/fixtures/` (skipping any
  whose toolchain isn't available on the current machine), profiles each
  through its hprofiler backend, and runs `summary`, `efficiency`, and
  `critical-path` against the resulting trace -- asserting none of them
  crash end-to-end. Some backends legitimately capture 0 events on a given
  machine (no working GPU driver, `perf_event_paranoid` blocking
  perf/likwid, etc.) -- that's fine and not a failure here; a crash is.

  ```bash
  ./tests/integration/run_matrix.sh
  ```

  NCCL isn't in the automated matrix -- it needs a machine-specific
  workaround for a system without a system-wide `libnccl.so` (see
  `hprofiler backends`); if you have one, adapt the manual steps:
  ```bash
  LD_LIBRARY_PATH=<path to libnccl.so>:$LD_LIBRARY_PATH \
    python3 hprofiler run --backend nccl --no-ui -o nccl.json -- ./my_nccl_app
  python3 hprofiler critical-path nccl.json
  ```

## A real bug this test suite caught

While validating `criticalpath.py` against a real OpenMP trace (not just
synthetic unit tests), `hprofiler critical-path` reported "Wall time: 9.12ms,
Time accounted for: 10.989s" -- a ~1200x overshoot on a program that actually
ran for ~700ms. Two distinct bugs, both now covered by regression tests:

1. `Trace.duration_ns` is meaningless for a trace reconstructed by
   `load_trace_from_json` (its `TraceMetadata.start_time_ns` defaults to the
   *load* time, not the original run's start) -- `criticalpath.py` now
   derives wall time from the spans themselves instead, the same way
   `src/output/summary.py` and `src/analysis/cct.py` already did.
2. The backward critical-path walk didn't enforce causality: a rendezvous
   ("arrival"-gated, e.g. OpenMP barrier) predecessor could be picked even
   though it hadn't actually happened yet relative to the point being
   explained, and a full mutual clique between all rendezvous participants
   let the walk chain through arrival edges repeatedly instead of stopping
   at the single true "last arriver". See `TestCausalityEnforcement` and
   `_add_last_arriver_edges`'s docstring in `criticalpath.py` for the fix.

`hprofiler critical-path` on that same real trace now reports Wall time
702.90ms / Time accounted for 698.67ms -- sane.
