# hprofiler test suite

Four layers:

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
  - `test_wire_protocol.py` -- `_parse_record`'s `inst:` tag-segment parsing
    (`src/core/runner.py`): regression test for a bug where instant events'
    trailing tags were silently discarded entirely.
  - `test_pop_efficiency.py` -- `src/analysis/pop_efficiency.py`: Load
    Balance / Communication Efficiency exact-formula checks, the
    self-calibrated alpha/beta latency-bandwidth fit (verified against a
    known synthetic model), the NCCL bus-bandwidth formula, computational
    scaling, and edge cases (empty trace, single rank).
  - `test_criticalpath.py` -- `src/analysis/criticalpath.py`: each
    dependency-edge builder (program order, GPU stream order, device sync,
    MPI/NCCL point-to-point and collective pairing -- including resolved
    wildcard matches, commid=-scoped rendezvous, and async Isend/Irecv+Wait*
    pairing -- OpenMP barrier rendezvous, explicit span_id/parent_span_id
    correlation including `;`-separated multi-request lists), edge
    confidence tiers, the formal DAG longest-path DP (including a
    hand-verified case where it finds a materially better answer than the
    old greedy walk on the same graph), cycle-fallback safety, and
    `xs=`-aware exec-start gap correction, plus a hand-computed 2-backend
    (MPI+CUDA) known-answer case and a causality-enforcement regression
    test (see below).
  - `test_multinode.py` -- `src/analysis/multinode.py`: Cristian's-algorithm
    clock-offset arithmetic (including an asymmetric-latency case proving
    the returned error bound actually brackets the real error, not just the
    symmetric exact case), trace merging (pid remapping avoids cross-node
    collisions, MPI rank=/peer= tags deliberately left unremapped), and
    post-merge causality validation.

- **Native (C-level) stress tests** (`tests/native/`): infrastructure with
  no GPU/MPI dependency, verified with much stronger tools than the Python
  suite can apply.

  ```bash
  bash tests/native/run_native_tests.sh
  ```

  `ringbuffer_stress.c` -- `hooks/common/ringbuffer.h`'s lock-free
  per-thread ring buffer, arena allocator, and name-interning table:
  concurrent producer/consumer correctness under real pthread scheduling,
  drop-counter exactness under intentional overflow, FIFO order across
  repeated wrap-around, and (best-effort, environment permitting) a clean
  ThreadSanitizer pass -- plus a real, measured latency comparison against
  the mutex+`send()` pattern it's designed to replace.

- **Integration tests** (`tests/integration/`):

  - `run_matrix.sh` -- crash-safety matrix: builds the small per-backend
    fixture programs in `tests/fixtures/` (skipping any whose toolchain
    isn't available on the current machine), profiles each through its
    hprofiler backend, and runs `summary`, `efficiency`, and
    `critical-path` against the resulting trace -- asserting none of them
    crash end-to-end. Some backends legitimately capture 0 events on a
    given machine (no working GPU driver, `perf_event_paranoid` blocking
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

  - `test_mpi_protocol.py` -- unlike everything else here, this is
    *measurement-correctness* verification, not a crash check: builds the
    real `libhprofiler_mpi.so`, `LD_PRELOAD`s it into a self-communicating
    fixture, captures the actual wire-protocol bytes over a real
    `AF_UNIX` socket, and parses them with the production `_parse_record`
    -- asserting on resolved semantics (wildcard matching, `Waitany`/
    `Waitsome`/`Test*` completion, `commid=` self-consistency), not just
    that the process didn't crash.

    ```bash
    python3 -m unittest tests.integration.test_mpi_protocol -v
    ```

- **Validation suite** (`tests/validation/`): the direct response to "these
  tests mostly verify 'does not crash', not correctness" -- an aggregate,
  quantitative precision/recall report (per confidence tier) across a
  battery of synthetic scenarios with known ground-truth causal edges,
  plus a determinism check (the same scenario, spans inserted in several
  different orders, must produce byte-identical results). Not part of
  `unittest discover tests` (a different directory, run separately):

  ```bash
  python3 -m unittest discover -s tests/validation -p "test_*.py" -v
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
