# hprofiler test suite

How to run everything is in
[DOCUMENTATION.md → Running the tests](../DOCUMENTATION.md#running-the-tests);
what each layer establishes is in
[Verification status](../DOCUMENTATION.md#verification-status). This file
maps the files to what they cover.

```bash
QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests -p 'test_*.py'   # unit + native
python3 -m unittest tests.integration.test_profiling_accuracy ...                 # integration, by name
python3 -m unittest discover -s tests/validation -p 'test_*.py'                   # causal-graph validation
bash tests/native/run_native_tests.sh                                             # ring-buffer stress
bash tests/integration/run_matrix.sh                                              # CLI end to end
```

Integration modules have no `__init__.py`, so discovery skips them; each
skips (rather than fails) when its toolchain or hardware is missing.

## Unit tests (`tests/test_*.py`)

Small synthetic traces with hand-computed expected answers.

| Area | Files |
|---|---|
| Collection | `test_receiver.py` (collector: counting, partial records, backlog), `test_wire_protocol.py` (`inst:` tags), `test_runner_stack_correlation.py` (`stk:` matching across hooks), `test_perf_script_parsing.py`, `test_peer_real_exe.py`, `test_zero_event_warning.py`, `test_transport_native.py` (builds `tests/native/*.c`: transport scenarios plain / ASan+UBSan / TSan, OpenCL trampolines) |
| Trace store and files | `test_trace_store_parity.py` (memory vs. disk store, every consumer), `test_store_errors.py` (damaged/interrupted traces), `test_chrome_trace_roundtrip.py`, `test_trace_lanes.py`, `test_otlp.py` |
| GPU host/device model | `test_gpu_activity.py` (correlation, de-duplication, timing sources, critical-path edges), `test_device.py`, `test_device_bandwidth.py` |
| Analyses | `test_criticalpath.py`, `test_pop_efficiency.py`, `test_cct.py`, `test_activity_buckets.py`, `test_multinode.py`, `test_compare.py`, `test_causal_compare.py` (scenarios in `compare_scenarios.py`), `test_call_tree_build.py`, `test_call_graph.py`, `test_flamegraph_tree.py`, `test_classifier_roofline.py`, `test_addr2line.py` |
| Disassembly | `test_disasm_categories.py` (call-site tags, `symfile=`, perf filter; one test compiles a real binary), `test_disasm_widget_message.py` |
| TUI (Textual `run_test`) | `test_dashboard.py`, `test_timeline_widget.py`, `test_timeline_connectors.py`, `test_braille_canvas.py`, `test_flamegraph_widget.py` |
| GUI (PySide6, skipped without it) | `test_gui_bridge.py`, `test_gui_models.py`, `test_gui_settings.py`, `test_gui_persistence.py`, `test_gui_loader.py`, `test_gui_controller.py`, `test_gui_errors.py`, `test_gui_logging.py`, `test_gui_launch.py`, `test_gui_shortcuts.py`, `test_x11_check.py`; real QML interaction in `test_gui_timeline_hover.py` (one shared engine for the whole module — a second engine in one process corrupts Qt Quick Controls), and in separate processes `test_gui_compare_launch.py`, `test_gui_compare_interaction.py` (driver: `gui_compare_driver.py`), `test_gui_timeline_waits.py` |

## Integration tests (`tests/integration/`)

Real programs (`tests/fixtures/`) through the real hooks and CLI.

| File | Covers | Needs |
|---|---|---|
| `test_profiling_accuracy.py` | timing against ground-truth programs (`*_truth.c`); `accuracy_report.py --trials N` prints the statistics | gcc/clang, mpicc, OpenCL |
| `test_gomp_hook.py` | GNU libgomp interception, per thread and construct | gcc |
| `test_mpi_protocol.py`, `test_mpi_rma.py` | MPI wire semantics (wildcards, requests, `commid=`, RMA synchronization) in a self-communicating process | mpicc |
| `test_callsite_e2e.py` | call sites → saved trace → disassembly, through a launcher | mpicc, clang, gcc, CMake |
| `test_native_gpu_records.py` | CUPTI / ROCprofiler-SDK decoders against the real vendor headers | the headers (`HPROFILER_CUPTI_INCLUDE`, `HPROFILER_ROCPROFILER_SDK_INCLUDE`) |
| `test_cuda_native_activity.py` | CUDA device activity on a real GPU, all three modes and a static-runtime build | nvcc, a CUDA GPU, CUPTI |
| `test_gui_cancel.py` | SIGINT during a GUI load exits 130 | PySide6 |
| `test_store_stress.py` | 2M-span store: bounded memory, window latency, exact answers (`HPROFILER_STRESS_EVENTS` to resize) | — |
| `run_matrix.sh` | `run` + `summary`/`efficiency`/`critical-path` per backend, crash-safety only | per-backend toolchains |

## Other

- `tests/validation/test_causal_accuracy.py` — precision/recall of the
  dependency graph against hand-built ground-truth edges, per confidence
  tier, plus determinism under reordered input.
- `tests/native/run_native_tests.sh` — `ringbuffer_stress.c`: the ring
  primitives under concurrent producers, exact overflow accounting, TSan,
  and a latency comparison.
- `tests/fixtures/` — the profiled programs (`*_mini`, `*_truth.c`,
  `mpi_proto*.c`, `mpi_win_self.c`, `*_callsite.c`, `omp_hotpath.c`,
  `cuda_streams.cu`).
