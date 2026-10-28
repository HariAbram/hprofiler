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
| Collection | `test_receiver.py` (collector: counting, partial records, backlog), `test_wire_protocol.py` (`inst:` tags), `test_runner_stack_correlation.py` (`stk:` matching across hooks), `test_perf_script_parsing.py`, `test_peer_real_exe.py`, `test_zero_event_warning.py`, `test_transport_native.py` (builds `tests/native/*.c`: transport scenarios plain / ASan+UBSan / TSan, OpenCL trampolines and kernel names), `test_runner_process.py` (descriptor inheritance, exit status, collector socket fallback incl. a real hook on an abstract socket) |
| Trace store and files | `test_trace_store_parity.py` (memory vs. disk store, every consumer), `test_store_errors.py` (damaged/interrupted traces), `test_chrome_trace_roundtrip.py`, `test_trace_lanes.py`, `test_sycl_category.py` (host-run SYCL kernels are `sycl`, not `opencl`: old hook records, old JSON, one-time relabel of existing stores), `test_otlp.py` |
| GPU host/device model | `test_gpu_activity.py` (correlation incl. unique ids before their submission and wrapped ids, device-clock drift correction -- synthetic and the real drifting run in `fixtures/cupti_drift_mixbench.json` --, de-duplication, timing sources, critical-path edges), `test_device.py`, `test_device_bandwidth.py` |
| MPI matching | `test_mpi_matching.py` (world-rank + communicator keys, legacy key, per-process request ids, cancellation, send variants, `Sendrecv`, diagnostics, the coverage CLI output; both edge builders), `test_exact_p2p_rule.py` (the opt-in exact-match rule and each of its conditions) |
| Backends | `test_backend_probes.py` (installed vs usable: CUDA/ROCm/OpenCL/LIKWID probes with mocked hardware states, auto-detection, `hprofiler backends`), `test_python_jit.py` (`--python-jit`: Triton compile spans and cache hits from a stub `triton`, the program's own sitecustomize and listener chained) |
| Analyses | `test_cp_model.py` (event-point critical-path model: graph and time-accounting invariants on random multi-rank traces, checked against a brute force over all paths -- modeled length and bounds, every/some-path sets, slack, counterfactuals; overlapping and nested spans, start-gated dependencies, fork/join, ambiguous and contradicting edges, equal-length paths and ties, provenance), `test_criticalpath.py`, `test_criticalpath_comb.py` (annotation ranges are no graph nodes; path time = time covered, each instant once), `test_criticalpath_predictions.py` (devices in stream keys, device syncs, CUDA graph kernels, OpenMP teams and non-barrier waits, event loss, NCCL communicators; both edge builders), `test_evidence.py` (evidence model: the default view equals the builders' edges, statuses, match groups, explain(), persistence), `test_cp_clock.py` (clock domains: the offset solver, a shifted OpenCL clock, cross-node contradictions, undecidable steps), `test_cp_alternatives.py` (alternative worlds: only admissible pairings, each send completing at most one receive in order; the verdict), `test_cp_attribution.py` (blame with joint releases, replay counterfactuals), `test_edge_calibration.py` (isotonic calibration method, ECE / Brier / log loss), `test_semantics_mpi.py`, `test_semantics_gpu.py`, `test_semantics_runtimes.py` (runtime semantics edges: MPI persistent / non-blocking / rooted collectives, CUDA host functions / blocking copies / legacy stream, OpenCL queues, NCCL pairs and groups, OpenMP task joins / dependences / locks; both edge builders), `test_critical_path_cli.py` (`critical-path` text, `--json`, `--export` tags), `test_timeline_analysis_evidence.py` (the Timeline's evidence fields), `test_pop_efficiency.py`, `test_cct.py`, `test_activity_buckets.py`, `test_multinode.py` (clock offsets: same boot_id, drift, asymmetric latency, message-order tightening, merge-nodes CLI), `test_compare.py`, `test_causal_compare.py` (scenarios in `compare_scenarios.py`), `test_call_tree_build.py`, `test_span_links.py` (span ids shared by two hooks' spans: call tree a forest, critical-path links by hook family, `hp_ids.h` spaces differ), `test_call_graph.py`, `test_flamegraph_tree.py`, `test_classifier_roofline.py`, `test_addr2line.py` |
| Disassembly | `test_disasm_categories.py` (call-site tags, `symfile=`, perf filter; one test compiles a real binary), `test_disasm_widget_message.py`, `test_disasm_modules.py` (module ids: the C table `tests/native/hp_module_test.c` plain and ASan+UBSan incl. ELF / fat-binary image sizes, lookup/flags, serialization, the runner, launch counts by module / by name / unknown, which executables AoT code is read from under any hash seed, the extractor on two real ELF objects, JIT stages) |
| TUI (Textual `run_test`) | `test_dashboard.py`, `test_timeline_widget.py`, `test_timeline_connectors.py`, `test_braille_canvas.py`, `test_flamegraph_widget.py` |
| GUI (PySide6, skipped without it) | `test_timeline_render.py` (timeline level of detail, hierarchy, painting pixels, critical-path band, span prefetch parity, comparison matching -- no QML scene), `test_gui_bridge.py`, `test_gui_models.py`, `test_gui_settings.py`, `test_gui_persistence.py`, `test_gui_loader.py`, `test_gui_controller.py`, `test_gui_errors.py`, `test_gui_logging.py`, `test_gui_launch.py`, `test_gui_shortcuts.py`, `test_x11_check.py`, `test_timeline_gui_evidence.py` (the critical-path model as the Timeline presents it: alternative worlds not evaluated / evaluated on request / none found / truncated, no false "every world", variants and the modeled range with no bound labels, unknown vs gated blocking time in data and pixels, per-event gated tails, contradicted / dropped edges and undecidable steps, blame incl. joint releases and passed-on waiting, sensitivity / replay, uncalibrated and calibrated probabilities, membership marks; scenarios in `evidence_scenarios.py`), `test_gui_compare_joint.py` (the Compare bridge on a joint release: every origin incl. one off the path, direct / jointly caused / wait blame apart, navigation); real QML interaction in `test_gui_timeline_hover.py` (one shared engine for the whole module — a second engine in one process corrupts Qt Quick Controls), and in separate processes `test_gui_compare_launch.py`, `test_gui_compare_interaction.py` (driver: `gui_compare_driver.py`), `test_gui_evidence_interaction.py` (drivers: `gui_evidence_driver.py` in default / large-trace / calibrated modes, `gui_compare_joint_driver.py`; also writes screenshots with a SHOT_DIR argument), `test_gui_text_overflow.py` (driver: `gui_text_overflow_driver.py`; no text past its popup, card or panel at a 13 pt font in a 1100 px window and at the default font), `test_gui_timeline_waits.py` |

## Integration tests (`tests/integration/`)

Real programs (`tests/fixtures/`) through the real hooks and CLI.

| File | Covers | Needs |
|---|---|---|
| `test_profiling_accuracy.py` | timing against ground-truth programs (`*_truth.c`); `accuracy_report.py --trials N` prints the statistics | gcc/clang, mpicc, OpenCL |
| `test_gomp_hook.py` | GNU libgomp interception, per thread and construct | gcc |
| `test_mpi_protocol.py`, `test_mpi_rma.py` | MPI wire semantics (wildcards, requests, `commid=`, RMA synchronization) in a self-communicating process | mpicc |
| `test_mpi_multirank.py` | 4 real ranks (`mpi_status.c`): statuses and cancellation as the program sees them, sub-communicators, `Comm_dup`, `Sendrecv*`, newly wrapped calls, no unmatched receive; (`mpi_proto.c`): wildcard resolution through `Waitany`/`Waitsome`, `Test` polling, cancellation, cross-process `commid=` agreement, the dependency edges -- against the fixture's ground truth, unprofiled and profiled | an MPI whose launcher starts 4 ranks (Intel MPI via `HPROFILER_TEST_MPIVARS`, default `/opt/intel/oneapi/mpi/latest/env/vars.sh`) |
| `test_opencl_names.py`, `test_opencl_modules.py` | OpenCL kernel names of any length; module ids and per-module disassembly (`ocl_variants.c`), `stage=` | gcc, OpenCL headers and a device |
| `test_gomp_teams.py` | barriers of two concurrent nested teams stay separate rendezvous (`gomp_teams.c`) | gcc |
| `test_openmp_sync.py` | tasks (GNU trampoline), taskwait/taskgroup, locks, critical, ordered and task dependences counted exactly for GNU libgomp and LLVM libomp, output unchanged (`omp_sync_counts.c`); OMPT sync-region kinds (`sk=`), task parents and the task graph (51 joins, 1 dependence); explicit tasks only, one non-fragment `omp_task` per task executed (`omp_task_counts.c`: nested task, taskwait, taskyield) | gcc, clang |
| `test_mpi_persist.py` | 4 real ranks (`mpi_persist.c`): `MPI_Start`/`Startall` posts and their pairing, rooted collectives' `root=`/`wroot=` on a split communicator and their data-flow edges, non-blocking collectives completed by `MPI_Wait` | as `test_mpi_multirank.py` |
| `test_opencl_deps.py` | queue / event / wait-list ids, blocking transfers, markers, barriers, `clFinish`, `clWaitForEvents`, and the queue edges (`ocl_deps.c`) | gcc, OpenCL headers and a device |
| `test_cuda_deps.py` | `cudaMemcpy`'s `blocking=` per direction, stream creation flags, host functions as stream-ordered spans, and the legacy-stream / host-function / blocking-copy edges, with CUPTI and with the proxy (`cuda_deps.cu`) | nvcc, a CUDA GPU |
| `test_nccl_hook.py` | the NCCL hook against a stand-in libnccl (`nccl_mock.cu`, `nccl_prog.c`): `pseq=`, grouped operations timed per stream, asynchronous timing | nvcc, a CUDA GPU |
| `test_callsite_e2e.py` | call sites → saved trace → disassembly, through a launcher | mpicc, clang, gcc, CMake |
| `test_native_gpu_records.py` | CUPTI / ROCprofiler-SDK decoders against the real vendor headers | the headers (`HPROFILER_CUPTI_INCLUDE`, `HPROFILER_ROCPROFILER_SDK_INCLUDE`) |
| `test_cuda_native_activity.py` | CUDA device activity on a real GPU, all three modes and a static-runtime build | nvcc, a CUDA GPU, CUPTI |
| `test_cuda_module_launches.py` | every kernel span carries its module id and every disassembly entry counts its launches: Triton's `cuLaunchKernelEx`-via-`dlsym` pattern (`cuda_launch_ex.c`), an AoT kernel with a shared cudart (its fat binary's id), a static cudart behind a forking wrapper and `env` (5 runs, same disassembly) | gcc, nvcc, CUDA headers, a CUDA GPU |
| `test_gui_cancel.py` | SIGINT during a GUI load exits 130 | PySide6 |
| `test_store_stress.py` | 2M-span store: bounded memory, window latency, exact answers (`HPROFILER_STRESS_EVENTS` to resize) | — |
| `run_matrix.sh` | `run` + `summary`/`efficiency`/`critical-path` per backend, crash-safety only | per-backend toolchains |

## Other

- `tests/validation/test_causal_accuracy.py` — precision/recall of the
  dependency graph against hand-built ground-truth edges, per confidence
  tier, plus determinism under reordered input.
- `tests/validation/test_edge_calibration.py` — the calibration method on
  simulated MPI programs with known pairings and lost send records: fit on
  some seeds, ECE / Brier / log loss on others (synthetic data only).

Integration tests that need hook libraries look in `HPROFILER_HOOKS_DIR`
when set (else `build/lib`): `HPROFILER_HOOKS_DIR=<scratch build>/lib
python3 -m unittest tests.integration.<module>` runs them against a fresh
build without touching `build/`. Tests of tags a hook library predates skip
with "the hook library predates ...".
- `tests/native/run_native_tests.sh` — `ringbuffer_stress.c`: the ring
  primitives under concurrent producers, exact overflow accounting, TSan,
  and a latency comparison.
- `tests/fixtures/` — the profiled programs (`*_mini`, `*_truth.c`,
  `mpi_proto*.c`, `mpi_status.c`, `mpi_win_self.c`, `*_callsite.c`,
  `omp_hotpath.c`, `ocl_names.c`, `ocl_variants.c`, `ocl_deps.c`, `gomp_teams.c`, `omp_sync_counts.c`,
  `mpi_persist.c`, `cuda_streams.cu`, `cuda_deps.cu`, `nccl_mock.cu`, `nccl_prog.c`).
