# hprofiler — Heterogeneous Profiler

Multi-device CPU/GPU profiler for Linux. Traces programs across CUDA, ROCm, OpenCL, OpenMP, NCCL, and MPI simultaneously — with a terminal UI (and an optional native Qt GUI, see [GUI Viewer](#gui-viewer) below), a live Flame Graph tab and a native TUI roofline viewer, and cross-layer causal attribution: one dependency graph over every backend active in a run, with a confidence-graded, formally-computed critical path instead of a single-runtime or heuristic one (see [Cross-Layer Causal Attribution](#cross-layer-causal-attribution) below). CPU sampling is provided via Linux `perf`.

## Requirements

- Python 3.10+, CMake 3.16+, GCC/Clang
- `pip install click textual rich numpy capstone` (or `pip install -r requirements.txt`)
- Optional GUI: `pip install "hprofiler[gui]"` (PySide6) plus the system library `libxcb-cursor0` / `xcb-util-cursor` — see [DOCUMENTATION.md](DOCUMENTATION.md#requirements)
- TUI roofline viewer: `pip install plotly "kaleido==0.2.1"` (0.2.1 specifically — later versions require Chrome and break on clusters); the Flame Graph tab needs neither
- Backend-specific: CUDA toolkit, a `libamdhip64` (ROCm/HIP) runtime, LLVM `libomp` or GNU `libgomp`, an MPI implementation (`mpicc`, or a Cray Programming Environment `cc` wrapper), or `perf` — see [DOCUMENTATION.md](DOCUMENTATION.md#requirements) for exact search paths

## Build

```bash
pip install -r requirements.txt
python3 hprofiler build
```

Produces `build/lib/libhprofiler_{cuda,opencl,ompt,gomp,rocm,nccl,mpi}.so`.

## Quick Start

```bash
# Profile with auto-detected backends
python3 hprofiler run -- ./my_program

# Specific backends
python3 hprofiler run --backend cuda,cpu      -- ./cuda_app
python3 hprofiler run --backend openmp        -- ./omp_app     # LLVM libomp or GNU libgomp
python3 hprofiler run --backend rocm          -- ./hip_app
python3 hprofiler run --backend cuda,nccl     -- ./multi_gpu_app
python3 hprofiler run --backend mpi           -- mpirun -np 4 ./mpi_app

# Call tree (adds Call Tree tab; compile with -fno-omit-frame-pointer -rdynamic)
python3 hprofiler run --call-tree --backend cuda -- ./app

# Per-kernel disassembly (adds Source tab)
python3 hprofiler run --backend cuda --disasm -- ./app

# Instruction-level GPU heat map + stall annotation (CUDA, libcupti.so loaded at runtime)
python3 hprofiler run --backend cuda --disasm --gpu-pc-sampling -- ./app

# Instruction-level CPU heat (OpenCL CPU runtime via ACPP)
ACPP_VISIBILITY_MASK=ocl python3 hprofiler run --backend opencl,cpu --disasm -- ./app

# Save trace, skip TUI
python3 hprofiler run --no-ui -o trace.json -- ./app

# Open a saved trace
python3 hprofiler view trace.json

# Text summary only
python3 hprofiler summary trace.json

# Native Qt GUI instead of the TUI (falls back to the TUI automatically
# if PySide6/X11 aren't available — see GUI Viewer below)
python3 hprofiler run --gui --backend cuda -- ./cuda_app
python3 hprofiler gui trace.json

# Flame graph — populates the Flame Graph tab in both the TUI and the GUI
python3 hprofiler run --perf-callgraph dwarf -- ./my_program
python3 hprofiler run --backend cuda --perf-callgraph dwarf --call-tree -- ./cuda_app  # + GPU API overhead

# Roofline chart — opens TUI viewer by default (requires plotly + kaleido)
python3 hprofiler roofline --backend cuda    -- ./cuda_app
python3 hprofiler roofline --backend cpu     -- ./cpu_app
python3 hprofiler roofline --backend rocm    -- ./hip_app
python3 hprofiler roofline --html --backend cuda -- ./cuda_app  # write HTML + open browser

# Hardware PMU counters via LIKWID
HPROFILER_LIKWID_GROUP=MEM python3 hprofiler run --backend likwid -- ./app

# List available backends on this machine
python3 hprofiler backends
```

Always separate hprofiler options from the target program with `--`.

## Backends

| Name | Alias | Injection | What is traced |
|------|-------|-----------|----------------|
| `cpu` | `perf` | `perf record` subprocess | CPU samples, optional DWARF/fp/lbr call-graph |
| `cuda` | — | LD_PRELOAD | Kernel launches, memcpy, syncs, NVTX ranges, memory counters |
| `opencl` | `cl` | LD_PRELOAD | Kernel enqueues (host side) and device execution, buffer transfers, JIT compile time |
| `openmp` | `omp` | `OMP_TOOL_LIBRARIES` (OMPT, LLVM `libomp`) and LD_PRELOAD (`GOMP_*`, GNU `libgomp`) | Parallel regions with each thread's share, loops, tasks, barriers, critical sections — whichever runtime the binary links is covered |
| `rocm` | `hip` | LD_PRELOAD | HIP kernel launches, memcpy, memory counters |
| `nccl` | — | LD_PRELOAD | Collectives (AllReduce, Broadcast, …), point-to-point — GPU-accurate timing |
| `mpi` | — | PMPI / LD_PRELOAD | Send/Recv, collectives, one-sided ops — wall-clock timing |
| `likwid` | `hwc` | `likwid-perfctr` wrapper | Hardware PMU counters: FLOPS, DRAM bandwidth, cache rates, CPI |

## TUI Viewer

Opens automatically after `hprofiler run`. Tabs:

| Tab | When shown | Contents |
|-----|-----------|---------|
| Overview | Always | Diagnosis, wall time, GPU active %, MPI/sync wait %, peak memory, top findings, hot kernels, source correlation |
| Timeline | Always | Gantt view: per-thread lanes, per-stream CUDA/ROCm lanes, per-process lanes when processes share an id |
| Kernels | Always | Filterable/sortable function table |
| Call Tree | Only with `--call-tree` and/or `--perf-callgraph fp\|dwarf\|lbr` | Stack-frame tree from captured call graphs |
| Flame Graph | Same condition as Call Tree | Proportional icicle chart of the same call-stack data — see [Flame Graph Tab Controls](#flame-graph-tab-controls) |
| Roofline | Only when kernel metrics are available | Kernels on the device's roofline |
| Source | With `--disasm`, or a saved trace that contains disassembly | Per-kernel assembly with instruction-type color coding, runtime heat % and stall columns (CPU via perf, CUDA via `--gpu-pc-sampling`), and static optimization hints |
| System | Always | Device specs, FP16/32/64/Tensor TFLOP/s, bandwidth, IPC, LLC/branch miss rates, RSS |
| Profile | Always | GPU activity %, time breakdown by category, top hotspots, bottleneck advisor |

## GUI Viewer

An optional native Qt/QML desktop GUI (`pip install "hprofiler[gui]"`) covering the same tabs as the TUI (including Flame Graph), plus GUI-specific additions: smooth wheel-zoom/drag-pan, filtering, grouping/collapsing, event search, and bookmarks/named ranges on the Timeline; real sortable/filterable/exportable tables (Kernels, System, Call Tree, Overview) instead of hand-rolled lists; a 10th **Compare** tab for diffing two runs (matched by stable `(category,name)` identifiers, with a disclosed noise-floor threshold — not a statistical test — for improved/regressed classification); an idle-time overlay so a span blocked at a nested barrier/sync call visibly shows that within its own bar instead of looking continuously busy; and a 3-panel Source tab with instruction-mix/static-advisor analysis alongside the assembly.

```bash
hprofiler run --gui --backend cuda -- ./cuda_app
hprofiler gui trace.hprofiler.json
hprofiler run --gui --perf-callgraph dwarf -- ./app   # + populate the Flame Graph tab
hprofiler gui after.hprofiler.json --compare before.hprofiler.json   # Compare tab
```

- **Loading** happens on a background thread before the window opens, with stage and percentage progress in the terminal. **Ctrl+C cancels it** (exit code 130). A trace that can't be loaded gets a short, classified error message plus the path to the GUI log (`~/.local/share/hprofiler/hprofiler/hprofiler-gui.log`) instead of a traceback.
- **File → Open Profile… (Ctrl+O)** opens another trace in a new GUI process; the current window stays usable and only closes once the new one is open. If opening fails, the current workspace is untouched and the error is shown with *Show technical details*, *Copy diagnostics* and *Open log* actions.
- **Command palette (Ctrl+K)** — jump to a tab, run a command, or find a function/kernel by name. **F1** (Help → Keyboard Shortcuts) lists every shortcut.
- Tabs without data show an explicit empty/unsupported state rather than a blank panel; icon-only controls have tooltips and accessible names; the Timeline has a dismissible first-use tip and a collapsible activity-colour legend.
- Window geometry, theme, the legend state and dismissed tips persist (`~/.config/hprofiler/hprofiler.conf`). Timeline view state (filters, zoom, bookmarks) and table layouts are **not** restored yet. The profiled command line is never written to that file.

Falls back to the TUI automatically — no error shown — if PySide6 isn't installed, X11 isn't reachable, or GPU-rendered Qt Quick fails over indirect/forwarded X11 (retried once with software rendering first). See [DOCUMENTATION.md](DOCUMENTATION.md) §20 for the full tab reference (Timeline exploration, Tables, Comparison mode, menus and shortcuts) and §2 for install/troubleshooting (including the `libxcb-cursor0` system-library requirement and a VNC fallback for machines where installing it isn't an option).

## Flame Graph Tab Controls

Works in any terminal — plain character-cell rendering, no inline-image protocol required (unlike the roofline viewer below). Same controls in the GUI's Flame Graph tab, mouse-driven there too.

| Action | TUI | GUI |
|--------|-----|-----|
| Zoom into a frame | Click | Click |
| Zoom out one level | Right-click or Backspace | Right-click |
| Reset to full view | Escape | Escape / Reset button |
| Search — regex-highlight matching frames | Type in the search box | Type in the search box |

## Roofline TUI Controls

| Key | Action |
|-----|--------|
| `n` / `p` | Cycle through kernels (shows crosshairs with headroom annotation) |
| Esc | Deselect kernel / hide crosshairs |
| `+` / `=` | Zoom in |
| `-` | Zoom out |
| `←` `→` `↑` `↓` | Pan |
| `r` | Reset zoom |
| `w` | Open HTML version in browser |
| `q` | Quit |

## Output Files

| File | Viewer |
|------|--------|
| `<prog>.hprofiler.json` | [Perfetto](https://ui.perfetto.dev) or `chrome://tracing` |
| `<prog>.roofline.html` | Any browser (self-contained) |

The JSON trace is lossless for hprofiler itself: reloading it (as the GUI, `view`, `summary`, `efficiency`, `critical-path` and `merge-nodes` all do) reproduces the in-memory trace from `hprofiler run`, including span/request ids, real GPU thread ids and the profiling window.

## How Time Is Attributed

The Overview/Profile time breakdowns, the one-line diagnosis ("openmp-bound", …) and "*f* dominates" findings count **exclusive** time: on each thread, every instant belongs to the innermost instrumented call covering it, so a barrier inside a parallel region counts once, as synchronization. GPU/device work is never counted against the thread that launched it; perf samples only count when their thread isn't inside an instrumented call; shares are relative to available thread-time (threads × wall), so a program that spends 2% of its run in MPI is not called "mpi-bound". The MPI/sync **wait %** is averaged per rank/thread (the old union across threads read ~80% for threads that waited ~38%). The Kernels/Hotspots tables still show inclusive per-function totals. Details: [DOCUMENTATION.md](DOCUMENTATION.md) §5 (Overview Tab).

## Measurement Accuracy

Validated against small ground-truth programs that log their own `CLOCK_MONOTONIC` timestamps (`tests/fixtures/*_truth.c`), 10 trials each on a laptop CPU:

| Workload | Slowdown of the program | Missing events | Start-time error (median / p95) |
|---|---|---|---|
| OpenMP, LLVM libomp (OMPT) | +0.2 % … +1.1 % | 0 / 240 | 1.9 µs / 4.7 µs |
| OpenMP, GNU libgomp | +0.08 % | 0 / 240 | 0.25 µs / 1.4 µs |
| MPI (single rank) | +1.2 % | 0 / 240 | 0.22 µs / 1.7 µs |
| OpenCL (Intel CPU device) | within noise | 0 / 50 | device kernel time matches `CL_PROFILING` exactly |

About 5 µs per intercepted call on call-bound loops; no events dropped at 200 000 MPI calls. CUDA/ROCm/NCCL, real multi-rank MPI and perf sampling could not be validated on the development machine. Known artifacts (e.g. CUDA/ROCm kernel spans start at the host launch time) are listed in [DOCUMENTATION.md](DOCUMENTATION.md) §13 "Measured accuracy".

## OpenTelemetry Export

Export spans and metrics to any [OTLP](https://opentelemetry.io/docs/specs/otlp/)-compatible collector. No extra Python dependencies — uses stdlib only.

```bash
# Send live to a local collector (Grafana Alloy, otelcol, Jaeger ≥ 1.35, Tempo, …)
python3 hprofiler run --backend cuda --otlp-endpoint http://localhost:4318 -- ./app

# Write OTLP JSON to file (replay later with curl)
python3 hprofiler run --backend cuda --otlp-file trace.otlp.json -- ./app

# Export from a saved trace
python3 hprofiler view --otlp-endpoint http://localhost:4318 app.hprofiler.json
python3 hprofiler view --otlp-file trace.otlp.json app.hprofiler.json

# Replay a saved OTLP file to a collector
curl -X POST http://localhost:4318/v1/traces \
     -H 'Content-Type: application/json' -d @trace.otlp.json
```

OTLP mapping: each `SpanEvent` becomes an OTLP span (all root-level, no parent inference); `CounterEvent` values (IPC, bandwidth, memory usage) become OTLP gauge metrics sent to `/v1/metrics`; hprofiler category, tags, PID, and TID become span attributes.

## Disassembly

Pass `--disasm` to collect post-run per-kernel disassembly (runs in background, TUI opens immediately):

| Backend | Tool needed |
|---------|-------------|
| CUDA AoT | `cuobjdump` (CUDA toolkit) |
| CUDA JIT (ACPP) | Built-in PTX parser |
| ROCm | `llvm-objdump` (`apt install llvm`) |
| CPU / OpenMP | `capstone` (`pip install capstone`) or `objdump` |
| OpenCL JIT (ACPP SSCP generic) | `objdump` on the `.jit.so` emitted by ACPP SSCP |
| OpenCL CPU (Intel CPU OCL) | `objdump` on x86-64 ELF extracted via `clGetProgramInfo` |

## Cross-Layer Causal Attribution

hprofiler's core contribution: for programs combining several backends at
once (e.g. MPI+OpenMP+CUDA), it builds one dependency graph over every
captured span — CUDA, ROCm, OpenCL, OpenMP, MPI, NCCL together, not a
separate per-runtime trace to merge — using each programming model's real
synchronization semantics (resolved MPI wildcard matching, real
communicator identity, stream/device-sync ordering, OpenMP barriers), and
finds the true critical path via a formal DAG longest-path computation,
not a heuristic walk. Every edge in that graph is tagged with how directly
the underlying data proves it (`certain`/`high`/`medium` — see
[DOCUMENTATION.md](DOCUMENTATION.md) §17), so the result never presents a
call-order guess with the same confidence as a hardware-enforced ordering.

```bash
# POP-style parallel efficiency breakdown (Load Balance, Communication
# Efficiency, GPU/NCCL efficiency, ...) computed from a single trace
python3 hprofiler efficiency trace.json

# N-way cross-runtime critical path + blame attribution across every
# backend active in the trace (generalizes CASITA/HPCToolkit-style
# critical-path analysis beyond MPI+CUDA-only or CPU+GPU-only), with a
# per-hop confidence breakdown
python3 hprofiler critical-path trace.json

# Multi-node: profile each node separately, then merge onto one timeline
# before running critical-path/efficiency across node boundaries
python3 hprofiler merge-nodes node0.json node1.json node2.json -o merged.json
python3 hprofiler critical-path merged.json
```

See [DOCUMENTATION.md](DOCUMENTATION.md) §16–19 for the exact formulas,
edge-confidence model, formal critical-path algorithm, multi-node clock
synchronization, and what's approximate vs. exact vs. still
hardware-unverified on this development machine (documented honestly, not
glossed over — see §13's Known Limitations table).

## Tests

```bash
# Unit tests (~12 s)
QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests -p 'test_*.py'

# Integration tests: real programs through the real hooks (skipped when a
# toolchain is missing). Not found by discovery -- run them by name.
python3 -m unittest tests.integration.test_profiling_accuracy \
    tests.integration.test_gomp_hook tests.integration.test_mpi_protocol \
    tests.integration.test_mpi_rma tests.integration.test_gui_cancel
bash tests/integration/run_matrix.sh                        # CLI end-to-end, per backend
python3 tests/integration/accuracy_report.py --trials 10    # accuracy statistics
```

## Recent Changes

- **Removed:** the LLM/AI performance analysis (`hprofiler analyze`, and `--analyze`/`--llm*`/`--analysis-report` on `run`) and `setup_llm.sh`.
- **Trace files are now lossless** — span/request ids, real GPU thread ids, the profiling window, instant tags and counter units were previously dropped on save, so `critical-path` on a saved MPI trace lost every communication dependency, and "Total time" after reloading was the time since the file was opened.
- **Consistent time accounting** (see [How Time Is Attributed](#how-time-is-attributed)): the same OpenMP program was previously diagnosed "sync-bound" under LLVM libomp and "openmp-bound" under GNU libgomp; the OMPT tool now also records each thread's share of a parallel region.
- **Critical path** no longer breaks at nested spans (it explained only 28% of a busy OpenMP program's run time).
- **Timeline** gives each process its own lane when processes share a stream or thread id (every MPI rank's default CUDA stream used to be drawn in one lane).
- **Compare tab:** a comparison row can no longer be matched twice; changing thresholds updates every panel and the export; missing values show "—" instead of 0.
- **GUI:** background loading with progress and Ctrl+C cancellation, classified errors with diagnostics and a log file, Open Profile in a new process, command palette, shortcuts reference, menus, first-use tip, collapsible legend, accessibility names, and persisted window/theme settings.
- **Accuracy test suite** against ground-truth programs, with explicit tolerances (see [Measurement Accuracy](#measurement-accuracy)).

See [DOCUMENTATION.md](DOCUMENTATION.md) for the full CLI reference, backend details, wire protocol, and how to extend the profiler.
