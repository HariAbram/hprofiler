# hprofiler

A CPU/GPU profiler for Linux that records CUDA, ROCm/HIP, OpenCL, OpenMP,
NCCL and MPI activity — plus `perf` CPU samples — from one run into one
trace, and explains it: a cross-runtime critical path, POP-style efficiency
metrics, and a structure-aware comparison of two runs. Traces open in a
terminal UI, an optional Qt GUI, text reports, or Perfetto.

This README covers installation, a first run, the common workflows and how
to read the output. [DOCUMENTATION.md](DOCUMENTATION.md) is the reference
manual; `hprofiler <command> --help` gives exact option syntax.

## 1. What it does

- **Captures** host API calls and device work for every backend active in a
  run, through hook libraries injected with `LD_PRELOAD` (and the OpenMP
  tools interface). No recompilation is needed; nothing is linked into
  your program.
- **Separates host from device time**: a CUDA/HIP launch is a host span on
  the calling thread *and* a device span on its stream, with the device
  span measured by CUPTI / ROCprofiler-SDK when available.
- **Labels provenance**: every value is measured, a proxy, derived, or a
  heuristic estimate, and every capture reports what it lost.
- **Explains** the run: one dependency graph across all runtimes, each
  edge graded by how directly the data proves it, and a formal
  critical path with blame for idle time; POP-style efficiency; ranked
  causes when comparing two runs.
- **Scales**: events stream into an indexed on-disk store (`.hpstore`)
  while the program runs; viewers read only the visible window.

## 2. Install and requirements

- Linux, Python ≥ 3.10, CMake ≥ 3.16, GCC or Clang.
- Python packages: `pip install -r requirements.txt` (click, textual, rich,
  capstone, numpy).
- Optional GUI: `pip install "hprofiler[gui]"` (PySide6 ≥ 6.5) **and** the
  system library `libxcb-cursor0` / `xcb-util-cursor`. Without them every
  GUI command falls back to the terminal UI.
- Optional roofline TUI: `pip install plotly "kaleido==0.2.1"` (exactly
  0.2.1 — newer versions need Chrome).

What each backend needs at run time:

| Backend | Needs |
|---|---|
| `cpu` | `perf`, with `kernel.perf_event_paranoid` ≤ 1 |
| `cuda` | NVIDIA driver; for device-measured timing, `libcupti` (CUDA toolkit) and CUPTI headers at build time |
| `rocm` | `libamdhip64`; for device-measured timing, ROCprofiler-SDK (ROCm ≥ 6.2) |
| `opencl` | any OpenCL ICD loader |
| `openmp` | a program linked against LLVM `libomp` or GNU `libgomp` |
| `nccl` | `libnccl` + the CUDA runtime |
| `mpi` | an MPI implementation (`mpicc`, or a Cray PE `cc`) when building |
| `likwid` | `likwid-perfctr` with PMU access |

Exact search paths, remote-display (VNC) setup and troubleshooting are in
[DOCUMENTATION.md → Installation](DOCUMENTATION.md#1-installation-and-build).

## 3. Build

```bash
cd hprofiler
pip install -r requirements.txt
./hprofiler build        # compiles build/lib/libhprofiler_*.so with CMake
./hprofiler backends     # shows which backends are usable on this machine
```

`hprofiler` runs from the repository without installation; the examples
below call it as `hprofiler` (add the directory to your `PATH`, or use
`./hprofiler`). The build picks up optional inputs automatically:
libunwind (accurate `--call-tree` stacks), CUPTI and ROCprofiler-SDK
headers (device-measured GPU timing), and MPI. Rebuild after pulling new
hook code.

## 4. Quick start and workflows

Always put `--` between hprofiler's options and your program.

### First run

```bash
hprofiler run -- ./app
```

This enables every available backend, runs `./app`, prints a summary and
opens the terminal UI. The trace is saved as `app.hprofiler.hpstore` (the
indexed store) and `app.hprofiler.json` (Chrome Trace JSON). If
`likwid-perfctr` is installed but cannot access the PMUs, auto-detection
still selects it and the program never starts — pass `--backend` explicitly
in that case.

### Choosing backends

```bash
hprofiler run --backend openmp -- ./omp_app        # LLVM libomp (OMPT) or GNU libgomp, detected automatically
hprofiler run --backend cuda   -- ./cuda_app       # host API + device activity (CUPTI)
hprofiler run --backend opencl -- ./ocl_app
hprofiler run --backend rocm   -- ./hip_app        # not runnable on the development machine (no AMD GPU)
hprofiler run --backend cuda,nccl -- ./multi_gpu_app   # not runnable on the development machine (no NCCL)
OMP_NUM_THREADS=4 hprofiler run --backend mpi,openmp -o hybrid.json -- mpirun -np 2 ./hybrid
```

Names: `cpu`, `cuda`, `opencl`, `rocm`, `openmp`, `nccl`, `mpi`, `likwid`
(aliases `perf`, `cl`, `hip`, `omp`, `hwc`). For complete CUDA detail build
your program with the shared runtime (`nvcc -cudart shared`); statically
linked runtimes are recorded through CUPTI with less detail.

### MPI jobs and launchers

Put the launcher after `--`, and name the output (the default name comes
from the first word of the command — `mpirun.hprofiler.json`):

```bash
hprofiler run --backend mpi -o app.json -- mpirun -np 4 ./mpi_app
hprofiler run --backend mpi -o gmx.json -- srun --export=ALL -n 4 gmx_mpi mdrun -s topol.tpr   # SLURM: not available on the development machine
```

Every rank must inherit `LD_PRELOAD`, `HPROFILER_SOCKET` and
`OMP_TOOL_LIBRARIES`. MPICH/Hydra and Open MPI pass the environment to
local ranks; otherwise use `mpirun -x LD_PRELOAD -x HPROFILER_SOCKET -x
OMP_TOOL_LIBRARIES …` (Open MPI) or `srun --export=ALL` (SLURM). The
collector only reaches ranks on its own node: for multi-node jobs, capture
each node separately and merge:

```bash
hprofiler merge-nodes node0.json node1.json -o merged.json
```

A run that captures nothing at all prints a warning naming the launcher as
the likely cause.

### Saving without the UI, reopening later

```bash
hprofiler run --backend openmp --no-ui -o run1.json -- ./app   # writes run1.hpstore + run1.json
hprofiler run --backend openmp --no-ui --no-json -o big.hpstore -- ./app   # store only
hprofiler view run1.hpstore        # terminal UI
hprofiler gui  run1.hpstore        # Qt GUI (falls back to the terminal UI)
hprofiler summary run1.json        # text report; a run's JSON opens its store
```

Every command accepts a `.hpstore` or any JSON hprofiler wrote; `--no-json`
saves time and disk on very large captures.

### Terminal UI and GUI

```bash
hprofiler run --gui --backend cuda -- ./cuda_app   # open the GUI instead of the TUI after the run
hprofiler gui run1.hpstore
```

The TUI works over any SSH session: tabs are numbered (`1`–`9` jump),
the Timeline zooms with `+`/`-` and scrolls with the arrow keys, and
hovering an MPI/NCCL span draws its communication partners. The GUI adds
Timeline filtering, grouping, search and bookmarks, sortable/exportable
tables, a Compare tab, a command palette (Ctrl+K) and a shortcut list (F1).
It runs in its own process and prints its rendering tier and load progress
to the terminal; Ctrl+C cancels a load.

### Comparing two runs

The first trace is the **baseline**, the second the **candidate**:

```bash
hprofiler compare baseline.hpstore candidate.hpstore            # BEFORE = baseline, AFTER = candidate
hprofiler compare baseline.json candidate.json --format json -o diff.json
hprofiler gui candidate.hpstore --compare baseline.hpstore      # GUI: the opened trace is the candidate
```

Runs are aligned by structure (repeated phases, call paths, roles), each
regression gets a cause (more work, more calls, queueing, synchronization,
communication, lost overlap, a changed dependency, moving onto the critical
path), and contributors are ranked by the critical-path time they account
for. Changes must exceed a disclosed noise floor (5 % and 1 ms by default;
`--min-pct`, `--min-ns`) — single runs carry no variance, so this is not a
significance test.

### Analyses

```bash
hprofiler summary --top 10 run1.hpstore                    # hotspots, GPU activity, capture warnings
hprofiler critical-path run1.hpstore                        # cross-runtime critical path and blame
hprofiler critical-path run1.hpstore --export path.json     # spans tagged on_critical_path=1 for Perfetto
hprofiler efficiency run1.hpstore                           # POP metrics: load balance, communication, …
```

**Call trees and flame graphs** (Call Tree and Flame Graph tabs):

```bash
hprofiler run --backend openmp --call-tree -- ./app        # stacks at every intercepted call; build with -fno-omit-frame-pointer -rdynamic
hprofiler run --perf-callgraph dwarf -- ./app              # perf-sampled CPU stacks; needs perf access (not available on the development machine)
```

**Disassembly** (Source tab), including call sites of OpenMP and MPI
events:

```bash
hprofiler run --backend openmp --disasm -o dis.json -- ./app
hprofiler disasm dis.hpstore --list
hprofiler disasm dis.hpstore -k omp_barrier
```

**Roofline:**

```bash
hprofiler roofline --backend cuda -- ./cuda_app     # hardware counters (ncu / rocprof / LIKWID / perf stat), re-runs the program
hprofiler roofline --html cuda_dis.hpstore          # estimates from a trace recorded with --disasm; writes cuda_dis.roofline.html
```

On a machine without counter access the first form stops with the command
that grants it.

### Warnings, logs, dropped events and timing provenance

hprofiler reports everything it lost or could only approximate — at the
end of `run`, at the top of `summary`, and in the Overview of both UIs:

```
[hprofiler][warn] gomp hook, pid 1524292: 55389 dropped (ring full after waiting)
[hprofiler][warn] CUDA pid 1555103: cupti unavailable (enable_kernel_activity_failed_14) -- device times are host-side proxies, not device measurements
```

- **Dropped events**: each hook counts records it dropped because a
  thread's buffer stayed full or could not be delivered; raise
  `HPROFILER_RING_KB` if you see drops.
- **Timing provenance**: `hprofiler summary` states the source of GPU
  activity (`timing source : device-measured (CUPTI / ROCprofiler-SDK)`, or
  a proxy), and the GUI inspector shows it per span.
- **Errors**: a damaged or truncated trace prints one line and exits with
  status 2; `HPROFILER_DEBUG=1` shows the traceback. The GUI logs to
  `~/.local/share/hprofiler/hprofiler/hprofiler-gui.log`.
- hprofiler does not report your program's exit status — check its own
  output if a run has fewer events than expected.

## 5. Understanding the output

| Output | What it is |
|---|---|
| `<name>.hpstore/` | Indexed SQLite trace store; what every command reads |
| `<name>.json` | Chrome Trace JSON for [Perfetto](https://ui.perfetto.dev) (timestamps in µs); lossless, reopens in hprofiler |
| summary on stdout | Events by category, hotspots, GPU timeline analysis, CPU counters, capture warnings |
| `<name>.roofline.html` | Self-contained roofline chart |

**Host vs. device.** For CUDA and ROCm, every launch, copy or memset is two
spans: the API call on its CPU thread (`side=cpu`) and the work on the GPU
(`side=gpu`), drawn on the stream's Timeline lane. Device time is never
counted as CPU-thread time, and GPU-active time is computed from one timing
source only.

**Measured, proxy, derived.**

| Kind | Example |
|---|---|
| measured | host call durations; device start/end from CUPTI or ROCprofiler-SDK |
| proxy | GPU-event duration placed at the submission time (`timing=proxy_event`) when no native tracer is available |
| derived | queueing delay = device start − host call end |
| heuristic | phase alignment in `compare`; disassembly-based roofline estimates |

**How time is attributed.** Breakdowns and the one-line diagnosis use
*exclusive* time: each instant on a thread belongs to the innermost
instrumented call, so a barrier inside a parallel region counts once, as
synchronization. The Kernels table shows *inclusive* totals. The critical
path labels each hop `certain`, `high` or `medium` by how directly it is
proven.

## 6. Backends and limitations

| Backend | Attached via | Records | Verified on |
|---|---|---|---|
| `cpu` | `perf record` | CPU samples, optional call graphs | parser only (perf blocked on the development machine) |
| `cuda` | `LD_PRELOAD` + CUPTI | API calls, kernels/copies/memsets, NVTX, memory | a GeForce MX550 (functional checks) |
| `rocm` | `LD_PRELOAD` + ROCprofiler-SDK | API calls, dispatches, copies | decoder tests only — never run on an AMD GPU |
| `opencl` | `LD_PRELOAD` | enqueues, device execution, transfers, builds | Intel CPU OpenCL device, against ground truth |
| `openmp` | OMPT + `GOMP_*` interposition | regions, per-thread work, loops, tasks, barriers, critical sections | ground truth for both runtimes; GROMACS on an HPC cluster |
| `nccl` | `LD_PRELOAD` | collectives, send/recv, groups | not run |
| `mpi` | `LD_PRELOAD` (PMPI) | point-to-point, requests, collectives, RMA sync, communicators | one process (multi-rank not available on the development machine) |
| `likwid` | `likwid-perfctr` wrapper | PMU counter groups | not run |

Limitations to know before trusting a number (all of them, with
workarounds, are in [Known limitations](DOCUMENTATION.md#known-limitations)):

- The collector is node-local; multi-node traces need per-node captures
  and `merge-nodes`, whose clock alignment is unverified on real clusters.
- Without CUPTI / ROCprofiler-SDK, GPU device spans are proxies that start
  at the launch time.
- NCCL wrappers wait for each operation to finish (serializing it with the
  host), and NCCL timing has never run on real hardware.
- NVTX v3 (header-only) ranges are not visible, and `schedule(static)`
  loops under GNU libgomp get no loop span.
- `--gpu-pc-sampling` disabled native device timing on the one GPU tested.
- Roofline from a saved trace works only with `--html` at the moment;
  `view`/`gui --disasm` do not add disassembly to a `.hpstore`.

## 7. Testing and documentation

```bash
QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests -p 'test_*.py'   # unit tests
python3 -m unittest tests.integration.test_profiling_accuracy \
    tests.integration.test_gomp_hook tests.integration.test_mpi_protocol \
    tests.integration.test_mpi_rma tests.integration.test_callsite_e2e \
    tests.integration.test_gui_cancel tests.integration.test_native_gpu_records \
    tests.integration.test_cuda_native_activity          # real programs through the hooks; skip without toolchains
python3 -m unittest tests.integration.test_store_stress  # 2M-span trace store (~1 min)
bash tests/integration/run_matrix.sh                      # CLI end to end, per backend
python3 tests/integration/accuracy_report.py --trials 10  # repeated-trial accuracy statistics
```

| Where | What |
|---|---|
| [DOCUMENTATION.md](DOCUMENTATION.md) | Reference: capture options and environment variables, backend timing semantics, every analysis, trace format and architecture, overhead, [verification status](DOCUMENTATION.md#verification-status), limitations, experimental validation |
| `hprofiler <command> --help` | Exact command-line syntax |
| [tests/README.md](tests/README.md) | Map of the test suite |
