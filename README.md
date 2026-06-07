# hprofiler

Multi-device CPU/GPU profiler for Linux. Traces programs across CUDA, ROCm, OpenCL, OpenMP, NCCL, and MPI — simultaneously — with a terminal UI. CPU sampling is provided via Linux perf.

## Requirements

- Python 3.10+, CMake 3.16+, GCC/Clang
- `pip install click textual rich capstone`
- Backend-specific: CUDA toolkit, ROCm at `/opt/rocm`, LLVM `libomp`, `mpicc`, or `perf`

## Build

```bash
pip install click textual rich capstone
python3 hprofiler build
```

Produces `build/lib/libhprofiler_{cuda,opencl,ompt,rocm,nccl,mpi}.so`.

## Quick start

```bash
# Profile with auto-detected backends
python3 hprofiler run -- ./my_program

# Specific backends
python3 hprofiler run --backend cuda,cpu      -- ./cuda_app
python3 hprofiler run --backend openmp        -- ./omp_app
python3 hprofiler run --backend rocm          -- ./hip_app
python3 hprofiler run --backend cuda,nccl     -- ./multi_gpu_app
python3 hprofiler run --backend mpi           -- mpirun -np 4 ./mpi_app

# Call tree from main (adds Call Tree tab; compile app with -fno-omit-frame-pointer -rdynamic)
python3 hprofiler run --call-tree --backend cuda -- ./app

# With per-kernel disassembly (adds Disasm tab to TUI)
python3 hprofiler run --backend cuda --disasm -- ./app

# Save trace, skip TUI
python3 hprofiler run --no-ui -o trace.json -- ./app

# Open a saved trace
python3 hprofiler view trace.json

# Text summary
python3 hprofiler summary trace.json

# Flame graph (interactive HTML — click to zoom, regex search)
python3 hprofiler flamegraph -- ./my_program
python3 hprofiler flamegraph --backend cuda -- ./cuda_app        # GPU API overhead in stacks
python3 hprofiler flamegraph --callgraph dwarf -- ./my_program   # no frame-pointer binary

# Roofline chart (hardware counters)
python3 hprofiler roofline --backend cuda    -- ./cuda_app
python3 hprofiler roofline --backend cpu     -- ./cpu_app
python3 hprofiler roofline --backend rocm    -- ./hip_app

# Hardware PMU counters via LIKWID
HPROFILER_LIKWID_GROUP=MEM python3 hprofiler run --backend likwid -- ./app

# List available backends on this machine
python3 hprofiler backends
```

Always separate hprofiler options from the target program with `--`.

## Backends

| Name | Alias | Injection | What is traced |
|------|-------|-----------|----------------|
| `cpu` | `perf` | `perf record` subprocess | CPU samples, DWARF call-graph |
| `cuda` | — | LD_PRELOAD | Kernel launches, memcpy, syncs, NVTX ranges, memory counters |
| `opencl` | `cl` | LD_PRELOAD | Kernel enqueues, buffer transfers, JIT compile time |
| `openmp` | `omp` | `OMP_TOOL_LIBRARIES` (OMPT) | Parallel regions, tasks, loops, barriers (requires LLVM `libomp`, not GCC `libgomp`) |
| `rocm` | `hip` | LD_PRELOAD | HIP kernel launches, memcpy, memory counters |
| `nccl` | — | LD_PRELOAD | Collectives (AllReduce, Broadcast, …), point-to-point, group boundaries — GPU-accurate timing via CUDA events |
| `mpi` | — | PMPI link | Send/Recv, collectives, one-sided ops — wall-clock timing, bytes and peer rank tagged |
| `likwid` | `hwc` | `likwid-perfctr` wrapper | Hardware PMU counters: FLOPS, DRAM bandwidth, cache rates, CPI |

## Output

| File | Viewer |
|------|--------|
| TUI (opens automatically) | Overview · Timeline · Hotspots · [Flame] · [Call Tree] · [Disasm] |
| `<prog>.hprofiler.json` | [Perfetto](https://ui.perfetto.dev) or `chrome://tracing` |
| `<prog>.flamegraph.html` | Any browser — click to zoom, regex search, hover tooltips |
| `<prog>.roofline.html` | Any browser (self-contained) |

Tabs in brackets are conditional: Flame appears only with CPU data, Call Tree only with `--call-tree`, Disasm only with `--disasm`.

## Disassembly

Pass `--disasm` to collect post-run per-kernel disassembly (runs in background, TUI opens immediately):

| Backend | Tool needed |
|---------|-------------|
| CUDA AoT | `cuobjdump` (CUDA toolkit) |
| CUDA JIT (ACPP) | built-in PTX parser — cubins saved to `/tmp/hprofiler_cubin_*.bin` |
| ROCm | `llvm-objdump` (`apt install llvm`) |
| CPU / OpenMP | `capstone` (`pip install capstone`) or `objdump` |
| OpenCL JIT | `objdump` on `.jit.so` emitted by ACPP SSCP |

## Flame Graph

Interactive CPU flame graph — `perf record` captures call stacks and renders them as a self-contained HTML file:

```bash
python3 hprofiler flamegraph -- ./app
# output: app.flamegraph.html (open in any browser)

# Inject CUDA/OpenCL/MPI hooks so GPU API time shows in the CPU stacks
python3 hprofiler flamegraph --backend cuda -- ./cuda_app
ACPP_VISIBILITY_MASK=cuda python3 hprofiler flamegraph --backend cuda -- ./sycl_app
```

**Controls:** click to zoom, right-click to go up, Esc to reset, regex search box.

## Roofline

Hardware-counter roofline chart — re-runs the application under `ncu` (CUDA), `rocprof` (ROCm), or `perf stat` (CPU/OpenMP) and plots FLOPs vs bandwidth:

```bash
python3 hprofiler roofline --backend cuda -- ./app
# output: app.roofline.html
```

See [DOCUMENTATION.md](DOCUMENTATION.md) for full CLI reference, backend details, wire protocol, and how to extend the profiler.
