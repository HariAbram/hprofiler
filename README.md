# hprofiler

Multi-device CPU/GPU profiler for Linux. Traces programs via CUDA, ROCm, OpenCL, OpenMP, and perf — simultaneously — with a terminal UI.

## Requirements

- Python 3.10+, CMake 3.16+, GCC/Clang
- `pip install click textual rich`
- Backend-specific: CUDA toolkit, ROCm at `/opt/rocm`, LLVM libomp, or `perf`

## Build

```bash
pip install click textual rich
python3 hprofiler build
```

Produces `build/lib/libhprofiler_{cuda,opencl,ompt,rocm}.so`.

## Usage

```bash
# Profile (auto-detect backends)
python3 hprofiler run -- ./my_program

# Specific backends
python3 hprofiler run --backend cuda,cpu -- ./app
python3 hprofiler run --backend openmp  -- ./omp_app
python3 hprofiler run --backend rocm    -- ./hip_app

# With per-kernel disassembly
python3 hprofiler run --backend cuda --disasm -- ./app

# Save trace, skip TUI
python3 hprofiler run --no-ui -o trace.json -- ./app

# Open a saved trace
python3 hprofiler view trace.json

# Text summary only
python3 hprofiler summary trace.json

# Roofline chart (hardware counters)
python3 hprofiler roofline --backend cuda -- ./app
python3 hprofiler roofline --backend cpu  -- ./app

# Flame graph
python3 hprofiler flamegraph -- ./app

# List available backends on this machine
python3 hprofiler backends
```

Always separate hprofiler options from the target program with `--`.

## Backends

| Name | Alias | How it works |
|------|-------|--------------|
| `cpu` | `perf` | `perf record` sampling |
| `cuda` | — | LD_PRELOAD hook (CUDA Runtime + Driver API) |
| `opencl` | `cl` | LD_PRELOAD hook (command-queue profiling) |
| `openmp` | `omp` | OMPT tool via `OMP_TOOL_LIBRARIES` (requires LLVM libomp, not GCC libgomp) |
| `rocm` | `hip` | LD_PRELOAD hook + roctracer |

## Output

- **TUI** — opens automatically after a run; timeline, kernel stats, disassembly
- **`<prog>.hprofiler.json`** — Chrome Trace format (open in `chrome://tracing` or Perfetto)
- **`<prog>.roofline.html`** — interactive roofline chart
- **`<prog>.flamegraph.svg`** — interactive flame graph

## Disassembly

Pass `--disasm` to collect post-run disassembly. Requires:

- CUDA AoT: `cuobjdump` (CUDA toolkit)
- CUDA JIT: `nvdisasm`; cubins are captured automatically to `/tmp/hprofiler_cubin_*.bin`
- ROCm: `llvm-objdump`
- CPU/OpenCL: `capstone` (`pip install capstone`) or `objdump`
