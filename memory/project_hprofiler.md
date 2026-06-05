---
name: project-hprofiler
description: Overview of the hprofiler project — multi-backend CPU/GPU profiler CLI
metadata:
  type: project
---

hprofiler is a multi-backend CPU/GPU profiler with a terminal UI. It profiles programs using CUDA, ROCm, OpenCL, OpenMP, and perf (CPU) backends simultaneously.

**Why:** General-purpose performance profiling tool for HPC workloads.

**How to apply:** When suggesting commands or usage examples, use `hprofiler` as the CLI name and `libhprofiler_*.so` for the hook libraries.

## Key structure
- `hprofiler` — Python CLI entry point (click-based), was previously named `profiler`
- `src/backends/` — per-backend Python classes (cuda, opencl, openmp, rocm, perf/cpu)
- `src/core/runner.py` — process runner; injects LD_PRELOAD hooks, listens on `HPROFILER_SOCKET`
- `src/disasm/` — post-run disassembly extraction (objdump, cuobjdump, llvm-objdump)
- `src/output/` — chrome trace JSON, flame graph SVG, roofline HTML, text summary
- `src/ui/app.py` — Textual TUI viewer
- `src/analysis/` — roofline model, hardware counter collection, device queries
- `hooks/` — C LD_PRELOAD hooks built as `libhprofiler_{cuda,opencl,ompt,rocm}.so`

## Hook communication
Hooks write newline-delimited ASCII records over a Unix domain socket set via `HPROFILER_SOCKET`. Temp files use prefix `hprofiler_` (e.g. `/tmp/hprofiler_cubin_*.bin` for CUDA JIT cubins).

## Build
`hprofiler build` runs CMake to produce the hook `.so` files in `build/lib/`.
