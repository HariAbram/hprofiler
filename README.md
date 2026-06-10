# hprofiler — Heterogeneous Profiler

Multi-device CPU/GPU profiler for Linux. Traces programs across CUDA, ROCm, OpenCL, OpenMP, NCCL, and MPI simultaneously, with a terminal UI and native TUI viewers for flame graphs and roofline charts. CPU sampling is provided via Linux `perf`.

## Requirements

- Python 3.10+, CMake 3.16+, GCC/Clang
- `pip install click textual rich capstone`
- TUI flamegraph/roofline viewers: `pip install plotly "kaleido==0.2.1"` (0.2.1 specifically — later versions require Chrome and break on clusters)
- Backend-specific: CUDA toolkit, ROCm at `/opt/rocm`, LLVM `libomp`, `mpicc`, or `perf`

## Build

```bash
pip install click textual rich capstone
python3 hprofiler build
```

Produces `build/lib/libhprofiler_{cuda,opencl,ompt,rocm,nccl,mpi}.so`.

## Quick Start

```bash
# Profile with auto-detected backends
python3 hprofiler run -- ./my_program

# Specific backends
python3 hprofiler run --backend cuda,cpu      -- ./cuda_app
python3 hprofiler run --backend openmp        -- ./omp_app
python3 hprofiler run --backend rocm          -- ./hip_app
python3 hprofiler run --backend cuda,nccl     -- ./multi_gpu_app
python3 hprofiler run --backend mpi           -- mpirun -np 4 ./mpi_app

# Call tree (adds Call Tree tab; compile with -fno-omit-frame-pointer -rdynamic)
python3 hprofiler run --call-tree --backend cuda -- ./app

# Per-kernel disassembly (adds Disasm tab)
python3 hprofiler run --backend cuda --disasm -- ./app

# Save trace, skip TUI
python3 hprofiler run --no-ui -o trace.json -- ./app

# Open a saved trace
python3 hprofiler view trace.json

# Text summary only
python3 hprofiler summary trace.json

# Flame graph — opens TUI viewer by default (requires plotly + kaleido)
python3 hprofiler flamegraph -- ./my_program
python3 hprofiler flamegraph --backend cuda -- ./cuda_app
python3 hprofiler flamegraph --callgraph dwarf -- ./my_program  # no frame-pointer binary
python3 hprofiler flamegraph --html -- ./my_program             # write HTML + open browser

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
| `opencl` | `cl` | LD_PRELOAD | Kernel enqueues, buffer transfers, JIT compile time |
| `openmp` | `omp` | `OMP_TOOL_LIBRARIES` (OMPT) | Parallel regions, tasks, loops, barriers (requires LLVM `libomp`, not GCC `libgomp`) |
| `rocm` | `hip` | LD_PRELOAD | HIP kernel launches, memcpy, memory counters |
| `nccl` | — | LD_PRELOAD | Collectives (AllReduce, Broadcast, …), point-to-point — GPU-accurate timing |
| `mpi` | — | PMPI / LD_PRELOAD | Send/Recv, collectives, one-sided ops — wall-clock timing |
| `likwid` | `hwc` | `likwid-perfctr` wrapper | Hardware PMU counters: FLOPS, DRAM bandwidth, cache rates, CPI |

## TUI Viewer

Opens automatically after `hprofiler run`. Tabs:

| Tab | When shown | Contents |
|-----|-----------|---------|
| System | Always | Device specs, FP16/32/64/Tensor TFLOP/s, bandwidth, IPC, LLC/branch miss rates, RSS |
| Profile | Always | GPU activity%, time breakdown by category, top hotspots, bottleneck advisor |
| Timeline | Always | Gantt view with per-stream CUDA/ROCm lanes |
| Hotspots | Always | Filterable/sortable function table |
| Call Tree | Only with `--call-tree` | Stack-frame tree from captured call graphs |
| Flame | Only with CPU/perf data | ASCII bar chart of top CPU-sampled functions |
| Disasm | Only with `--disasm` | Per-kernel assembly with instruction-type breakdown |

## Flamegraph TUI Controls

Requires an inline-image terminal: kitty, WezTerm, Ghostty (Kitty protocol), iTerm2, or xterm/mlterm (Sixel). Falls back to browser if no protocol is detected.

| Key | Action |
|-----|--------|
| click | Zoom into that frame |
| `u` / Esc | Zoom out one level |
| `r` | Reset to full view |
| `/` | Search — highlight frames by name substring |
| `w` | Open HTML version in browser |
| `q` | Quit |

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
| `<prog>.flamegraph.html` | Any browser — click to zoom, search |
| `<prog>.roofline.html` | Any browser (self-contained) |

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
| OpenCL JIT | `objdump` on the `.jit.so` emitted by ACPP SSCP |

See [DOCUMENTATION.md](DOCUMENTATION.md) for the full CLI reference, backend details, wire protocol, and how to extend the profiler.

## AI Performance Analysis

hprofiler can use an LLM to analyse a profile and produce a written report of bottlenecks, root causes, and prioritised optimisation recommendations.

### Quick start

```bash
# Analyse an existing trace (auto-detects LLM from env vars)
python3 hprofiler analyze trace.hprofiler.json

# Profile and analyse in one step
python3 hprofiler analyze --backend cuda -- ./app

# Add AI analysis to the normal run workflow
python3 hprofiler run --analyze --backend cuda -- ./app

# Compare two runs and report what changed
python3 hprofiler analyze --compare before.json after.json trace.hprofiler.json

# Save report to a Markdown file
python3 hprofiler analyze --output-report report.md trace.hprofiler.json
```

### LLM provider setup

The provider is auto-detected from environment variables. Set one of the following before running:

```bash
# Anthropic Claude (recommended)
export ANTHROPIC_API_KEY=sk-ant-...
python3 hprofiler analyze trace.hprofiler.json
# Defaults to claude-sonnet-4-6; override with --llm-model claude-opus-4-8

# OpenAI GPT
export OPENAI_API_KEY=sk-...
python3 hprofiler analyze trace.hprofiler.json
# Defaults to gpt-4o

# Ollama (local, no API key needed)
ollama serve                        # start the Ollama daemon
ollama pull llama3.1:8b             # pull a model
python3 hprofiler analyze trace.hprofiler.json
# Defaults to llama3.1:8b; any pulled model works

# Any OpenAI-compatible endpoint (vLLM, LM Studio, Groq, Together.ai, …)
python3 hprofiler analyze \
  --llm openai-compat \
  --llm-endpoint http://localhost:8080 \
  --llm-model Qwen2.5-72B-Instruct \
  trace.hprofiler.json
```

### Persistent configuration via environment variables

```bash
export HPROFILER_LLM_PROVIDER=anthropic   # anthropic | openai | ollama | openai-compat
export HPROFILER_LLM_MODEL=claude-opus-4-8
export HPROFILER_LLM_API_KEY=sk-ant-...   # if not using ANTHROPIC_API_KEY / OPENAI_API_KEY
export HPROFILER_LLM_ENDPOINT=http://...  # for openai-compat / custom Ollama host
```

### How it works

The agent calls a suite of tools to drill into the profile — hotspots, kernel details, memory patterns, timeline phases, synchronisation overhead, MPI communication — before writing its final report. For models without tool-use support it falls back to a single comprehensive prompt. No new Python packages are required; all HTTP calls use `urllib.request`.
