/*
 * hp_ids.h -- span ids (sid=/psid=) that never collide across hooks.
 *
 * Span ids are joined by value within a process: an MPI completion's psid=
 * names its request, a device span its host call, an OpenMP span its
 * parallel region, a host call the NVTX range open around it. Every hook
 * numbers its own spans, and they used to share one layout -- (pid << 32) | n
 * with n counted from 1 by each hook (NVTX and OpenCL: a nanosecond
 * timestamp) -- so the MPI hook's first request and the CUDA hook's first
 * host call had the same id (816 shared ids in one MPI + CUDA trace; the call
 * tree merged them into cycles). Now each hook has its own space: its tag in
 * bits 56..62, the pid in bits 32..55 (Linux pids are below 2^22), its own
 * counter below. src/analysis/span_links.py still resolves the shared ids of
 * traces recorded before.
 */
#pragma once

#include <stdint.h>

enum {
    HP_ID_CUDA   = 1,     /* host calls (cuda_hook.c); CUPTI callbacks: bit 31 of n set */
    HP_ID_ROCM   = 2,     /* host calls (rocm_hook.c) */
    HP_ID_MPI    = 3,     /* requests and their completions (mpi_hook.c) */
    HP_ID_OMPT   = 4,     /* parallel regions (ompt_tool.c) */
    HP_ID_NVTX   = 5,     /* NVTX ranges (cuda_hook.c) */
    HP_ID_OPENCL = 6,     /* host-side kernel / transfer spans (opencl_hook.c) */
    HP_ID_NCCL   = 7,     /* host calls with HPROFILER_NCCL_ASYNC=1 (nccl_hook.c) */
};

static inline uint64_t hp_span_id(unsigned tag, uint32_t pid, uint32_t n) {
    return ((uint64_t)(tag & 0x7fu) << 56) | ((uint64_t)(pid & 0xffffffu) << 32) | n;
}
