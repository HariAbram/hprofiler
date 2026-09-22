/*
 * Resolve a call-site return address to a symbol name or library+offset,
 * for tagging spans with sym=/lib=,offset= so src/core/runner.py's
 * _collect_disasm() can later disassemble the actual user code that made
 * the call (not the hprofiler-internal span label itself, which has no
 * corresponding ELF symbol -- "omp_parallel_region" or "MPI_Bcast" are
 * event names this project invents, not functions objdump can find).
 *
 * Extracted from hooks/ompt_tool/ompt_tool.c's original implementation
 * (the only hook that had it) so hooks/gomp_hook/gomp_hook.c and
 * hooks/mpi_hook/mpi_hook.c can emit the same sym=/lib=,offset= tags
 * ompt_tool.c already does -- until this fix, only OMPT-visible OpenMP
 * spans got real disassembly; GNU libgomp spans (gomp_hook.c) and every
 * MPI span had the tag but nothing populating it, so the Source tab's
 * "No disassembly available" was unconditional for them, not a fallback.
 */
#pragma once

#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define HPROFILER_VMA_CACHE_CAP 1024

typedef struct {
    uintptr_t lo, hi;
    uint64_t  file_off;
    char      path[256];
} HprofilerVmaEntry;

static HprofilerVmaEntry hprofiler_vma_cache[HPROFILER_VMA_CACHE_CAP];
static int               hprofiler_vma_n     = 0;
static int               hprofiler_vma_ready = 0;
static pthread_mutex_t   hprofiler_vma_mutex = PTHREAD_MUTEX_INITIALIZER;

static void hprofiler_vma_build(void) {
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return;
    char line[512];
    hprofiler_vma_n = 0;
    while (fgets(line, sizeof(line), f) && hprofiler_vma_n < HPROFILER_VMA_CACHE_CAP) {
        uintptr_t lo, hi, off;
        char perms[8], dev[16], path[256];
        int inode;
        path[0] = '\0';
        if (sscanf(line, "%lx-%lx %7s %lx %15s %d %255s",
                   &lo, &hi, perms, &off, dev, &inode, path) < 6)
            continue;
        if (path[0] == '\0' || path[0] == '[') continue;
        HprofilerVmaEntry *e = &hprofiler_vma_cache[hprofiler_vma_n++];
        e->lo = lo; e->hi = hi; e->file_off = (uint64_t)off;
        strncpy(e->path, path, sizeof(e->path) - 1);
        e->path[sizeof(e->path) - 1] = '\0';
    }
    fclose(f);
    hprofiler_vma_ready = 1;
}

static int hprofiler_vma_lookup(uintptr_t addr, char *out_lib, size_t lib_sz, uint64_t *out_off) {
    pthread_mutex_lock(&hprofiler_vma_mutex);
    if (!hprofiler_vma_ready) hprofiler_vma_build();
    for (int i = 0; i < hprofiler_vma_n; i++) {
        if (addr >= hprofiler_vma_cache[i].lo && addr < hprofiler_vma_cache[i].hi) {
            *out_off = hprofiler_vma_cache[i].file_off + (addr - hprofiler_vma_cache[i].lo);
            strncpy(out_lib, hprofiler_vma_cache[i].path, lib_sz - 1);
            out_lib[lib_sz - 1] = '\0';
            pthread_mutex_unlock(&hprofiler_vma_mutex);
            return 1;
        }
    }
    /* Miss: rebuild once in case new libraries were loaded since last build. */
    hprofiler_vma_build();
    for (int i = 0; i < hprofiler_vma_n; i++) {
        if (addr >= hprofiler_vma_cache[i].lo && addr < hprofiler_vma_cache[i].hi) {
            *out_off = hprofiler_vma_cache[i].file_off + (addr - hprofiler_vma_cache[i].lo);
            strncpy(out_lib, hprofiler_vma_cache[i].path, lib_sz - 1);
            out_lib[lib_sz - 1] = '\0';
            pthread_mutex_unlock(&hprofiler_vma_mutex);
            return 1;
        }
    }
    pthread_mutex_unlock(&hprofiler_vma_mutex);
    return 0;
}

/* Resolve a return address:
 *  1. Try dladdr() -- works when the symbol is exported.
 *  2. Fall back to the VMA cache -- finds the library and computes the
 *     static file offset even for non-exported / internal symbols.
 *
 * On success, writes into `out_sym` (symbol name) or `out_lib`+`out_off`
 * (library path + offset). At most one of sym / lib+off is filled.
 * Returns 1 if anything was resolved, 0 otherwise.
 */
static int hprofiler_resolve_codeptr(const void *codeptr,
                                     const char **out_sym,
                                     char *out_lib, size_t lib_sz,
                                     uint64_t *out_off) {
    if (!codeptr) return 0;
    *out_sym = NULL;
    out_lib[0] = '\0';
    *out_off   = 0;

    Dl_info info;
    if (dladdr(codeptr, &info) && info.dli_sname && info.dli_sname[0]) {
        *out_sym = info.dli_sname;
        return 1;
    }

    return hprofiler_vma_lookup((uintptr_t)codeptr, out_lib, lib_sz, out_off);
}
