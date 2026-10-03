/*
 * Resolve a call-site return address to a symbol (dladdr) or library+offset
 * (/proc/self/maps), for the sym=/symfile= or lib=/offset= tags that let the
 * Source tab disassemble the user code that made the call -- span names such
 * as "omp_parallel_region" or "MPI_Bcast" are not ELF symbols. Shared by the
 * OMPT, GNU libgomp and MPI hooks.
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
    char      path[1024];
} HprofilerVmaEntry;

static HprofilerVmaEntry hprofiler_vma_cache[HPROFILER_VMA_CACHE_CAP];
static int               hprofiler_vma_n     = 0;
static int               hprofiler_vma_ready = 0;
static pthread_mutex_t   hprofiler_vma_mutex = PTHREAD_MUTEX_INITIALIZER;

static void hprofiler_vma_build(void) {
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return;
    char line[2048];
    hprofiler_vma_n = 0;
    while (fgets(line, sizeof(line), f) && hprofiler_vma_n < HPROFILER_VMA_CACHE_CAP) {
        uintptr_t lo, hi, off;
        char perms[8], dev[16], path[1024];
        int inode;
        path[0] = '\0';
        if (sscanf(line, "%lx-%lx %7s %lx %15s %d %1023s",
                   &lo, &hi, perms, &off, dev, &inode, path) < 6)
            continue;
        if (path[0] == '\0' || path[0] == '[') continue;
        HprofilerVmaEntry *e = &hprofiler_vma_cache[hprofiler_vma_n++];
        e->lo = lo; e->hi = hi; e->file_off = (uint64_t)off;
        memcpy(e->path, path, strlen(path) + 1);   /* both buffers 1024; sscanf bounded to 1023 */
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

/* Resolve a return address: dladdr() for exported symbols, else the VMA
 * cache (library path + static file offset, also for internal symbols).
 * Fills out_sym + out_symfile, or out_lib + out_off; returns 1 if anything
 * was resolved. out_symfile (dladdr's dli_fname) is the ELF holding the
 * symbol: the profiled command is often a launcher (srun, mpirun), whose
 * argv[0] is the wrong file to disassemble. */
static int hprofiler_resolve_codeptr(const void *codeptr,
                                     const char **out_sym,
                                     char *out_symfile, size_t symfile_sz,
                                     char *out_lib, size_t lib_sz,
                                     uint64_t *out_off) {
    if (!codeptr) return 0;
    *out_sym = NULL;
    if (symfile_sz) out_symfile[0] = '\0';
    out_lib[0] = '\0';
    *out_off   = 0;

    Dl_info info;
    if (dladdr(codeptr, &info) && info.dli_sname && info.dli_sname[0]) {
        *out_sym = info.dli_sname;
        /* a path that does not fit is left out (disassembly then falls back
         * to the profiled command) rather than truncated into a wrong file */
        if (info.dli_fname && info.dli_fname[0] && symfile_sz &&
            strlen(info.dli_fname) < symfile_sz)
            memcpy(out_symfile, info.dli_fname, strlen(info.dli_fname) + 1);
        return 1;
    }

    if (!hprofiler_vma_lookup((uintptr_t)codeptr, out_lib, lib_sz, out_off)) return 0;
    if (strlen(out_lib) >= lib_sz - 1) { out_lib[0] = '\0'; return 0; }   /* path did not fit */
    return 1;
}

/* Append the call-site tags for `codeptr` to the null-terminated tag string
 * in `buf`: ",sym=<name>[,symfile=<elf>]" or ",lib=<path>,offset=0x<hex>".
 * All or nothing: when the tags do not fit, ",codeptr=truncated" is
 * appended instead (if that fits) so a cut-off name or path never reaches
 * the trace. Used by every hook that tags spans with their call site. */
static void hprofiler_append_codeptr_tag(char *buf, size_t bufsz, const void *codeptr) {
    const char *sym = NULL;
    char symfile[1024];
    char lib[1024];
    uint64_t off = 0;
    if (!hprofiler_resolve_codeptr(codeptr, &sym, symfile, sizeof(symfile), lib, sizeof(lib), &off))
        return;
    size_t used = strlen(buf);
    if (used >= bufsz) return;
    int n;
    if (sym && symfile[0])
        n = snprintf(buf + used, bufsz - used, ",sym=%s,symfile=%s", sym, symfile);
    else if (sym)
        n = snprintf(buf + used, bufsz - used, ",sym=%s", sym);
    else if (lib[0])
        n = snprintf(buf + used, bufsz - used, ",lib=%s,offset=0x%llx", lib, (unsigned long long)off);
    else
        return;
    if (n < 0 || (size_t)n >= bufsz - used) {
        buf[used] = '\0';
        if (bufsz - used > 18) memcpy(buf + used, ",codeptr=truncated", 19);
    }
}
