/*
 * hp_module.h -- which module a GPU kernel launch runs code from.
 *
 * A module id is the 64-bit FNV-1a hash of the code blob a hook saved for
 * disassembly (exactly the bytes written to the /tmp file), printed as 16
 * hex digits. The hooks put mod=<id> on the module's jit span (next to its
 * path=) and on every launch of a kernel from that module, so the Python
 * side can key disassembly by (kernel name, module): two programs that both
 * define `benchmark_func` keep separate entries, and a launch shows the code
 * it actually ran. src/disasm/extractor.py's module_id() computes the same
 * hash from a saved file.
 *
 * The function table maps a kernel handle (CUfunction, hipFunction_t, or a
 * registered host stub) to its full name -- a heap copy, no length limit --
 * and the module it came from. Unloading a module drops its functions, so an
 * address the runtime reuses never picks up an earlier kernel's name.
 */
#pragma once

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define HP_FNV1A64_INIT 0xcbf29ce484222325ULL

/* Continue a hash over the next n bytes (start from HP_FNV1A64_INIT). */
static inline uint64_t hp_fnv1a64_update(uint64_t h, const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static inline uint64_t hp_fnv1a64(const void *data, size_t n) {
    return hp_fnv1a64_update(HP_FNV1A64_INIT, data, n);
}

/* File size of a 64-bit little-endian ELF image handed over in memory with
 * no length (cuModuleLoadData, hipModuleLoadData): the furthest end of the
 * section header table, the program header table, every section with file
 * data and every segment's file part, or 0 if that exceeds limit or the
 * headers look wrong. The section header table is not always last: CUDA
 * cubins from NVRTC/ptxas put the program headers after it. SHT_NOBITS
 * sections (.bss) occupy no file bytes. */
static inline size_t hp_elf64_image_size(const void *image, size_t limit) {
    const uint8_t *p = (const uint8_t *)image;
    if (!p || p[0] != 0x7f || p[1] != 'E' || p[2] != 'L' || p[3] != 'F' || p[4] != 2 || p[5] != 1)
        return 0;
    uint64_t phoff, shoff;
    uint16_t phesz, phnum, shesz, shnum;
    memcpy(&phoff, p + 32, 8);
    memcpy(&shoff, p + 40, 8);
    memcpy(&phesz, p + 54, 2);
    memcpy(&phnum, p + 56, 2);
    memcpy(&shesz, p + 58, 2);
    memcpy(&shnum, p + 60, 2);
    if ((shnum && (shesz < 64 || shoff < 64 || shoff > limit || (uint64_t)shesz * shnum > limit - shoff)) ||
        (phnum && (phesz < 56 || phoff < 64 || phoff > limit || (uint64_t)phesz * phnum > limit - phoff)))
        return 0;
    uint64_t end = 64;
    if (shnum && shoff + (uint64_t)shesz * shnum > end) end = shoff + (uint64_t)shesz * shnum;
    if (phnum && phoff + (uint64_t)phesz * phnum > end) end = phoff + (uint64_t)phesz * phnum;
    for (uint16_t i = 0; i < shnum; i++) {
        const uint8_t *sh = p + shoff + (uint64_t)i * shesz;
        uint32_t type;
        uint64_t off, sz;
        memcpy(&type, sh + 4, 4);
        memcpy(&off, sh + 24, 8);
        memcpy(&sz, sh + 32, 8);
        if (type == 8 /* SHT_NOBITS */ || !sz) continue;
        if (off > limit || sz > limit - off) return 0;
        if (off + sz > end) end = off + sz;
    }
    for (uint16_t i = 0; i < phnum; i++) {
        const uint8_t *ph = p + phoff + (uint64_t)i * phesz;
        uint64_t off, filesz;
        memcpy(&off, ph + 8, 8);
        memcpy(&filesz, ph + 32, 8);
        if (!filesz) continue;
        if (off > limit || filesz > limit - off) return 0;
        if (off + filesz > end) end = off + filesz;
    }
    return (size_t)end;
}

/* Size of a CUDA fat binary handed over in memory (magic 0xBA55ED50, then a
 * 16-bit version, the header size and the 64-bit payload size): header plus
 * payload -- the bytes nvcc put in .nv_fatbin -- or 0 if it is none or
 * exceeds limit. */
static inline size_t hp_fatbin_size(const void *image, size_t limit) {
    const uint8_t *p = (const uint8_t *)image;
    if (!p) return 0;
    uint32_t magic;
    uint16_t hdr;
    uint64_t payload;
    memcpy(&magic, p, 4);
    memcpy(&hdr, p + 6, 2);
    memcpy(&payload, p + 8, 8);
    if (magic != 0xBA55ED50u || hdr < 16 || payload > limit || hdr > limit - payload) return 0;
    return (size_t)hdr + (size_t)payload;
}

typedef struct { const void *handle; uint64_t mod; } HpModEntry;
typedef struct { const void *fn; const void *module; char *name; uint64_t mod; } HpFnEntry;

static HpModEntry     *g_hpm_mods;
static size_t          g_hpm_nmods, g_hpm_capmods;
static HpFnEntry      *g_hpm_fns;
static size_t          g_hpm_nfns, g_hpm_capfns;
static pthread_mutex_t g_hpm_mutex = PTHREAD_MUTEX_INITIALIZER;

static inline uint64_t hp_module_id_locked(const void *handle) {
    if (!handle) return 0;
    for (size_t i = 0; i < g_hpm_nmods; i++)
        if (g_hpm_mods[i].handle == handle) return g_hpm_mods[i].mod;
    return 0;
}

/* A module was loaded (mod 0: its code was not captured -- nothing to record). */
static inline void hp_module_add(const void *handle, uint64_t mod) {
    if (!handle || !mod) return;
    pthread_mutex_lock(&g_hpm_mutex);
    for (size_t i = 0; i < g_hpm_nmods; i++) {
        if (g_hpm_mods[i].handle == handle) {
            g_hpm_mods[i].mod = mod;
            pthread_mutex_unlock(&g_hpm_mutex);
            return;
        }
    }
    if (g_hpm_nmods == g_hpm_capmods) {
        size_t cap = g_hpm_capmods ? g_hpm_capmods * 2 : 64;
        HpModEntry *p = (HpModEntry *)realloc(g_hpm_mods, cap * sizeof(*p));
        if (!p) { pthread_mutex_unlock(&g_hpm_mutex); return; }
        g_hpm_mods = p;
        g_hpm_capmods = cap;
    }
    g_hpm_mods[g_hpm_nmods].handle = handle;
    g_hpm_mods[g_hpm_nmods].mod = mod;
    g_hpm_nmods++;
    pthread_mutex_unlock(&g_hpm_mutex);
}

/* A module was unloaded: forget it and every function looked up from it. */
static inline void hp_module_remove(const void *handle) {
    if (!handle) return;
    pthread_mutex_lock(&g_hpm_mutex);
    for (size_t i = 0; i < g_hpm_nmods; i++) {
        if (g_hpm_mods[i].handle == handle) {
            g_hpm_mods[i] = g_hpm_mods[--g_hpm_nmods];
            break;
        }
    }
    size_t keep = 0;
    for (size_t i = 0; i < g_hpm_nfns; i++) {
        if (g_hpm_fns[i].module == handle) free(g_hpm_fns[i].name);
        else g_hpm_fns[keep++] = g_hpm_fns[i];
    }
    g_hpm_nfns = keep;
    pthread_mutex_unlock(&g_hpm_mutex);
}

/* A kernel handle was obtained from `module` (NULL: no module known). */
static inline void hp_function_add(const void *fn, const void *module, const char *name) {
    if (!fn || !name || !name[0]) return;
    char *copy = strdup(name);
    if (!copy) return;
    pthread_mutex_lock(&g_hpm_mutex);
    uint64_t mod = hp_module_id_locked(module);
    for (size_t i = 0; i < g_hpm_nfns; i++) {
        if (g_hpm_fns[i].fn == fn) {
            free(g_hpm_fns[i].name);
            g_hpm_fns[i].name = copy;
            g_hpm_fns[i].module = module;
            g_hpm_fns[i].mod = mod;
            pthread_mutex_unlock(&g_hpm_mutex);
            return;
        }
    }
    if (g_hpm_nfns == g_hpm_capfns) {
        size_t cap = g_hpm_capfns ? g_hpm_capfns * 2 : 256;
        HpFnEntry *p = (HpFnEntry *)realloc(g_hpm_fns, cap * sizeof(*p));
        if (!p) { pthread_mutex_unlock(&g_hpm_mutex); free(copy); return; }
        g_hpm_fns = p;
        g_hpm_capfns = cap;
    }
    g_hpm_fns[g_hpm_nfns].fn = fn;
    g_hpm_fns[g_hpm_nfns].module = module;
    g_hpm_fns[g_hpm_nfns].name = copy;
    g_hpm_fns[g_hpm_nfns].mod = mod;
    g_hpm_nfns++;
    pthread_mutex_unlock(&g_hpm_mutex);
}

/* Name of a known kernel handle, copied into a per-thread buffer that stays
 * valid until this thread's next call (a concurrent unload cannot free it);
 * NULL when the handle is unknown. *mod gets the module id (0: none). */
static inline const char *hp_function_lookup(const void *fn, uint64_t *mod) {
    static __thread char  *tls_name;
    static __thread size_t tls_cap;
    if (mod) *mod = 0;
    if (!fn) return NULL;
    const char *found = NULL;
    pthread_mutex_lock(&g_hpm_mutex);
    for (size_t i = g_hpm_nfns; i-- > 0; ) {      /* most recent first */
        if (g_hpm_fns[i].fn != fn) continue;
        size_t n = strlen(g_hpm_fns[i].name) + 1;
        if (n > tls_cap) {
            char *p = (char *)realloc(tls_name, n);
            if (!p) break;
            tls_name = p;
            tls_cap = n;
        }
        memcpy(tls_name, g_hpm_fns[i].name, n);
        if (mod) *mod = g_hpm_fns[i].mod;
        found = tls_name;
        break;
    }
    pthread_mutex_unlock(&g_hpm_mutex);
    return found;
}

/* ",mod=<16 hex digits>" (or "" for mod 0) into buf[cap]. */
static inline const char *hp_mod_tag(char *buf, size_t cap, uint64_t mod) {
    if (!mod || cap < 22) { if (cap) buf[0] = '\0'; return buf; }
    static const char hex[] = "0123456789abcdef";
    memcpy(buf, ",mod=", 5);
    for (int i = 0; i < 16; i++) buf[5 + i] = hex[(mod >> (60 - 4 * i)) & 0xf];
    buf[21] = '\0';
    return buf;
}
