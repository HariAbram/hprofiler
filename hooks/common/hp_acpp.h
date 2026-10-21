/*
 * hp_acpp.h -- AdaptiveCpp SSCP runtime (stage-2) JIT compilation as jit spans.
 *
 * AdaptiveCpp compiles a kernel at its first launch: LLVM IR from the binary
 * -> SPIR-V, PTX, AMDGCN or a host .jit.so (stage 2, in the language
 * runtime, before any OpenCL/CUDA/HIP call). Every backend reaches it through
 * kernel_cache::get_or_construct_jit_code_object (include/hipSYCL/runtime/
 * kernel_cache.hpp), a template compiled into the backend library:
 *
 *   persistent_cache_lookup(id, out)  -> true: the on-disk cache had it (warm)
 *   jit_compile(out)                  (stage 2)
 *   persistent_cache_store(id, out)
 *
 * The two cache calls are exported members of libacpp-rt.so, called through
 * the PLT from the backend libraries (loaded with dlopen(RTLD_NOW), no
 * RTLD_DEEPBIND -- src/common/dylib_loader.cpp), so a preloaded definition
 * interposes them. A miss followed by the store on the same thread brackets
 * the compilation:
 *
 *   span jit "acpp_jit_compile"  type=jit_compile,stage=runtime,cache=miss,id=,bytes=
 *   instant jit "acpp_jit_cache" type=jit_cache,cache=hit,id=       (warm: no compile)
 *
 * The real functions are looked up inside libacpp-rt.so itself (not
 * RTLD_NEXT), so when several hooks that include this header are preloaded,
 * the first one's wrapper calls the library directly and nothing is counted
 * twice. Included by the OpenCL, CUDA and ROCm hooks after hp_transport.h.
 */
#pragma once

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <link.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* kernel_cache::code_object_id = std::array<uint64_t, 2>, passed by value
 * (two integer registers on x86-64 and aarch64, like this struct). */
typedef struct { uint64_t v[2]; } hp_acpp_id;

#define HP_ACPP_LOOKUP "_ZNK7hipsycl2rt12kernel_cache23persistent_cache_lookupESt5arrayImLm2EERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE"
#define HP_ACPP_STORE  "_ZNK7hipsycl2rt12kernel_cache22persistent_cache_storeESt5arrayImLm2EERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE"

static __thread uint64_t   hp_acpp_t0;       /* lookup missed at; 0: none pending */
static __thread hp_acpp_id hp_acpp_pending;

static uint64_t hp_acpp_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int hp_acpp_find_rt(struct dl_phdr_info *info, size_t size, void *data) {
    (void)size;
    if (info->dlpi_name && strstr(info->dlpi_name, "libacpp-rt.so")) {
        *(const char **)data = info->dlpi_name;
        return 1;
    }
    return 0;
}

/* `name` as defined in libacpp-rt.so (NULL when it is not loaded). */
static void *hp_acpp_rt_sym(const char *name) {
    const char *path = NULL;
    dl_iterate_phdr(hp_acpp_find_rt, &path);
    if (!path) return NULL;
    void *h = dlopen(path, RTLD_LAZY | RTLD_NOLOAD);
    if (!h) return NULL;
    void *f = dlsym(h, name);
    dlclose(h);                                 /* NOLOAD took a reference */
    return f;
}

/* libstdc++ (C++11 ABI) std::string: { char *p; size_t len; ... } */
static size_t hp_acpp_strlen(const void *s) {
    size_t n = 0;
    if (s) memcpy(&n, (const char *)s + sizeof(char *), sizeof(n));
    return n;
}

typedef bool (*hp_acpp_lookup_t)(const void *, hp_acpp_id, void *);
typedef void (*hp_acpp_store_t)(const void *, hp_acpp_id, const void *);

bool _ZNK7hipsycl2rt12kernel_cache23persistent_cache_lookupESt5arrayImLm2EERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(
        const void *self, hp_acpp_id id, void *out) {
    static hp_acpp_lookup_t real = NULL;
    if (!real) real = (hp_acpp_lookup_t)hp_acpp_rt_sym(HP_ACPP_LOOKUP);
    if (!real) return false;               /* not reachable: the caller lives in libacpp-rt's users */
    uint64_t t0 = hp_acpp_now();
    bool hit = real(self, id, out);
    if (hit) {
        hp_acpp_t0 = 0;
        if (hp_tx_enabled())
            hp_tx_emitf("inst:jit:%d:%d:%llu:acpp_jit_cache:type=jit_cache,cache=hit,"
                        "id=%016llx%016llx,bytes=%zu,lookup_ns=%llu\n",
                        (int)hp_tx_pid(), (int)hp_tx_tid(), (unsigned long long)t0,
                        (unsigned long long)id.v[0], (unsigned long long)id.v[1],
                        hp_acpp_strlen(out), (unsigned long long)(hp_acpp_now() - t0));
    } else {
        hp_acpp_t0 = hp_acpp_now();        /* stage 2 starts now */
        hp_acpp_pending = id;
    }
    return hit;
}

void _ZNK7hipsycl2rt12kernel_cache22persistent_cache_storeESt5arrayImLm2EERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(
        const void *self, hp_acpp_id id, const void *data) {
    static hp_acpp_store_t real = NULL;
    if (!real) real = (hp_acpp_store_t)hp_acpp_rt_sym(HP_ACPP_STORE);
    uint64_t t1 = hp_acpp_now();
    if (hp_acpp_t0 && memcmp(&id, &hp_acpp_pending, sizeof(id)) == 0 && hp_tx_enabled())
        hp_tx_emitf("span:jit:%d:%d:%llu:%llu:acpp_jit_compile:type=jit_compile,stage=runtime,"
                    "cache=miss,id=%016llx%016llx,bytes=%zu\n",
                    (int)hp_tx_pid(), (int)hp_tx_tid(), (unsigned long long)hp_acpp_t0,
                    (unsigned long long)(t1 - hp_acpp_t0),
                    (unsigned long long)id.v[0], (unsigned long long)id.v[1], hp_acpp_strlen(data));
    hp_acpp_t0 = 0;
    if (real) real(self, id, data);
}
