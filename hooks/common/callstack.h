/*
 * callstack.h — CPU call-stack capture for hprofiler hooks.
 *
 * Include this file AFTER defining the following in the including .c file:
 *   static int             g_sock;          // socket fd (-1 if not connected)
 *   static pid_t           g_pid;           // process id
 *   static pid_t gettid_compat(void);       // thread id
 *   static void send_all(const char*, int); // socket send helper
 *
 * The including file must also call cs_init() from its constructor to read
 * the HPROFILER_CALLSTACK environment variable.
 *
 * Wire format emitted when active:
 *   stk:<pid>:<tid>:<start_ns>:<frame0>;<frame1>;...\n
 * Frames are in innermost-first order (backtrace order).
 * The receiver reverses them to build a root→leaf call tree.
 *
 * Requires: -fno-omit-frame-pointer on the profiled binary for accurate stacks.
 */
#pragma once
#include <execinfo.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

/* Defined here (static) — one instance per compilation unit, which is exactly
 * one per hook .so since each hook is a single translation unit. */
static int g_callstack = 0;

static void cs_init(void)
{
    g_callstack = !!getenv("HPROFILER_CALLSTACK");
}

/* Libraries whose frames should be stripped from the captured stack. */
static const char *const _CS_SKIP[] = {
    "libhprofiler_",
    "libcuda", "libcudart",
    "libamdhip", "libhip", "librocm",
    "libOpenCL",
    "libmpi", "libmpich", "libopen-pal", "libopen-rte",
    "libnccl",
    "libgomp", "libiomp", "libomp", "libkmp",   /* GCC/Intel/LLVM OpenMP runtimes */
    "libpthread",
    "libc.so", "libc-",
    "ld-linux", "libdl.", "libstdc++",
    NULL
};

/*
 * emit_callstack — called inside emit_span while g_sock_mutex is held and
 * g_sock is known to be valid.  Captures the current backtrace, strips
 * hook/runtime frames, demangles C++ names, and sends a stk: record.
 */
static void emit_callstack(uint64_t start_ns)
{
    if (!g_callstack) return;

    void *frames[32];
    int nf = backtrace(frames, 32);
    if (nf <= 0) return;

    char **syms = backtrace_symbols(frames, nf);
    if (!syms) return;

    /* Locate __cxa_demangle once via dlsym — available if the target links C++. */
    typedef char *(*dem_fn_t)(const char *, char *, size_t *, int *);
    static dem_fn_t s_dem = (dem_fn_t)(uintptr_t)1; /* sentinel = not yet looked up */
    if (s_dem == (dem_fn_t)(uintptr_t)1)
        s_dem = (dem_fn_t)dlsym(RTLD_DEFAULT, "__cxa_demangle");

    char buf[8192];
    int pos = snprintf(buf, sizeof(buf), "stk:%d:%d:%llu:",
                       (int)g_pid, (int)gettid_compat(),
                       (unsigned long long)start_ns);

    int first = 1;
    for (int i = 0; i < nf && pos < (int)sizeof(buf) - 256; i++) {
        const char *sym = syms[i];

        /* Skip hook and known runtime frames. */
        int skip = 0;
        for (int k = 0; _CS_SKIP[k]; k++) {
            if (strstr(sym, _CS_SKIP[k])) { skip = 1; break; }
        }
        if (skip) continue;

        /* Extract mangled symbol between '(' and '+' (or ')').
         * backtrace_symbols gives  path(symbol+offset) [addr]
         * For PIE binaries without -rdynamic the symbol part is empty;
         * fall back to dladdr which resolves from the runtime symbol table. */
        char fname[512] = "";
        const char *lp = strchr(sym, '(');
        const char *rp = lp ? strchr(lp + 1, '+') : NULL;
        if (!rp && lp) rp = strchr(lp + 1, ')');
        if (lp && rp && rp > lp + 1) {
            size_t len = (size_t)(rp - lp - 1);
            if (len < sizeof(fname) - 1) {
                memcpy(fname, lp + 1, len);
                fname[len] = '\0';
            }
        }
        /* dladdr fallback: works when -rdynamic or the symbol is in a .so */
        if (!fname[0]) {
            Dl_info di;
            if (dladdr(frames[i], &di) && di.dli_sname && di.dli_sname[0])
                strncpy(fname, di.dli_sname, sizeof(fname) - 1);
        }
        if (!fname[0]) continue;

        /* Demangle. */
        const char *name = fname;
        char *dem = NULL;
        if (s_dem) {
            int status;
            dem = s_dem(fname, NULL, NULL, &status);
            if (status == 0 && dem) name = dem;
        }

        /* Sanitize: replace ';' (frame separator) with ',' so parsing is unambiguous. */
        if (!first) {
            if (pos < (int)sizeof(buf) - 1) buf[pos++] = ';';
        }
        for (const char *p = name; *p && pos < (int)sizeof(buf) - 2; p++) {
            buf[pos++] = (*p == ';') ? ',' : *p;
        }
        first = 0;
        free(dem);
    }
    free(syms);

    if (first) return; /* no user frames captured — don't emit */
    buf[pos++] = '\n';
    send_all(buf, pos);
}
