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
 *
 * Each frame has the form:
 *   sym_name|/path/to/lib.so|0xoffset
 * where sym_name is the demangled C++ name, /path/to/lib.so is the shared
 * library containing the frame, and 0xoffset is the instruction-pointer
 * offset from the library's load base (suitable for addr2line -e lib 0xoffset).
 * If no library info is available the frame is just sym_name.
 *
 * Frames are in innermost-first order (backtrace order).
 * The receiver reverses them to build a root→leaf call tree.
 *
 * Build with HPROFILER_USE_LIBUNWIND defined to use libunwind for more
 * accurate unwinding (no -fno-omit-frame-pointer required on the profiled
 * binary).  Without it, falls back to glibc backtrace().
 *
 * Requires libunwind-dev (apt) / libunwind-devel (dnf) for the fast path.
 */
#pragma once
#include <execinfo.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

#ifdef HPROFILER_USE_LIBUNWIND
#  define UNW_LOCAL_ONLY
#  include <libunwind.h>
#endif

/* Defined here (static) — one instance per compilation unit. */
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
    "libgomp", "libiomp", "libomp", "libkmp",
    "libpthread",
    "libc.so", "libc-",
    "ld-linux", "libdl.", "libstdc++",
    NULL
};

static int _cs_should_skip(const char *s)
{
    if (!s) return 0;
    for (int k = 0; _CS_SKIP[k]; k++)
        if (strstr(s, _CS_SKIP[k])) return 1;
    return 0;
}

/* Sanitize a string written into buf at pos, replacing frame-separator
 * characters (';' and '|') with ','.  Returns new pos. */
static int _cs_write_sanitized(char *buf, int pos, int bufsz, const char *s)
{
    for (; *s && pos < bufsz - 2; s++)
        buf[pos++] = (*s == ';' || *s == '|') ? ',' : *s;
    return pos;
}

/*
 * emit_callstack — called inside emit_span while g_sock_mutex is held and
 * g_sock is known to be valid.
 *
 * Appends lib+offset to each frame for source-level resolution via addr2line:
 *   frame format: sym_name|/path/to/lib.so|0xoffset
 */
static void emit_callstack(uint64_t start_ns)
{
    if (!g_callstack) return;

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

#ifdef HPROFILER_USE_LIBUNWIND
    /* ── libunwind path (accurate, no frame-pointer required) ──────────── */
    {
        unw_context_t ctx;
        unw_cursor_t  cursor;
        if (unw_getcontext(&ctx) != 0 || unw_init_local(&cursor, &ctx) != 0)
            goto _fallback;

        while (unw_step(&cursor) > 0 && pos < (int)sizeof(buf) - 512) {
            unw_word_t ip = 0;
            unw_get_reg(&cursor, UNW_REG_IP, &ip);
            if (!ip) continue;

            /* Skip hook and runtime frames by library name. */
            Dl_info di;
            memset(&di, 0, sizeof(di));
            dladdr((void *)(uintptr_t)ip, &di);
            if (_cs_should_skip(di.dli_fname)) continue;

            /* Symbol name via libunwind (works even without -rdynamic). */
            char raw[512] = "";
            unw_word_t sym_off = 0;
            unw_get_proc_name(&cursor, raw, sizeof(raw), &sym_off);

            /* Demangle. */
            const char *name = raw[0] ? raw : "[unknown]";
            char *dem = NULL;
            if (s_dem && raw[0]) {
                int status = 0;
                dem = s_dem(raw, NULL, NULL, &status);
                if (status == 0 && dem) name = dem;
            }

            if (!first && pos < (int)sizeof(buf) - 1)
                buf[pos++] = ';';

            pos = _cs_write_sanitized(buf, pos, (int)sizeof(buf), name);
            free(dem);

            /* Append |lib|offset for addr2line resolution. */
            if (di.dli_fname && di.dli_fname[0] && di.dli_fbase) {
                uintptr_t off = (uintptr_t)ip - (uintptr_t)di.dli_fbase;
                int n = snprintf(buf + pos, sizeof(buf) - pos - 2,
                                 "|%s|0x%lx", di.dli_fname,
                                 (unsigned long)off);
                if (n > 0 && pos + n < (int)sizeof(buf) - 2)
                    pos += n;
            }
            first = 0;
        }
        if (!first) goto _done;
    }
_fallback:
    /* Reset buffer position for fallback. */
    pos = snprintf(buf, sizeof(buf), "stk:%d:%d:%llu:",
                   (int)g_pid, (int)gettid_compat(),
                   (unsigned long long)start_ns);
    first = 1;
#endif /* HPROFILER_USE_LIBUNWIND */

    /* ── glibc backtrace() fallback ─────────────────────────────────────── */
    {
        void *frames[32];
        int nf = backtrace(frames, 32);
        if (nf <= 0) return;

        char **syms = backtrace_symbols(frames, nf);
        if (!syms) return;

        for (int i = 0; i < nf && pos < (int)sizeof(buf) - 256; i++) {
            const char *sym = syms[i];
            if (_cs_should_skip(sym)) continue;

            /* Extract mangled symbol between '(' and '+' (or ')'). */
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
            /* dladdr fallback: works when -rdynamic or in a .so */
            Dl_info di2;
            memset(&di2, 0, sizeof(di2));
            dladdr(frames[i], &di2);
            if (!fname[0] && di2.dli_sname && di2.dli_sname[0])
                strncpy(fname, di2.dli_sname, sizeof(fname) - 1);
            if (!fname[0]) continue;

            /* Demangle. */
            const char *name = fname;
            char *dem = NULL;
            if (s_dem) {
                int status = 0;
                dem = s_dem(fname, NULL, NULL, &status);
                if (status == 0 && dem) name = dem;
            }

            if (!first && pos < (int)sizeof(buf) - 1)
                buf[pos++] = ';';

            pos = _cs_write_sanitized(buf, pos, (int)sizeof(buf), name);
            free(dem);

            /* Append |lib|offset from dladdr. */
            if (di2.dli_fname && di2.dli_fname[0] && di2.dli_fbase) {
                uintptr_t off = (uintptr_t)frames[i] - (uintptr_t)di2.dli_fbase;
                int n = snprintf(buf + pos, sizeof(buf) - pos - 2,
                                 "|%s|0x%lx", di2.dli_fname,
                                 (unsigned long)off);
                if (n > 0 && pos + n < (int)sizeof(buf) - 2)
                    pos += n;
            }
            first = 0;
        }
        free(syms);
    }

#ifdef HPROFILER_USE_LIBUNWIND
_done:
#endif
    if (first) return; /* no user frames captured — don't emit */
    buf[pos++] = '\n';
    send_all(buf, pos);
}
