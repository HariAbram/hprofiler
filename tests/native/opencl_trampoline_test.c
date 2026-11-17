/*
 * Regression test for the OpenCL hook's JIT-kernel trampolines (driven by
 * tests/test_transport_native.py). Every trampoline g_trampolines[i] must
 * dispatch through g_slots[i]: the dlsym() override hands out trampoline i
 * after filling slot i. A trampoline that parsed its name token as a
 * decimal index ("100" for slot 64) would, from the 65th JIT kernel on,
 * dispatch through an empty or out-of-bounds slot and the kernel would
 * silently not run.
 *
 * Only slot i is populated while trampoline i is called, so a trampoline
 * reading any other slot is detected (the recorder is not called).
 *
 * Also checks the Intel USM extension wrappers (ext_wrap): each platform's
 * wrapper calls that platform's real function, a fifth platform gets the
 * real pointer back (counted), other names pass through untouched; and the
 * span names extract_kernel_name() gives ACPP SSCP
 * kernels: the short name when "ZZ<len><name>" follows, otherwise the WHOLE
 * symbol -- it used to keep the first 64 characters, which merged
 * mixbench's 100 templated SYCL kernels into 2 names.
 */
#include "../../hooks/opencl_hook/opencl_hook.c"

static int g_called_with = -1;

static void recorder(void *a0, void *a1, void *a2, void *a3, void *a4, void *a5) {
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    g_called_with = (int)(intptr_t)a0;
}

/* Intel USM extension wrappers: fake real functions, one per platform
 * slot, report which one ran (an error return keeps the recording off the
 * event table). */
static int g_ext_ran = -1;
#define FAKE_EXT(K) \
static cl_int fake_memcpy_##K(cl_command_queue q, cl_bool b, void *d, const void *s, size_t n, cl_uint nwl, \
                              const cl_event *ewl, cl_event *ev) { \
    (void)q; (void)b; (void)d; (void)s; (void)n; (void)nwl; (void)ewl; (void)ev; g_ext_ran = K; return -5; } \
static cl_int fake_fill_##K(cl_command_queue q, void *p, const void *pat, size_t ps, size_t n, cl_uint nwl, \
                            const cl_event *ewl, cl_event *ev) { \
    (void)q; (void)p; (void)pat; (void)ps; (void)n; (void)nwl; (void)ewl; (void)ev; g_ext_ran = 10 + K; return -5; }
FAKE_EXT(0)
FAKE_EXT(1)
FAKE_EXT(2)
FAKE_EXT(3)
FAKE_EXT(4)
static void *const g_fake_memcpy[5] = {(void *)fake_memcpy_0, (void *)fake_memcpy_1, (void *)fake_memcpy_2,
                                       (void *)fake_memcpy_3, (void *)fake_memcpy_4};
static void *const g_fake_fill[5] = {(void *)fake_fill_0, (void *)fake_fill_1, (void *)fake_fill_2,
                                     (void *)fake_fill_3, (void *)fake_fill_4};

static int check_ext(void) {
    int bad = 0;
    void *other = (void *)check_ext;
    if (ext_wrap((cl_platform_id)0x10, "clGetKernelSubGroupInfoKHR", other) != other) bad++;   /* not ours */
    if (ext_wrap((cl_platform_id)0x10, "clEnqueueMemcpyINTEL", NULL) != NULL) bad++;
    void *cpy[5], *fill[5];
    for (int k = 0; k < 5; k++) {                       /* platforms 0x10..0x14; slot 4 does not exist */
        cpy[k] = ext_wrap((cl_platform_id)(intptr_t)(0x10 + k), "clEnqueueMemcpyINTEL", g_fake_memcpy[k]);
        fill[k] = ext_wrap((cl_platform_id)(intptr_t)(0x10 + k), "clEnqueueMemFillINTEL", g_fake_fill[k]);
    }
    /* the same platform again: the same wrapper */
    if (ext_wrap((cl_platform_id)0x11, "clEnqueueMemcpyINTEL", g_fake_memcpy[1]) != cpy[1]) bad++;
    for (int k = 0; k < 5; k++) {
        int wrapped = k < EXT_PLATFORMS;
        if ((cpy[k] == g_fake_memcpy[k]) == wrapped || (fill[k] == g_fake_fill[k]) == wrapped) {
            fprintf(stderr, "platform %d: wrapped=%d expected %d\n", k, cpy[k] != g_fake_memcpy[k], wrapped);
            bad++;
        }
        g_ext_ran = -1;
        ((memcpy_intel_t)cpy[k])(NULL, 0, NULL, NULL, 8, 0, NULL, NULL);
        if (g_ext_ran != k) { fprintf(stderr, "memcpy wrapper %d ran %d\n", k, g_ext_ran); bad++; }
        g_ext_ran = -1;
        ((fill_intel_t)fill[k])(NULL, NULL, NULL, 4, 8, 0, NULL, NULL);
        if (g_ext_ran != 10 + k) { fprintf(stderr, "fill wrapper %d ran %d\n", k, g_ext_ran); bad++; }
    }
    if (g_ext_unwrapped != 2) { fprintf(stderr, "unwrapped=%llu want 2\n", (unsigned long long)g_ext_unwrapped); bad++; }
    return bad;
}

static int check_name(const char *sym, const char *want) {
    char *got = extract_kernel_name(sym);
    int ok = got && strcmp(got, want) == 0;
    if (!ok) fprintf(stderr, "extract_kernel_name(%s) = %s, want %s\n", sym, got ? got : "(null)", want);
    free(got);
    return ok;
}

int main(void) {
    int bad = 0;
    static char names[TRAMPOLINE_N][8];
    /* two kernels that share the first 64+ characters (nested lambdas in
     * one templated function) must keep distinct names */
    const char *nested0 = "_Z18__acpp_sscp_kernelIN7hipsycl4glue15__sscp_dispatch20ndrange_parallel_forIZZZ8"
                          "runbenchILj0EEvRNS0_4sycl5queueEPvlbbbmENKUlvE_clEvENKUlRNS5_7handlerEE_clESB_E";
    const char *nested1 = "_Z18__acpp_sscp_kernelIN7hipsycl4glue15__sscp_dispatch20ndrange_parallel_forIZZZ8"
                          "runbenchILj0EEvRNS0_4sycl5queueEPvlbbbmENKUlvE0_clEvENKUlRNS5_7handlerEE_clESB_E";
    int names_bad = !check_name(nested0, nested0) + !check_name(nested1, nested1)
                  + !check_name("_Z18__acpp_sscp_kernelIZZ17runbench_warmup_0vENKUlvE_clEvEUlvE_",
                                "runbench_warmup_0")
                  + !check_name("plain_kernel", "plain_kernel");
    /* kernel calls are SYCL work, never shown as OpenCL */
    if (strcmp(HP_JIT_KERNEL_CATEGORY, "sycl") != 0) {
        fprintf(stderr, "JIT-kernel spans use category %s, want sycl\n", HP_JIT_KERNEL_CATEGORY);
        names_bad++;
    }
    for (int i = 0; i < TRAMPOLINE_N; i++) {
        memset(g_slots, 0, sizeof(g_slots));
        g_slots[i].real_fn = (void (*)(void))recorder;
        snprintf(names[i], sizeof(names[i]), "k%d", i);
        g_slots[i].name = names[i];
        g_slots[i].used = 1;
        g_called_with = -1;
        g_trampolines[i]((void *)(intptr_t)i, NULL, NULL, NULL, NULL, NULL);
        if (g_called_with != i) {
            if (bad < 5) fprintf(stderr, "trampoline %d did not run its kernel\n", i);
            bad++;
        }
    }
    int ext_bad = check_ext();
    printf("RESULT trampolines=%d wrong=%d\n", TRAMPOLINE_N, bad);
    printf("RESULT names_wrong=%d\n", names_bad);
    printf("RESULT ext_wrong=%d\n", ext_bad);
    return (bad || names_bad || ext_bad) ? 1 : 0;
}
