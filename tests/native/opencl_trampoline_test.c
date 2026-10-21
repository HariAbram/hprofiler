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
 * Also checks the span names extract_kernel_name() gives ACPP SSCP
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
    printf("RESULT trampolines=%d wrong=%d\n", TRAMPOLINE_N, bad);
    printf("RESULT names_wrong=%d\n", names_bad);
    return (bad || names_bad) ? 1 : 0;
}
