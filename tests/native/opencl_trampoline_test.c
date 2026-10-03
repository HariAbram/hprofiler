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
 */
#include "../../hooks/opencl_hook/opencl_hook.c"

static int g_called_with = -1;

static void recorder(void *a0, void *a1, void *a2, void *a3, void *a4, void *a5) {
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    g_called_with = (int)(intptr_t)a0;
}

int main(void) {
    int bad = 0;
    for (int i = 0; i < TRAMPOLINE_N; i++) {
        memset(g_slots, 0, sizeof(g_slots));
        g_slots[i].real_fn = (void (*)(void))recorder;
        snprintf(g_slots[i].name, sizeof(g_slots[i].name), "k%d", i);
        g_slots[i].used = 1;
        g_called_with = -1;
        g_trampolines[i]((void *)(intptr_t)i, NULL, NULL, NULL, NULL, NULL);
        if (g_called_with != i) {
            if (bad < 5) fprintf(stderr, "trampoline %d did not run its kernel\n", i);
            bad++;
        }
    }
    printf("RESULT trampolines=%d wrong=%d\n", TRAMPOLINE_N, bad);
    return bad ? 1 : 0;
}
