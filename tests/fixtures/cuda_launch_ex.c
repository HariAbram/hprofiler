/*
 * Triton's launch pattern (triton/backends/nvidia/driver.py): a module
 * loaded from PTX with cuModuleLoadData, its function from
 * cuModuleGetFunction, launched through cuLaunchKernelEx obtained with
 * dlopen("libcuda.so.1") + dlsym -- a call no LD_PRELOAD wrapper sees --
 * and once more through cuLaunchKernel (linked, intercepted).
 * tests/integration/test_cuda_module_launches.py checks that every kernel
 * span carries the module's id and the disassembly counts both launches.
 *
 * Build: gcc -I<cuda>/include cuda_launch_ex.c -o cuda_launch_ex -lcuda -ldl
 */
#include <cuda.h>
#include <dlfcn.h>
#include <stdio.h>

static const char ptx[] =
    ".version 7.0\n"
    ".target sm_52\n"
    ".address_size 64\n"
    ".visible .entry scale_kernel(.param .u64 p)\n"
    "{\n"
    "\t.reg .b64 %rd<2>;\n"
    "\t.reg .f32 %f<3>;\n"
    "\tld.param.u64 %rd1, [p];\n"
    "\tcvta.to.global.u64 %rd1, %rd1;\n"
    "\tld.global.f32 %f1, [%rd1];\n"
    "\tadd.f32 %f2, %f1, %f1;\n"
    "\tst.global.f32 [%rd1], %f2;\n"
    "\tret;\n"
    "}\n";

typedef CUresult (*launch_ex_t)(const CUlaunchConfig *, CUfunction, void **, void **);

#define CHECK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { \
        fprintf(stderr, "%s failed: %d\n", #x, (int)r_); return 1; } } while (0)

int main(void) {
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction fn;
    CUdeviceptr buf;
    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuCtxCreate(&ctx, 0, dev));
    CHECK(cuModuleLoadData(&mod, ptx));
    CHECK(cuModuleGetFunction(&fn, mod, "scale_kernel"));
    CHECK(cuMemAlloc(&buf, sizeof(float)));
    CHECK(cuMemsetD32(buf, 0x3f800000u, 1));                 /* 1.0f */
    void *args[] = { &buf };

    void *libcuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    launch_ex_t launch_ex = libcuda ? (launch_ex_t)dlsym(libcuda, "cuLaunchKernelEx") : NULL;
    if (!launch_ex) { fprintf(stderr, "no cuLaunchKernelEx\n"); return 1; }
    CUlaunchConfig cfg = {0};
    cfg.gridDimX = cfg.gridDimY = cfg.gridDimZ = 1;
    cfg.blockDimX = cfg.blockDimY = cfg.blockDimZ = 1;
    for (int i = 0; i < 3; i++)
        CHECK(launch_ex(&cfg, fn, args, NULL));                /* 3 launches the hook cannot see */
    CHECK(cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL));
    CHECK(cuCtxSynchronize());

    float out = 0;
    CHECK(cuMemcpyDtoH(&out, buf, sizeof(out)));
    printf("cuda_launch_ex: %.1f\n", out);                    /* 1 * 2^4 = 16.0 */
    cuMemFree(buf);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    return out == 16.0f ? 0 : 1;
}
