// CUDA graph structure, graph versions and stream / event lives for
// tests/integration/test_cuda_graphs.py. Every node runs its own kernel, so
// the device records name the node they belong to:
//
//   G1     kA -> (kB, kC) -> empty -> kD           instantiated once (E1)
//          launched twice; then the source graph's kB parameters changed and
//          cudaGraphExecUpdate (version 2), launched; then kB disabled in E1
//          (cudaGraphNodeSetEnabled, version 3), launched: kA -> kC -> kD and
//          kA -> kD through the disabled node
//   G2     kP -> child{ kX -> kY } -> kQ           (a child graph node)
//   G3     captured from two streams: kF -> (kG on s1, kH on s2) -> kJ
//   G4     captured: kM -> host function -> kN, launched three times (the
//          host function runs once per launch)
//
// and a non-blocking stream with a priority destroyed and re-created at
// (usually) the same address, an event destroyed and re-created until its
// address comes back; both lives of each are used.
#include <cstdio>
#include <cuda_runtime.h>

#define K(name) __global__ void name(int *p) { if (threadIdx.x == 0) atomicAdd(p, 1); }
K(kA) K(kB) K(kC) K(kD) K(kP) K(kX) K(kY) K(kQ) K(kF) K(kG) K(kH) K(kJ) K(kL1) K(kL2) K(kM) K(kN)

static int host_runs = 0;
static void CUDART_CB count_run(void *) { host_runs++; }

static cudaGraphNode_t add_kernel(cudaGraph_t g, void (*fn)(int *), int **arg, const cudaGraphNode_t *deps,
                                  size_t ndeps) {
    cudaKernelNodeParams kp = {};
    kp.func = (void *)fn;
    kp.gridDim = dim3(1);
    kp.blockDim = dim3(32);
    void **args = new void *[1];
    args[0] = arg;
    kp.kernelParams = args;
    cudaGraphNode_t n;
    cudaGraphAddKernelNode(&n, g, deps, ndeps, &kp);
    return n;
}

int main() {
    int *p, *q;
    if (cudaMalloc(&p, sizeof(int)) != cudaSuccess) { printf("no GPU\n"); return 2; }
    cudaMalloc(&q, sizeof(int));
    cudaStream_t s;
    cudaStreamCreate(&s);

    // ── G1 ──────────────────────────────────────────────────────────────
    cudaGraph_t g1;
    cudaGraphCreate(&g1, 0);
    cudaGraphNode_t A = add_kernel(g1, kA, &p, NULL, 0);
    cudaGraphNode_t B = add_kernel(g1, kB, &p, &A, 1);
    cudaGraphNode_t C = add_kernel(g1, kC, &p, &A, 1);
    cudaGraphNode_t bc[2] = {B, C}, E;
    cudaGraphAddEmptyNode(&E, g1, bc, 2);
    add_kernel(g1, kD, &p, &E, 1);
    cudaGraphExec_t e1;
    cudaGraphInstantiate(&e1, g1, 0);
    cudaGraphLaunch(e1, s);
    cudaGraphLaunch(e1, s);
    cudaStreamSynchronize(s);
    // change kB's argument in the source graph, update the exec: version 2
    cudaKernelNodeParams kp = {};
    cudaGraphKernelNodeGetParams(B, &kp);
    void *nargs[1] = {&q};
    kp.kernelParams = nargs;
    cudaGraphKernelNodeSetParams(B, &kp);
    cudaGraphExecUpdateResultInfo info;
    cudaError_t ur = cudaGraphExecUpdate(e1, g1, &info);
    cudaGraphLaunch(e1, s);
    cudaStreamSynchronize(s);
    // disable kB in the exec: version 3
    cudaGraphNodeSetEnabled(e1, B, 0);
    cudaGraphLaunch(e1, s);
    cudaStreamSynchronize(s);

    // ── G2: a child graph ───────────────────────────────────────────────
    cudaGraph_t child, g2;
    cudaGraphCreate(&child, 0);
    cudaGraphNode_t X = add_kernel(child, kX, &p, NULL, 0);
    add_kernel(child, kY, &p, &X, 1);
    cudaGraphCreate(&g2, 0);
    cudaGraphNode_t P = add_kernel(g2, kP, &p, NULL, 0), CH;
    cudaGraphAddChildGraphNode(&CH, g2, &P, 1, child);
    add_kernel(g2, kQ, &p, &CH, 1);
    cudaGraphExec_t e2;
    cudaGraphInstantiate(&e2, g2, 0);
    cudaGraphLaunch(e2, s);
    cudaStreamSynchronize(s);

    // ── G3: stream capture of a fork / join ─────────────────────────────
    cudaStream_t s1, s2;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
    cudaEvent_t fork, join;
    cudaEventCreateWithFlags(&fork, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&join, cudaEventDisableTiming);
    cudaGraph_t g3;
    cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal);
    kF<<<1, 32, 0, s1>>>(p);
    cudaEventRecord(fork, s1);
    cudaStreamWaitEvent(s2, fork, 0);
    kG<<<1, 32, 0, s1>>>(p);
    kH<<<1, 32, 0, s2>>>(p);
    cudaEventRecord(join, s2);
    cudaStreamWaitEvent(s1, join, 0);
    kJ<<<1, 32, 0, s1>>>(p);
    cudaStreamEndCapture(s1, &g3);
    cudaGraphExec_t e3;
    cudaGraphInstantiate(&e3, g3, 0);
    cudaGraphLaunch(e3, s);
    cudaStreamSynchronize(s);

    // ── stream and event lives ───────────────────────────────────────────
    int lo, hi;
    cudaDeviceGetStreamPriorityRange(&lo, &hi);
    cudaStream_t life;
    cudaStreamCreateWithPriority(&life, cudaStreamNonBlocking, hi);
    void *first = (void *)life;
    kL1<<<1, 32, 0, life>>>(p);
    cudaStreamSynchronize(life);
    cudaStreamDestroy(life);
    cudaStreamCreate(&life);
    printf("stream_reused=%d\n", (void *)life == first);
    kL2<<<1, 32, 0, life>>>(p);
    cudaStreamSynchronize(life);
    // an event destroyed and re-created until the driver reuses its address
    cudaEvent_t ev;
    cudaEventCreate(&ev);
    void *efirst = (void *)ev;
    cudaEventRecord(ev, life);
    cudaEventSynchronize(ev);
    cudaEventDestroy(ev);
    int reused = 0;
    cudaEvent_t spare[16];
    int nspare = 0;
    for (int tries = 0; tries < 16 && !reused; tries++) {
        cudaEventCreate(&ev);
        if ((void *)ev == efirst) { reused = 1; break; }
        spare[nspare++] = ev;          // keep it: the next create gets another address
    }
    printf("event_reused=%d\n", reused);
    cudaEventRecord(ev, life);
    cudaEventSynchronize(ev);
    for (int i = 0; i < nspare; i++) cudaEventDestroy(spare[i]);

    // ── G4: a host function captured into a graph ───────────────────────
    cudaGraph_t g4;
    cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal);
    kM<<<1, 32, 0, s1>>>(p);
    cudaLaunchHostFunc(s1, count_run, nullptr);
    kN<<<1, 32, 0, s1>>>(p);
    cudaStreamEndCapture(s1, &g4);
    cudaGraphExec_t e4;
    cudaGraphInstantiate(&e4, g4, 0);
    for (int i = 0; i < 3; i++) cudaGraphLaunch(e4, s);
    cudaStreamSynchronize(s);
    printf("host_runs=%d\n", host_runs);

    int h = 0;
    cudaMemcpy(&h, p, sizeof(int), cudaMemcpyDeviceToHost);
    printf("update=%d count=%d prio=%d\n", ur == cudaSuccess, h, hi);
    return 0;
}
