"""
A trace that exercises every part of the critical-path model the GUI shows
(tests/test_timeline_gui_evidence.py, tests/gui_evidence_driver.py, and the
screenshots): times in microseconds.

  ranks A / B     rank B sends three messages on one channel, rank A
                  recorded two receives: an undecided pairing (alternative
                  worlds, ambiguous edges); the receives are gated calls
                  with processing tails after their release
  rank A          then a receive nobody sends to: an unknown blocking call
  ranks C, D, E   C and D send to E, which waits for both in one
                  MPI_Waitall: a joint release
  GPUs            rank A's final sync is gated by kernels on two devices whose
                  ends are closer than the two clocks' bounds: an undecidable
                  step on the observed path
  device clock    a kernel the launch puts 40 us ahead of the host and its
                  sync 30 us behind: no offset within the device's 50 us
                  bound fits both -- a clock-contradicted (disabled) edge
  nodes           a message whose send (node 1) starts after its receive
                  (node 0) ended, beyond node 1's 1 us bound: an edge dropped
                  as contradicting the timestamps
  OpenCL          a kernel recorded after the wait for it returned: the
                  OpenCL device clock is shifted back
  pass-through    rank 8 waits for rank 7, which was itself waiting in a GPU
                  sync: that blame passes on to the kernel

Dependencies the builders cannot infer from tags alone (the device waits,
the OpenCL launch and wait) are persisted with the trace's edges, the way
`load_or_build_graph` finds them.
"""
from __future__ import annotations

import numpy as np

from src.core.events import Category, SpanEvent
from src.core.trace import Trace, TraceMetadata

US = 1000


def _s(name, cat, a, b, pid, tid=None, sid="", psid="", **tags):
    tags.setdefault("node", 0)              # one node unless a span says otherwise
    return SpanEvent(name, Category(cat), int(a * US), int((b - a) * US), pid, pid if tid is None else tid,
                     tags={k: str(v) for k, v in tags.items()}, span_id=sid, parent_span_id=psid)


def _work(name, pid, a, b, tid=None):
    return _s(name, "openmp", a, b, pid, tid, type="work")


def _send(pid, a, rank, peer, tag, end=None, node=None):
    extra = {"node": node} if node is not None else {}
    return _s("MPI_Send", "mpi", a, a + 50 if end is None else end, pid, type="send", rank=rank, peer=peer,
              wpeer=peer, tag=tag, commid=0, **extra)


def _recv(pid, a, b, rank, peer, tag, sid="", node=None):
    extra = {"node": node} if node is not None else {}
    return _s("MPI_Recv", "mpi", a, b, pid, sid=sid, type="recv", rank=rank, peer=peer, wpeer=peer, tag=tag,
              commid=0, **extra)


def evidence_trace(*, extra_edges: bool = True) -> Trace:
    t = Trace(TraceMetadata(command="mpirun -np 5 ./evidence", args=[]))
    t.metadata.clock = {"merge": {"nodes": [{"bound_ns": 0}, {"bound_ns": 1000}]}}
    spans = [
        # rank B (pid 2): three sends on one channel
        _work("c1", 2, 0, 300), _send(2, 300, 1, 0, 9), _work("c2", 2, 350, 800), _send(2, 800, 1, 0, 9),
        _work("c3", 2, 850, 1300), _send(2, 1300, 1, 0, 9),
        # rank A (pid 1): two receives, then one nobody sends to
        _work("prep", 1, 0, 200), _recv(1, 200, 400, 0, 1, 9), _work("mid", 1, 400, 600),
        _recv(1, 600, 1400, 0, 1, 9), _work("solve", 1, 1400, 1800), _recv(1, 1800, 2000, 0, 1, 11),
        _work("post", 1, 2000, 2300),
        # ranks C, D (pids 3, 4) -> E (pid 5): a wait released by both at once
        _work("produce", 3, 0, 900), _send(3, 900, 2, 4, 7), _work("produce", 4, 0, 900), _send(4, 900, 3, 4, 7),
        _work("gather_prep", 5, 0, 100),
        _s("MPI_Irecv", "mpi", 100, 105, 5, sid="ra", type="irecv", rank=4, peer=2, wpeer=2, tag=7, commid=0),
        _s("MPI_Irecv", "mpi", 105, 110, 5, sid="rb", type="irecv", rank=4, peer=3, wpeer=3, tag=7, commid=0),
        _s("MPI_Waitall", "mpi", 110, 1000, 5, psid="ra;rb", type="waitall", rank=4),
        _work("combine", 5, 1000, 1500),
        # rank A then launches a kernel on each of two GPUs and waits for both:
        # they end 10 us apart, closer than the two device clocks' bounds
        _s("cudaLaunchKernel", "cuda", 2300, 2305, 1, type="launch"),
        _s("cudaLaunchKernel", "cuda", 2305, 2310, 1, type="launch"),
        _s("k_left", "cuda", 2320, 2500, 1, tid=50, side="gpu", rt="cuda", op="kernel", timing="device", dev=0,
           stream=1),
        _s("k_right", "cuda", 2320, 2510, 1, tid=51, side="gpu", rt="cuda", op="kernel", timing="device", dev=1,
           stream=1),
        _s("cudaDeviceSynchronize", "sync", 2310, 2620, 1, type="sync"),
        # a device whose clock the launch puts >= 40 us ahead and the sync
        # >= 30 us behind (pid 10): no offset within its bound fits both
        _s("cudaLaunchKernel", "cuda", 100, 110, 10, type="launch"),
        _s("k_late", "cuda", 60, 200, 10, tid=60, side="gpu", rt="cuda", op="kernel", timing="device", dev=0,
           stream=2),
        _s("cudaStreamSynchronize", "sync", 120, 170, 10, type="sync"),
        # two nodes (pids 7, 8): the send on node 1 starts after the receive on node 0 ended
        _recv(8, 0, 400, 6, 5, 3, node=0), _work("after_recv", 8, 400, 900), _send(7, 405, 5, 6, 3, node=1),
        # pass-through (pids 11, 12): rank 8 waits in MPI_Recv for rank 7, which
        # was itself waiting in a GPU sync -- the receive's blame passes on to the kernel
        _s("cudaLaunchKernel", "cuda", 0, 10, 11, type="launch"),
        _s("pack", "cuda", 20, 420, 11, tid=70, side="gpu", rt="cuda", op="kernel", timing="device", dev=0, stream=3),
        _s("cudaStreamSynchronize", "sync", 15, 425, 11, type="sync"),
        _send(11, 430, 7, 8, 5, end=440),
        _work("compute", 12, 0, 150), _recv(12, 150, 450, 8, 7, 5), _work("unpack", 12, 450, 650),
        # OpenCL (pid 9): the device clock recorded the kernel after the wait returned
        _s("clEnqueueNDRangeKernel", "opencl", 0, 10, 9, side="cpu", type="kernel"),
        _s("ocl_kernel", "opencl", 500, 600, 9, tid=99, side="gpu", type="kernel"),
        _s("clWaitForEvents", "sync", 20, 200, 9, type="sync"),
    ]
    for s in spans:
        t.add(s)
    if extra_edges:
        _persist_extra_edges(t)
    return t


def _persist_extra_edges(t: Trace) -> None:
    from src.analysis import criticalpath as cp
    from src.core.store import EDGE_DTYPE
    ordinal = {(s.pid, s.tid, s.name, s.start_ns): i for i, s in enumerate(t.iter_spans())}

    def o(pid, tid, name, a):
        return ordinal[(pid, tid, name, int(a * US))]
    W, L = cp._KIND_CODE["device_wait"], cp._KIND_CODE["launch"]
    c = cp._CONF_CODE["certain"]
    sync = o(1, 1, "cudaDeviceSynchronize", 2310)
    left, right = o(1, 50, "k_left", 2320), o(1, 51, "k_right", 2320)
    late = o(10, 60, "k_late", 60)
    extra = [(left, o(1, 1, "cudaLaunchKernel", 2300), L, c), (right, o(1, 1, "cudaLaunchKernel", 2305), L, c),
             (sync, left, W, c), (sync, right, W, c),
             (late, o(10, 10, "cudaLaunchKernel", 100), L, c), (o(10, 10, "cudaStreamSynchronize", 120), late, W, c),
             (o(11, 70, "pack", 20), o(11, 11, "cudaLaunchKernel", 0), L, c),
             (o(11, 11, "cudaStreamSynchronize", 15), o(11, 70, "pack", 20), W, c),
             (o(9, 99, "ocl_kernel", 500), o(9, 9, "clEnqueueNDRangeKernel", 0), L, c),
             (o(9, 9, "clWaitForEvents", 20), o(9, 99, "ocl_kernel", 500), W, c)]
    seqs = t.store.span_seqs()
    edges = cp.build_edges(t, seqs)
    more = np.array(extra, dtype=EDGE_DTYPE)
    t.store.save_edges(cp.edges_version(), np.concatenate([edges, more]))


def big_trace(n_iter: int = 40) -> Trace:
    """A trace with an undecided channel, made larger than a lowered
    automatic alternatives limit by repetition (the GUI tests lower
    criticalpath.ALTERNATIVES_AUTO_NODE_LIMIT instead of building 50k spans)."""
    t = Trace(TraceMetadata(command="mpirun -np 2 ./loop", args=[]))
    c = 0
    for _ in range(n_iter):
        for s in (_work("compute", 2, c, c + 300), _send(2, c + 300, 1, 0, 9),
                  _work("compute", 1, c, c + 200), _recv(1, c + 200, c + 400, 0, 1, 9)):
            t.add(s)
        c += 500
    for s in (_send(2, c, 1, 0, 9), _send(2, c + 100, 1, 0, 9), _recv(1, c + 50, c + 300, 0, 1, 9),
              _work("tail", 1, c + 300, c + 600)):
        t.add(s)
    return t


def calibration_file(path: str) -> str:
    """A calibration for the posting-order point-to-point stratum (synthetic:
    for the GUI's calibrated state, not a real calibration)."""
    from src.analysis import edge_calibration as ec
    from src.analysis.evidence import evidence_version
    # scores (edge_calibration.evidence_score): ambiguous / medium / inferred
    # 210, inferred / medium / inferred 310
    cal = ec.Calibration(strata={"p2p/posting_order": {"x": [210.0, 310.0], "p": [0.55, 0.9], "n": 40}},
                         dataset="synthetic GUI test scenarios", examples=40, min_examples=20,
                         evidence_version=evidence_version())
    cal.save(path)
    return path
