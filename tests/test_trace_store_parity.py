"""
Parity between MemoryTraceStore and DiskTraceStore (src/core/store).

The same synthetic event stream -- several processes and threads, nested
spans, zero-duration spans, perf samples, CUDA host/device-model spans and
pre-split GPU spans, MPI point-to-point/collectives with request ids,
OpenMP barriers, stacks, NVTX ranges, instants, counters with units -- is
written into a memory-backed and a disk-backed Trace. Every consumer-facing
result must be identical: events, lanes, window queries, aggregates,
exclusive-time attribution, activity bins, text summary and dashboard
numbers, Chrome JSON export (byte for byte) and its reload, critical path,
multi-node merge, and metadata after save/reopen. Where an independent
reference exists (the original ExclusiveTime, the dict-based critical-path
pipeline, brute-force window filtering) results are checked against it too.
"""
from __future__ import annotations

import copy
import io
import json
import os
import random
import shutil
import sqlite3
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import activity_buckets, criticalpath as cp, dashboard as dash
from src.analysis import multinode as mn
from src.analysis.cct import gpu_starvation
from src.analysis.pop_efficiency import useful_time_by_pid
from src.core import trace_io
from src.core.events import Category, CounterEvent, InstantEvent, SpanEvent
from src.core.store import (DERIVED_VERSION, SCHEMA_VERSION, DiskTraceStore, MemoryTraceStore,
                            SpanFilter, StoreError)
from src.core.store.common import occupancy_from_spans
from src.core.trace import Trace, TraceMetadata
from src.output import chrome_trace
from src.output.summary import print_summary

T0 = 1_000_000_000


def synthetic_events(seed: int = 7, n: int = 2500) -> list:
    """A deterministic event stream exercising every lane rule, tag kind
    and dependency mechanism the stores and analyses care about."""
    rng = random.Random(seed)
    ev: list = []
    sid = 0
    for i in range(n):
        pid = rng.choice([101, 102, 103])
        tid = rng.choice([1, 2, 3])
        start = T0 + rng.randrange(0, 20_000_000)
        dur = rng.choice([0, rng.randrange(100, 300_000)])
        kind = rng.random()
        tags: dict = {}
        stack: list[str] = []
        span_id = parent = ""
        if kind < 0.20:                               # OpenMP work + nested barrier
            cat, name = Category.OPENMP, f"omp_region_{rng.randrange(4)}"
            tags = {"type": "parallel"}
            ev.append(SpanEvent(name, cat, start, dur + 50_000, pid, tid, tags))
            cat, name, tags = Category.SYNC, "omp_barrier", {"type": "barrier"}
            start, dur = start + 10_000, 20_000
        elif kind < 0.35:                             # MPI
            cat = Category.MPI
            typ = rng.choice(["send", "recv", "isend", "irecv", "barrier", "allreduce"])
            name = {"send": "MPI_Send", "recv": "MPI_Recv", "isend": "MPI_Isend", "irecv": "MPI_Irecv",
                    "barrier": "MPI_Barrier", "allreduce": "MPI_Allreduce"}[typ]
            tags = {"type": typ, "rank": str(pid - 101), "peer": str(rng.randrange(3)),
                    "tag": str(rng.randrange(2)), "commid": rng.choice(["-1", "4"])}
            if typ in ("isend", "irecv"):
                sid += 1
                span_id = f"r{sid}"
                ev.append(SpanEvent(name, cat, start, dur, pid, tid, tags, span_id=span_id))
                name, tags, span_id = "MPI_Wait", {"type": "wait", "rank": str(pid - 101)}, ""
                parent = f"r{sid}"
                start, dur = start + dur + 1_000, rng.randrange(1_000, 100_000)
        elif kind < 0.45:                             # CUDA host/device model
            sid += 1
            stream = str(rng.randrange(1, 3))
            host = {"type": "launch", "op": "kernel", "stream": stream, "side": "cpu", "rt": "cuda",
                    "timing": "host", "lid": str(sid), "corr": str(sid)}
            ev.append(SpanEvent("cudaLaunchKernel", Category.GPU_CUDA, start, 5_000, pid, tid, host,
                                span_id=f"g{sid}"))
            cat, name = Category.GPU_CUDA, f"_Z6kernel{rng.randrange(3)}v"
            tags = {"type": "kernel", "side": "gpu", "rt": "cuda", "op": "kernel", "timing": "device",
                    "src": "cupti", "corr": str(sid), "dev": "0", "ctx": "1", "nstream": str(int(stream) + 6)}
            start, dur, tid = start + 20_000, rng.randrange(1_000, 200_000), 0
        elif kind < 0.50:                             # pre-split CUDA span + OpenCL pair
            cat, name = rng.choice([(Category.GPU_CUDA, "old_kernel"), (Category.GPU_OPENCL, "ocl_k")])
            tags = {"type": "kernel", "stream": "0"} if cat == Category.GPU_CUDA else \
                {"type": "kernel", "side": rng.choice(["cpu", "gpu"])}
        elif kind < 0.60:                             # perf samples with stacks
            cat, name, dur = Category.CPU, f"hot_{rng.randrange(5)}", rng.choice([0, 1_010])
            stack = [f"caller_{rng.randrange(3)}", "main"]
        elif kind < 0.65:
            cat, name, tags = Category.NVTX, "phase", {"type": "nvtx_range"}
            sid += 1
            span_id = f"n{sid}"
        else:
            cat = rng.choice([Category.MEMORY, Category.JIT, Category.SYNC])
            name = rng.choice(["cudaMalloc", "jit_compile", "cudaDeviceSynchronize"])
            tags = {"type": rng.choice(["alloc", "jit_compile", "sync"]), "bytes": str(rng.randrange(1 << 20)),
                    "extra": rng.choice([1, 2.5, True, None, ["a", 1]])}
        ev.append(SpanEvent(name, cat, start, dur, pid, tid, tags, stack, span_id, parent))
        if i % 41 == 0:
            ev.append(InstantEvent("mark", Category.MPI, start, pid, tid, {"k": str(i)}))
        if i % 23 == 0:
            ev.append(CounterEvent(rng.choice(["ipc", "gpu_mem_used_bytes", "process_max_rss_bytes"]),
                                   Category.CPU, start, rng.random() * 100, rng.choice(["", "bytes"]), pid))
    return ev


def metadata() -> TraceMetadata:
    return TraceMetadata(command="./app", args=["-n", "4"], start_time_ns=T0, end_time_ns=T0 + 25_000_000,
                         pid=101, backends_used=["openmp", "mpi", "cuda"], hostname="h", cwd="/w",
                         capture_time_iso="2026-10-01T00:00:00",
                         device_activity={"101/cuda": {"status": "active", "dropped": 3}})


def build(store, events) -> Trace:
    t = Trace(metadata(), store=store)
    for e in events:
        t.add(copy.deepcopy(e))
    t._pc_samples["k"] = [(16, 2, 5)]
    t.store.flush()
    return t


def span_key(s: SpanEvent) -> tuple:
    return (s.seq, s.name, s.category.value, s.start_ns, s.duration_ns, s.pid, s.tid,
            json.dumps(s.tags, sort_keys=True), tuple(s.stack_frames), s.span_id, s.parent_span_id)


class _Pair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hprofiler_parity_")
        cls.events = synthetic_events()
        cls.mem = build(MemoryTraceStore(), cls.events)
        cls.disk = build(DiskTraceStore(os.path.join(cls.tmp, "t.hpstore"), batch_size=300), cls.events)

    @classmethod
    def tearDownClass(cls):
        cls.disk.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)


class TestEventsAndLanes(_Pair):
    def test_events_identical(self):
        self.assertEqual([span_key(s) for s in self.mem.spans], [span_key(s) for s in self.disk.spans])
        ik = lambda e: (e.seq, e.name, e.timestamp_ns, e.pid, e.tid, e.tags)
        self.assertEqual([ik(e) for e in self.mem.instants], [ik(e) for e in self.disk.instants])
        ck = lambda e: (e.seq, e.name, e.timestamp_ns, e.value, e.unit, e.pid)
        self.assertEqual([ck(e) for e in self.mem.counters], [ck(e) for e in self.disk.counters])
        self.assertEqual(self.mem.span_count(), self.disk.span_count())

    def test_tags_are_lossless(self):
        # JSON-typed tag values (ints, floats, bools, None, lists) survive.
        values = {repr(s.tags.get("extra")) for s in self.disk.spans if "extra" in s.tags}
        self.assertEqual(values, {"1", "2.5", "True", "None", "['a', 1]"})

    def test_lanes_identical(self):
        self.assertEqual(self.mem.lane_infos(), self.disk.lane_infos())
        lm, ld = self.mem.lanes(), self.disk.lanes()
        self.assertEqual(list(lm), list(ld))
        for name in lm:
            self.assertEqual([span_key(s) for s in lm[name]], [span_key(s) for s in ld[name]])
        self.assertTrue(any("@" in n for n in lm))        # shared bases get @PID

    def test_event_by_id_round_trip(self):
        for s in list(self.disk.iter_spans())[::97]:
            self.assertEqual(span_key(self.disk.event_by_id(s.eid)), span_key(s))
        for s in list(self.mem.iter_spans())[::97]:
            self.assertEqual(span_key(self.mem.event_by_id(s.eid)), span_key(s))


class TestQueries(_Pair):
    def test_windows_match_each_other_and_brute_force(self):
        rng = random.Random(3)
        filters = [None, SpanFilter(min_dur_ns=50_000), SpanFilter(name_query="mpi_"),
                   SpanFilter(name_query="^_Z6", name_is_regex=True), SpanFilter(buckets=frozenset({"Computation"})),
                   SpanFilter(range_start_ns=T0 + 1_000_000, range_end_ns=T0 + 2_000_000)]
        for ln in self.mem.lane_infos():
            all_spans = list(self.mem.iter_spans(lane=ln.name))
            for _ in range(3):
                a = T0 + rng.randrange(0, 20_000_000)
                b = a + rng.choice([1_000, 100_000, 5_000_000])
                for f in filters:
                    wm = [span_key(s) for s in self.mem.store.window(ln.name, a, b, filt=f)]
                    wd = [span_key(s) for s in self.disk.store.window(ln.name, a, b, filt=f)]
                    self.assertEqual(wm, wd, (ln.name, a, b, f))
                    brute = sorted((s for s in all_spans if s.start_ns <= b and s.end_ns > a
                                    and (f is None or f.matches(s))), key=lambda s: (s.start_ns, s.seq))
                    self.assertEqual(wm, [span_key(s) for s in brute])
                    self.assertEqual(self.mem.store.count_window(ln.name, a, b, filt=f),
                                     self.disk.store.count_window(ln.name, a, b, filt=f))
                self.assertEqual(self.mem.store.lane_count(ln.name, filters[1]),
                                 self.disk.store.lane_count(ln.name, filters[1]))

    def test_lanes_and_events_in_window(self):
        a, b = T0 + 3_000_000, T0 + 4_000_000
        lm = {k: [span_key(s) for s in v] for k, v in self.mem.lanes_in_window(a, b).items()}
        ld = {k: [span_key(s) for s in v] for k, v in self.disk.lanes_in_window(a, b).items()}
        self.assertEqual(lm, ld)
        em = [span_key(s) for s in self.mem.events_in_window(a, b)]
        self.assertEqual(em, [span_key(s) for s in self.disk.events_in_window(a, b)])
        self.assertEqual(em, sorted(em, key=lambda k: (k[3], k[0])))

    def test_lightweight_scans_identical_and_correct(self):
        excl = ("barrier", "alloc")
        for kw in ({}, {"categories": ("mpi", "sync")}, {"pid": 102}, {"pid": 103, "categories": ("cuda",)}):
            for timed in (True, False):
                im = list(self.mem.store.iter_intervals(exclude_types=excl, timed_only=timed, **kw))
                self.assertEqual(im, list(self.disk.store.iter_intervals(exclude_types=excl, timed_only=timed, **kw)))
                want = sorted(((s.start_ns, s.end_ns, s.seq) for s in self.mem.iter_spans(
                    pid=kw.get("pid"), categories=kw.get("categories"))
                    if (not timed or s.duration_ns > 0) and s.tags.get("type") not in excl))
                self.assertEqual(im, [(a, b) for a, b, _q in want])
        for cats in (None, ("cpu",)):
            got = [sorted(zip(*map(np.concatenate, zip(*st.interval_arrays(categories=cats, chunk=97)))))
                   for st in (self.mem.store, self.disk.store)]
            self.assertEqual(got[0], got[1])
            self.assertEqual(len(got[0]), sum(1 for _ in self.mem.iter_spans(categories=cats)))
        fm, fd = self.mem.store.first_with_tag("bytes"), self.disk.store.first_with_tag("bytes")
        self.assertEqual(list(fm), list(fd))
        self.assertEqual([span_key(s) for s in fm.values()], [span_key(s) for s in fd.values()])
        self.assertGreater(len(fm), 0)

    def test_aggregates_identical(self):
        self.assertEqual(self.mem.aggregate_stats(), self.disk.aggregate_stats())
        self.assertEqual(self.mem.store.pids(), self.disk.store.pids())
        self.assertEqual(self.mem.store.threads(), self.disk.store.threads())
        self.assertEqual(self.mem.store.span_extent(), self.disk.store.span_extent())
        self.assertEqual(self.mem.store.event_extent(), self.disk.store.event_extent())

    def test_exclusive_time_identical_and_equal_to_original(self):
        em, ed = self.mem.store.exclusive_aggregate(), self.disk.store.exclusive_aggregate()
        self.assertEqual(list(em.rows.items()), list(ed.rows.items()))   # same order too
        self.assertEqual(em.threads, ed.threads)
        et = activity_buckets.ExclusiveTime(self.mem.spans)
        reference = et.totals(lambda i, s: (s.pid, s.tid, s.category.value, s.name, et.bucket[i], et.device[i]))
        self.assertEqual(dict(sorted(reference.items())), em.rows)
        self.assertEqual(et.threads, em.threads)

    def test_activity_index_identical_and_correct(self):
        for ln in self.mem.lane_infos():
            am, ad = self.mem.store.activity(ln.name), self.disk.store.activity(ln.name)
            self.assertEqual(len(am), len(ad))
            for x, y in zip(am, ad):
                np.testing.assert_array_equal(x["busy"], y["busy"])
            np.testing.assert_array_equal(am[0]["starts"], ad[0]["starts"])
            self.assertEqual(int(am[0]["starts"].sum()), ln.count)
            # coarsest level == exact union coverage at that resolution
            lv = am[-1]
            a, b = lv["t0"], lv["t0"] + lv["bin_ns"] * lv["nbins"]
            exact = occupancy_from_spans(((s.start_ns, s.end_ns) for s in self.mem.iter_spans(
                order="start", lane=ln.name)), a, b, lv["nbins"])
            np.testing.assert_allclose(np.asarray(lv["busy"], dtype=float) / lv["bin_ns"], exact, atol=2e-4)
            occ_m = self.mem.store.occupancy(ln.name, a, b, 100)
            np.testing.assert_allclose(occ_m, self.disk.store.occupancy(ln.name, a, b, 100), atol=1e-6)


class TestDerivedResults(_Pair):
    def test_text_summary_identical(self):
        out = []
        for t in (self.mem, self.disk):
            buf = io.StringIO()
            with redirect_stdout(buf):
                print_summary(t)
            out.append(buf.getvalue())
        self.assertEqual(out[0], out[1])

    def test_dashboard_numbers_identical(self):
        for f in (dash.wait_fraction, dash.diagnose, dash.top_findings, dash.trace_wall_ns,
                  gpu_starvation, useful_time_by_pid):
            self.assertEqual(f(self.mem), f(self.disk), f.__name__)
        self.assertEqual(dash.bottleneck_analysis(self.mem, {}), dash.bottleneck_analysis(self.disk, {}))

    def test_critical_path_identical_and_equal_to_dict_pipeline(self):
        rm, rd = cp.analyze(self.mem), cp.analyze(self.disk)
        pm = [span_key(rm.spans[i]) for i in rm.path_span_indices]
        pd = [span_key(rd.spans[i]) for i in rd.path_span_indices]
        self.assertEqual(pm, pd)
        for attr in ("path_span_indices", "path_edge_confidence", "path_edge_kinds", "total_path_ns",
                     "wall_ns", "time_on_path_by_category", "wait_caused_by_category", "notes"):
            self.assertEqual(getattr(rm, attr), getattr(rd, attr), attr)
        spans, preds = cp.build_dependency_graph(self.mem)
        self.assertEqual(cp._compute_path_full(spans, preds),
                         (rm.path_span_indices, rm.path_edge_confidence, rm.path_edge_kinds))
        graph = cp.load_or_build_graph(self.disk)
        self.assertEqual(graph.preds_dict(), {k: v for k, v in preds.items() if v})

    def test_streaming_edge_builders_equal_dict_builders_on_mpi_corner_cases(self):
        # Wildcard receives resolved by Wait (rpeer/rtag) or Waitall
        # (rmatches), multi-request psids, reused request ids, scoped and
        # unscoped collectives, OpenMP barriers in several processes.
        for seed in range(8):
            rng = random.Random(seed)
            ev = []
            for k in range(600):
                pid, tid = rng.choice([1, 2, 3]), rng.choice([1, 2])
                st, d = T0 + rng.randrange(0, 2_000_000), rng.randrange(1, 50_000)
                rank, peer, tag = str(pid - 1), str(rng.randrange(3)), str(rng.randrange(2))
                r = rng.random()
                if r < 0.25:
                    typ = rng.choice(["send", "isend", "recv"])
                    tags = {"type": typ, "rank": rank, "peer": peer, "tag": tag}
                    if typ == "recv" and rng.random() < 0.3:
                        tags["wildcard"] = "1"
                    ev.append(SpanEvent("MPI_" + typ.title(), Category.MPI, st, d, pid, tid, tags,
                                        span_id=f"q{rng.randrange(40)}" if typ == "isend" else ""))
                elif r < 0.45:
                    rid = f"q{rng.randrange(40)}"
                    tags = {"type": "irecv", "rank": rank, "tag": tag}
                    if rng.random() < 0.4:
                        tags["wildcard"] = "1"
                    if rng.random() < 0.8:
                        tags["peer"] = peer
                    ev.append(SpanEvent("MPI_Irecv", Category.MPI, st, d, pid, tid, tags, span_id=rid))
                elif r < 0.65:
                    ids = [f"q{rng.randrange(40)}" for _ in range(rng.choice([1, 1, 2, 3]))]
                    tags = {"type": "wait", "rank": rank}
                    if len(ids) == 1 and rng.random() < 0.5:
                        tags.update(rpeer=peer, rtag=tag)
                    elif rng.random() < 0.6:
                        tags["rmatches"] = ";".join(f"{i}/{rng.randrange(3)}/{tag}" for i in ids[:2])
                    name = "MPI_Wait" if len(ids) == 1 else rng.choice(["MPI_Waitall", "MPI_Waitsome"])
                    ev.append(SpanEvent(name, Category.MPI, st, d, pid, tid, tags, parent_span_id=";".join(ids)))
                elif r < 0.85:
                    tags = {"type": rng.choice(["allreduce", "barrier", "bcast"]), "rank": rank,
                            "commid": rng.choice(["-1", "7", "9"])}
                    if rng.random() < 0.2:
                        del tags["commid"]
                    ev.append(SpanEvent("MPI_Coll", Category.MPI, st, d, pid, tid, tags))
                else:
                    ev.append(SpanEvent(rng.choice(["omp_barrier", "omp_taskwait"]), Category.SYNC,
                                        st, d, pid, tid, {"type": "barrier"}))
            for store in (MemoryTraceStore(), DiskTraceStore(os.path.join(self.tmp, f"mpi{seed}.hpstore"))):
                t = build(store, ev)
                spans, preds = cp.build_dependency_graph(t)
                self.assertEqual(cp.load_or_build_graph(t).preds_dict(), {k: v for k, v in preds.items() if v})
                r = cp.analyze(t)
                self.assertEqual(cp._compute_path_full(spans, preds),
                                 (r.path_span_indices, r.path_edge_confidence, r.path_edge_kinds))
                self.assertEqual(mn.validate_causality(t), mn.validate_causality(build(MemoryTraceStore(), ev)))
                t.close()

    def test_call_tree_identical(self):
        from src.analysis.call_tree import build_call_tree, _ct_build
        ser = lambda nodes: [(n.name, n.category, n.total_ns, n.count, n.self_ns, ser(n.children)) for n in nodes]
        self.assertEqual(ser(build_call_tree(self.mem)), ser(build_call_tree(self.disk)))
        self.assertEqual(ser(build_call_tree(self.mem)), ser(_ct_build(self.mem.spans)))


class TestExportsAndPersistence(_Pair):
    def test_chrome_json_identical_and_reloads(self):
        bm, bd = io.StringIO(), io.StringIO()
        chrome_trace.write(self.mem, bm)
        chrome_trace.write(self.disk, bd)
        self.assertEqual(bm.getvalue(), bd.getvalue())
        data = json.loads(bm.getvalue())                       # still one valid JSON document
        self.assertEqual(len([e for e in data["traceEvents"] if e["ph"] == "X"]), self.mem.span_count())
        path = os.path.join(self.tmp, "x.json")
        Path(path).write_text(bm.getvalue())
        r_mem = chrome_trace.load_trace_from_json(path)
        r_disk = trace_io.open_trace(path, disk=True)
        try:
            self.assertEqual([span_key(s)[1:] for s in r_mem.spans], [span_key(s)[1:] for s in r_disk.spans])
            self.assertEqual(r_mem.aggregate_stats(), self.mem.aggregate_stats())
            self.assertEqual(r_disk.aggregate_stats(), self.mem.aggregate_stats())
            for r in (r_mem, r_disk):
                self.assertEqual(r.metadata.command, "./app")
                self.assertEqual(r.metadata.device_activity, metadata().device_activity)
                self.assertEqual((r.metadata.start_time_ns, r.metadata.end_time_ns), (T0, T0 + 25_000_000))
        finally:
            r_disk.close()

    def test_old_single_line_json_still_loads(self):
        bm = io.StringIO()
        chrome_trace.write(self.mem, bm)
        data = json.loads(bm.getvalue())
        old = {"traceEvents": data["traceEvents"], "displayTimeUnit": "ms", "metadata": data["metadata"]}
        path = os.path.join(self.tmp, "old.json")
        Path(path).write_text(json.dumps(old))
        r = chrome_trace.load_trace_from_json(path)
        self.assertEqual(r.aggregate_stats(), self.mem.aggregate_stats())

    def test_store_reopen_restores_metadata_and_derived_tables(self):
        path = os.path.join(self.tmp, "reopen.hpstore")
        t = build(DiskTraceStore(path), self.events)
        t._has_stacks = True
        t.set_devices([])
        t.finalize()
        mem_rows = list(self.mem.store.exclusive_aggregate().rows.items())
        self.assertEqual(list(t.store.exclusive_aggregate().rows.items()), mem_rows)
        expected = (t.lane_infos(), t.aggregate_stats(), t.store.exclusive_aggregate().rows,
                    t.store.activity(t.lane_infos()[0].name)[0]["busy"].tolist())
        t.close()
        r = trace_io.open_trace(path)
        try:
            self.assertTrue(r.store.is_finalized())
            self.assertEqual(r.metadata, metadata())
            self.assertTrue(r._has_stacks)
            self.assertEqual(r._pc_samples, {"k": [(16, 2, 5)]})
            got = (r.lane_infos(), r.aggregate_stats(), r.store.exclusive_aggregate().rows,
                   r.store.activity(r.lane_infos()[0].name)[0]["busy"].tolist())
            self.assertEqual(got, expected)
            self.assertEqual(list(r.store.exclusive_aggregate().rows.items()), mem_rows)
            self.assertEqual(r.lane_infos(), self.mem.lane_infos())
        finally:
            r.close()

    def test_edges_persisted_and_invalidated_by_changes(self):
        path = os.path.join(self.tmp, "edges.hpstore")
        t = build(DiskTraceStore(path), self.events)
        t.finalize()
        cp.analyze(t)
        self.assertIsNotNone(t.store.load_edges(cp.EDGES_VERSION))
        t.close()
        r = trace_io.open_trace(path)
        self.assertIsNotNone(r.store.load_edges(cp.EDGES_VERSION))     # survives reopen
        s = next(iter(r.iter_spans()))
        s.tags["changed"] = "1"
        r.update_span(s)
        self.assertIsNone(r.store.load_edges(cp.EDGES_VERSION))        # stale after a change
        self.assertFalse(r.store.is_finalized())
        r.close()

    def test_multinode_merge_identical(self):
        nodes = lambda: [mn.NodeTrace(self.mem, 0), mn.NodeTrace(self.disk, 5_000, 10)]
        m, _ = mn.merge_traces(nodes())
        d, _ = mn.merge_traces(nodes(), into=trace_io.create_disk_trace(os.path.join(self.tmp, "m.hpstore")))
        try:
            self.assertEqual([span_key(s)[1:] for s in m.spans], [span_key(s)[1:] for s in d.spans])
            self.assertEqual(m.aggregate_stats(), d.aggregate_stats())
            self.assertEqual(mn.validate_causality(m), mn.validate_causality(d))
        finally:
            d.close()


class TestStoreSchemaAndConcurrency(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hprofiler_store_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_schema_version_recorded_and_newer_store_refused(self):
        path = os.path.join(self.tmp, "v.hpstore")
        s = DiskTraceStore(path)
        self.assertEqual(s.schema_version(), SCHEMA_VERSION)
        s.close()
        conn = sqlite3.connect(os.path.join(path, "catalog.sqlite"))
        conn.execute("UPDATE store_info SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION + 1),))
        conn.commit()
        conn.close()
        with self.assertRaises(StoreError):
            DiskTraceStore(path, create=False)

    def test_outdated_derived_tables_are_rebuilt_on_open(self):
        path = os.path.join(self.tmp, "d.hpstore")
        t = build(DiskTraceStore(path), synthetic_events(n=200))
        t.finalize()
        lanes = t.lane_infos()
        t.close()
        conn = sqlite3.connect(os.path.join(path, "catalog.sqlite"))
        conn.execute("UPDATE store_info SET value=? WHERE key='derived_version'", (str(DERIVED_VERSION - 1),))
        conn.execute("DELETE FROM lanes")
        conn.commit()
        conn.close()
        r = trace_io.open_trace(path)
        self.assertTrue(r.store.is_finalized())
        self.assertEqual(r.lane_infos(), lanes)
        r.close()

    def test_json_export_reopens_its_store_only_while_unchanged(self):
        store, js = os.path.join(self.tmp, "run.hpstore"), os.path.join(self.tmp, "run.json")
        t = build(DiskTraceStore(store), synthetic_events(n=300))
        t.finalize()
        trace_io.write_json_export(t, js)
        t.close()
        r = trace_io.open_trace(js)
        self.assertEqual(r.store.kind, "disk")                 # the capture store, not a JSON parse
        self.assertEqual(r.span_count(), len([e for e in synthetic_events(n=300) if isinstance(e, SpanEvent)]))
        r.close()
        Path(js).write_text(Path(js).read_text().replace('"./app"', '"./other"'))
        r = trace_io.open_trace(js)                             # JSON edited since: parse the JSON
        self.assertEqual((r.store.kind, r.metadata.command), ("memory", "./other"))

    def test_interrupted_json_import_is_not_reused(self):
        js = os.path.join(self.tmp, "big.json")
        chrome_trace.write(build(MemoryTraceStore(), synthetic_events(n=300)), js)
        cache = trace_io.import_cache_path(Path(js))
        partial = Trace(store=DiskTraceStore(cache))            # what a killed import leaves behind
        partial.add(SpanEvent("only", Category.CPU, T0, 5, 1, 1))
        partial.close()
        r = trace_io.open_trace(js, disk=True)
        self.assertGreater(r.span_count(), 1)
        r.close()
        r = trace_io.open_trace(js, disk=True)                  # completed import: reused as is
        self.assertEqual(r.store.load_meta("imported_from")["name"], "big.json")
        r.close()

    def test_stack_frame_source_annotation_identical_on_both_stores(self):
        # annotate_stack_frames streams stacked spans in chunks and writes
        # resolved frames back through update_spans().
        import shutil as _sh
        import subprocess
        from src.analysis.addr2line import _find_symbolizer
        from src.analysis.cct import annotate_stack_frames
        if not _sh.which("gcc") or not _sh.which("nm") or _find_symbolizer() is None:
            self.skipTest("gcc, nm or a symbolizer unavailable")
        src, exe = os.path.join(self.tmp, "m.c"), os.path.join(self.tmp, "m")
        Path(src).write_text("int main(void) {\n  return 0;\n}\n")
        subprocess.run(["gcc", "-g", "-O0", "-no-pie", "-o", exe, src], check=True)
        addr = next(line.split()[0] for line in subprocess.run(["nm", exe], capture_output=True, text=True,
                                                                check=True).stdout.splitlines()
                    if line.endswith(" main"))
        frame = f"main|{exe}|0x{int(addr, 16):x}"

        def make(store):
            t = Trace(store=store)
            for i in range(5):
                t.add(SpanEvent("work", Category.CPU, T0 + i, 10, 1, 1, stack_frames=[frame, "x|/no/such|0x1"]))
            t.add(SpanEvent("plain", Category.CPU, T0, 10, 1, 1))
            return t
        results = []
        for store in (MemoryTraceStore(), DiskTraceStore(os.path.join(self.tmp, "annot.hpstore"))):
            t = make(store)
            n = annotate_stack_frames(t)
            results.append((n, [s.stack_frames for s in t.iter_spans()]))
            t.close()
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0][0], 5)
        self.assertIn("(m.c:1)", results[0][1][0][0])
        self.assertEqual(results[0][1][0][1], "x|/no/such|0x1")

    def test_concurrent_appends_from_many_threads(self):
        path = os.path.join(self.tmp, "c.hpstore")
        t = Trace(store=DiskTraceStore(path, batch_size=97))

        def writer(k):
            for i in range(2000):
                t.add(SpanEvent(f"f{k}", Category.CPU, T0 + i * 10, 5, 100 + k % 3, k))
        threads = [threading.Thread(target=writer, args=(k,)) for k in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        t.finalize()
        self.assertEqual(t.span_count(), 16_000)
        self.assertEqual(sorted(s.seq for s in t.iter_spans()), list(range(16_000)))
        self.assertEqual({r["name"]: r["count"] for r in t.aggregate_stats()}, {f"f{k}": 2000 for k in range(8)})
        t.close()


try:
    from PySide6.QtGui import QGuiApplication
    _HAVE_QT = True
except Exception:
    _HAVE_QT = False


@unittest.skipUnless(_HAVE_QT, "PySide6 not installed")
class TestTimelineModelParity(_Pair):
    """The GUI timeline model gives the same answers on either store."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app = QGuiApplication.instance() or QGuiApplication([sys.argv[0]])
        from src.gui.models import TimelineModel
        from src.gui.theme import Theme
        cls.theme = Theme(dark=True)
        cls.tm = TimelineModel(cls.mem, cls.theme)
        cls.td = TimelineModel(cls.disk, cls.theme)

    def strip(self, spans):
        return [{k: v for k, v in d.items() if k != "spanIdx"} for d in spans]

    def test_lanes_rows_views_and_search(self):
        tm, td = self.tm, self.td
        self.assertEqual(tm.lanes, td.lanes)
        self.assertEqual(tm.rows, td.rows)
        self.assertEqual(len(tm.connectors), len(td.connectors))
        a, w = tm.viewStartNs, tm.traceDurationNs
        for lane in range(len(tm.lanes)):
            for frac in (1.0, 0.01, 0.0001):
                vm = tm.laneView(lane, a, a + w * frac, 800, 50)
                vd = td.laneView(lane, a, a + w * frac, 800, 50)
                self.assertEqual(vm["mode"], vd["mode"])
                if vm["mode"] == "spans":
                    self.assertEqual(self.strip(vm["spans"]), self.strip(vd["spans"]))
                else:
                    self.assertEqual(vm["bins"], vd["bins"])
            self.assertEqual(self.strip(tm.visibleSpans(lane, a, a + w, 40)),
                             self.strip(td.visibleSpans(lane, a, a + w, 40)))
        self.assertEqual(tm.groupCoverage(list(range(len(tm.lanes))), a, a + w, 64),
                         td.groupCoverage(list(range(len(td.lanes))), a, a + w, 64))
        self.assertEqual(tm.search("mpi_", False), td.search("mpi_", False))
        self.assertEqual([(m["laneIndex"], m["startNs"]) for m in tm.findByName("mpi", "MPI_Send", 20)],
                         [(m["laneIndex"], m["startNs"]) for m in td.findByName("mpi", "MPI_Send", 20)])
        tm.applyFilters({"nameQuery": "kernel", "minDurationNs": 10_000})
        td.applyFilters({"nameQuery": "kernel", "minDurationNs": 10_000})
        self.assertEqual(tm.rows, td.rows)
        tm.clearFilters()
        td.clearFilters()


if __name__ == "__main__":
    unittest.main()
