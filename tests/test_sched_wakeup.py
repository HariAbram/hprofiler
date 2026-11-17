"""
OS scheduler wakeups (hooks/os_tracer, src/analysis/semantics/sched.py):

  wire      the lines os_tracer forwards for each BPF event
            (hooks/os_tracer/sched_event_fmt.h, built from
            tests/native/sched_event_fmt_test.c under ASan+UBSan), parsed by
            the collector's own parser: the woken task's thread and process,
            the waker's thread, process and comm, the context, the waking's
            time, sanitized comms. The eBPF program itself is never loaded
            here (no CAP_BPF): what the kernel would put in each field is not
            tested;
  analysis  a blocking call whose thread slept inside it and was woken from
            task context by a traced thread gets a "wakeup" edge from the
            waker's innermost span at the waking -- "medium", inferred
            evidence, the call then waiting rather than unknown -- and only
            then. Spin waits, preemptions, an older tracer's records (no
            sleep= / no waker), interrupt or unknown context, another pid
            namespace, an untraced waker, no scheduler data: no edge, each
            counted.

Both pipelines (dict and streamed) build the same edges.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis import cp_model as cm
from src.analysis import criticalpath as cp
from src.analysis import evidence as ev
from src.core.events import Category, SpanEvent
from src.core.runner import _parse_record
from src.core.trace import Trace, TraceMetadata

PID, T_WAIT, T_WAKER = 100, 2, 3


def S(name, cat, start, end, pid=PID, tid=T_WAIT, **tags):
    return SpanEvent(name, Category(cat), start, end - start, pid, tid, tags={k: str(v) for k, v in tags.items()})


def off_cpu(start, end, tid=T_WAIT, **tags):
    return S("off_cpu", "sched", start, end, pid=0, tid=tid, comm="w", **tags)


def woken(start, end, wake_ns, waker_tid=T_WAKER, ctx="task", task_pid=PID, waker_pid=PID, tid=T_WAIT):
    return off_cpu(start, end, tid=tid, task_pid=task_pid, sleep=1, waker_pid=waker_pid, waker_tid=waker_tid,
                   waker_comm="p", ctx=ctx, wake_ns=wake_ns)


def wait_call(start=100, end=500):
    return S("futex_wait", "sync", start, end, type="sync")


def producer():
    return [S("produce", "other", 50, 400, tid=T_WAKER), S("signal", "other", 300, 320, tid=T_WAKER)]


def trace(spans):
    t = Trace(TraceMetadata())
    for s in spans:
        t.add(s)
    return t


def edges(t):
    spans, preds = cp.build_dependency_graph(t)
    full = [(v, u, cp._KIND_CODE[k], cp._CONF_CODE[c]) for v, pl in preds.items() for u, k, c in pl]
    assert sorted(cp.build_edges(t).tolist()) == sorted(full), "pipelines differ"
    return Counter((spans[u].name, spans[v].name, c) for v, pl in preds.items() for u, k, c in pl
                   if k == "wakeup")


def diag(t):
    return cp.dependency_diagnostics(t)


class TestWire(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("gcc"):
            raise unittest.SkipTest("needs gcc")
        cls.tmp = tempfile.mkdtemp()
        exe = os.path.join(cls.tmp, "fmt")
        src = str(ROOT / "tests/native/sched_event_fmt_test.c")
        for flags in (["-fsanitize=address,undefined"], []):
            r = subprocess.run(["gcc", "-O1", "-Wall", "-Wextra", "-Werror", *flags, src, "-o", exe],
                               capture_output=True, text=True)
            if r.returncode == 0:
                break
        else:
            raise unittest.SkipTest(f"cannot build: {r.stderr[:200]}")
        r = subprocess.run([exe], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        cls.lines = r.stdout.splitlines()
        cls.events = [_parse_record(ln) for ln in cls.lines]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    def test_a_sleep_names_its_waker(self):
        e = self.events[0]
        self.assertEqual((e.name, e.category, e.pid, e.tid, e.start_ns, e.duration_ns),
                         ("off_cpu", Category.SCHED, 0, 4242, 1000, 2000000))
        self.assertEqual(e.tags, {"comm": "worker", "task_pid": "4200", "sleep": "1", "waker_pid": "4200",
                                  "waker_tid": "4243", "waker_comm": "a_b_c_d", "ctx": "task",
                                  "wake_ns": "1990000"})

    def test_a_sleep_without_its_waking_and_a_preemption(self):
        self.assertEqual(self.events[1].tags, {"comm": "worker", "task_pid": "4200", "sleep": "1"})
        self.assertEqual(self.events[2].tags, {"comm": "worker", "task_pid": "4200", "sleep": "0"})
        self.assertEqual(self.events[3].tags["sleep"], "0")          # a waker without a sleep: ignored
        self.assertNotIn("waker_tid", self.events[3].tags)

    def test_wakeups_from_each_context(self):
        wk = [e for e in self.events if e is not None and e.name == "wakeup"]
        self.assertEqual([e.tags["ctx"] for e in wk], ["task", "irq", "unknown", "unknown"])
        self.assertEqual({k: wk[0].tags[k] for k in ("wakee_pid", "waker_pid", "waker_tid", "waker_comm", "wake_cpu")},
                         {"wakee_pid": "4200", "waker_pid": "4200", "waker_tid": "4243", "waker_comm": "main",
                          "wake_cpu": "3"})
        self.assertEqual((wk[0].tid, wk[0].pid, wk[0].timestamp_ns), (4242, 0, 6000000))

    def test_a_full_comm_and_what_is_not_emitted(self):
        mig = next(e for e in self.events if e is not None and e.name == "migrate")
        self.assertEqual(mig.tags, {"comm": "0123456789abcdef", "orig_cpu": "1", "dest_cpu": "2"})
        self.assertIn("none kind=99", self.lines)
        self.assertIn("tiny=0", self.lines)

    def test_parsed_lines_feed_the_analysis(self):
        # the formatter's first sleep, on hook spans of the same threads
        t = trace([self.events[0],
                   S("futex_wait", "sync", 500, 2100000, pid=4200, tid=4242, type="sync"),
                   S("produce", "other", 1000000, 2000000, pid=4200, tid=4243)])
        self.assertEqual(edges(t), Counter({("produce", "futex_wait", "medium"): 1}))


class TestWakeupEdges(unittest.TestCase):
    def test_the_waker_span_releases_the_call(self):
        t = trace([wait_call(), woken(150, 450, 310), *producer()])
        self.assertEqual(edges(t), Counter({("signal", "futex_wait", "medium"): 1}))      # innermost
        d = diag(t)
        self.assertEqual((d["sched_blocking_calls"], d["sched_wakeup_edges"]), (1, 1))
        self.assertFalse([k for k in d if k.startswith("sched_") and k not in
                          ("sched_blocking_calls", "sched_wakeup_edges")])
        self.assertEqual(cp.dependency_warnings(d), [])

    def test_in_the_model_inferred_and_waiting(self):
        t = trace([wait_call(), woken(150, 450, 310), *producer()])
        res = cm.analyze_model(t, cp.load_or_build_graph(t), sensitivity=0)
        g = res.graph
        wk = [e for e in range(len(g)) if cm.KINDS[g.kind[e]] == "wakeup"]
        self.assertEqual(len(wk), 1)
        e = wk[0]
        self.assertEqual((g.name(g.src[e] >> 1), g.src[e] & 1, g.name(g.dst[e] >> 1), g.dst[e] & 1),
                         ("signal", 0, "futex_wait", 1))                    # the waker's start -> the call's end
        prov, conf = cm.PROVS[g.prov[e]], cm.CONFS[g.conf[e]]
        self.assertEqual((prov, conf, cm.edge_class(prov, conf)), ("os_wakeup", "medium", "inferred"))
        modes = {cm.MODES[g.mode[x]] for x in range(len(g)) if g.owner[x] >= 0 and g.name(g.owner[x]) == "futex_wait"}
        self.assertEqual(modes, {"wait"})
        es = ev.build_evidence(t)
        rows = [c for c in es.cand if cp.EDGE_KINDS[c["kind"]] == "wakeup"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(ev.kinds_of(int(rows[0]["ev_for"])), ["os_wakeup"])
        self.assertEqual(ev.STATUSES[rows[0]["status"]], "inferred")

    def test_without_the_edge_the_wait_is_unknown(self):
        t = trace([wait_call(), off_cpu(150, 450, task_pid=PID, sleep=1), *producer()])
        res = cm.analyze_model(t, cp.load_or_build_graph(t), sensitivity=0)
        g = res.graph
        modes = {cm.MODES[g.mode[x]] for x in range(len(g)) if g.owner[x] >= 0 and g.name(g.owner[x]) == "futex_wait"}
        self.assertEqual(modes, {"unknown"})

    def test_the_last_sleep_in_the_call_decides(self):
        t = trace([wait_call(), woken(120, 140, 60), woken(150, 450, 310), *producer(),
                   S("early", "other", 55, 70, tid=4)])
        # first sleep's waker (tid 3 at 60, in "produce") is not the releaser
        self.assertEqual(edges(t), Counter({("signal", "futex_wait", "medium"): 1}))
        t = trace([wait_call(), woken(120, 140, 310), woken(150, 450, 60, waker_tid=4), *producer(),
                   S("early", "other", 55, 70, tid=4)])
        self.assertEqual(edges(t), Counter({("early", "futex_wait", "medium"): 1}))

    def fallback(self, spans, reason):
        t = trace([wait_call(), *spans, *producer()])
        self.assertEqual(edges(t), Counter())
        d = diag(t)
        self.assertEqual((d.get(reason), d["sched_wakeup_edges"]), (1, 0), d)
        w = cp.dependency_warnings(d)
        self.assertEqual(len(w), 1)
        self.assertIn("0 of 1 blocking call(s) got an OS wakeup edge", w[0])
        return d

    def test_a_spin_wait_gets_none(self):
        self.fallback([off_cpu(10, 40, task_pid=PID, sleep=1)], "sched_no_sleep")   # a sleep before the call

    def test_a_preemption_is_no_sleep(self):
        self.fallback([off_cpu(150, 450, task_pid=PID, sleep=0)], "sched_no_sleep")

    def test_an_older_tracer_cannot_tell_a_sleep(self):
        self.fallback([off_cpu(150, 450)], "sched_sleep_unflagged")

    def test_a_sleep_without_its_waking(self):
        self.fallback([off_cpu(150, 450, task_pid=PID, sleep=1)], "sched_waker_unrecorded")

    def test_interrupt_and_unknown_context(self):
        self.fallback([woken(150, 450, 310, ctx="irq")], "sched_waker_not_task")
        self.fallback([woken(150, 450, 310, ctx="unknown")], "sched_waker_not_task")

    def test_another_pid_namespace(self):
        self.fallback([woken(150, 450, 310, task_pid=7777)], "sched_pid_mismatch")

    def test_an_untraced_waker(self):
        self.fallback([woken(150, 450, 310, waker_tid=99)], "sched_waker_untraced")
        self.fallback([woken(150, 450, 310, waker_pid=55)], "sched_waker_untraced")
        self.fallback([woken(150, 450, 30)], "sched_waker_untraced")            # before its spans

    def test_a_thread_the_tracer_did_not_see(self):
        self.fallback([woken(150, 450, 310, tid=9)], "sched_threads_unseen")

    def test_no_scheduler_data_changes_nothing(self):
        t = trace([wait_call(), *producer()])
        self.assertEqual(edges(t), Counter())
        self.assertFalse([k for k in diag(t) if k.startswith("sched_")])

    def test_non_blocking_calls_and_device_spans_get_none(self):
        t = trace([S("compute", "cpu", 100, 500), woken(150, 450, 310), *producer(),
                   S("kern", "sync", 100, 500, tid=T_WAIT, type="sync", side="gpu")])
        self.assertEqual(edges(t), Counter())
        self.assertNotIn("sched_blocking_calls", diag(t))


if __name__ == "__main__":
    unittest.main()
