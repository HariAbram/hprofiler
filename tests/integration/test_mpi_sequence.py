"""
The MPI hook's sequence numbers and matched-probe links on 4 real ranks
(tests/fixtures/mpi_seq.c, see its header for the phases):

  * the application's output under the hook equals the unprofiled run;
  * sseq= / rseq= / cseq= are contiguous from 1 per key -- also for the
    two threads that post to one destination concurrently -- and every
    rank numbers each communicator's collectives alike (same type at each
    number);
  * every pairing in the dependency graph agrees with the values the
    receives got (each value names its send): exact, tag-wildcard and full
    wildcard receives, three communicators, two threads per side, and the
    contested channel (two threads receiving one tag), where a pairing
    either is right or is marked "low";
  * matched probes: each MPI_Mprobe names its message (sid=) and its
    MPI_Mrecv links back (psid=) although the library reused the message
    handle; MPI_Improbe + MPI_Imrecv + MPI_Wait; MPI_MESSAGE_NO_PROC is
    null=1; a message probed through PMPI_Mprobe is mid_unknown=1 and
    counted (mpi_msg_untracked), and the analysis reports it;
  * collectives group by (commid, cseq) with every member ("high");
  * a dropped record: one send record removed from the real trace -- its
    receive gets no edge, every other pairing is unchanged, the gap is
    reported.

Needs Intel MPI or another launcher that starts 4 real ranks (see
test_mpi_multirank.py) and a hook built from this source (the trace must
carry sseq=); skips otherwise.
"""
import re
import shutil
import sys
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tests.integration import test_mpi_multirank as mr     # the module, not its TestCases


def _edges_into(trace):
    from src.analysis import criticalpath as cp
    spans, preds = cp.build_dependency_graph(trace)
    return spans, {v: [(spans[u], c) for u, k, c in pl if k == "p2p"] for v, pl in preds.items()}


class TestMPISequence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mr._build_and_profile(cls, "mpi_seq")
        cls.mpi = list(cls.trace.iter_spans(categories=("mpi",)))
        if not any("sseq" in s.tags for s in cls.mpi):
            raise unittest.SkipTest("the MPI hook predates sseq= -- point HPROFILER_HOOKS_DIR at a fresh build")
        cls.rank0 = sorted((s for s in cls.mpi if s.tags.get("rank") == "0"), key=lambda s: s.start_ns)
        cls.out = cls.prof_out.splitlines()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    def lines(self, prefix):
        return [dict(kv.split("=") for kv in ln.split()[1:] if "=" in kv)
                for ln in self.out if ln.startswith(prefix + " ")]

    def test_output_equals_the_unprofiled_run(self):
        def stable(text):           # thread ids and the wildcards' order differ between runs
            lines = [re.sub(r"tid=\d+ ", "", ln) for ln in text.splitlines()
                     if ln[:2] in ("A ", "C2", "D ", "E ") and "handle=" not in ln]
            fixed = sorted(ln for ln in lines if "src=any" not in ln)
            anyof = sorted(ln.split("value=")[1] for ln in lines if "src=any" in ln)
            return fixed, anyof
        self.assertEqual(stable(self.prof_out), stable(self.base_out))

    # ── the numbers ─────────────────────────────────────────────────────
    def test_numbers_are_contiguous_per_key(self):
        sends, recvs, colls = defaultdict(list), defaultdict(list), defaultdict(list)
        for s in self.mpi:
            t = s.tags
            if "sseq" in t:
                sends[(s.pid, t["commid"], t["peer"])].append(int(t["sseq"]))
            if "rseq" in t and t.get("via") != "mprobe":
                recvs[(s.pid, t["commid"])].append(int(t["rseq"]))
            if "cseq" in t:
                colls[(s.pid, t["commid"])].append(int(t["cseq"]))
        for i in self.trace.iter_instants():          # MPI_Improbe that found a message
            if i.name == "MPI_Improbe" and "rseq" in i.tags:
                recvs[(i.pid, i.tags["commid"])].append(int(i.tags["rseq"]))
        for table in (sends, colls):
            for key, nums in table.items():
                self.assertEqual(sorted(nums), list(range(1, len(nums) + 1)), key)
        for key, nums in recvs.items():               # an Imrecv repeats its Improbe's number
            self.assertEqual(sorted(set(nums)), list(range(1, len(set(nums)) + 1)), key)
        # rank 1's two threads shared one (comm, destination) counter
        concurrent = [k for k, v in sends.items() if len(v) >= 40]
        self.assertTrue(concurrent, "the two sender threads' communicator")

    def test_every_rank_numbers_a_communicators_collectives_alike(self):
        by = defaultdict(dict)
        for s in self.mpi:
            if "cseq" in s.tags and not s.tags.get("type", "").endswith("_init"):
                t = s.tags.get("ctype") or s.tags["type"]
                by[(s.tags["commid"], s.tags["cseq"])][s.tags["rank"]] = t
        for key, members in by.items():
            self.assertEqual(len(set(members.values())), 1, (key, members))

    # ── the pairing against what the receives got ───────────────────────
    def test_pairings_agree_with_the_values_received(self):
        spans, into = _edges_into(self.trace)
        idx = {id(s): i for i, s in enumerate(spans)}

        def src_of(span):
            e = into.get(idx[id(span)], [])
            self.assertEqual(len(e), 1, (span.name, span.tags))
            return e[0]

        def value_of(send):
            return 1000 * int(send.tags["rank"]) + int(send.tags["sseq"])

        # A: rank 0's receive completions before its first barrier, in call order
        first_barrier = next(s.start_ns for s in self.rank0 if s.name == "MPI_Barrier")
        world = [s for s in self.rank0 if s.start_ns < first_barrier and s.name in ("MPI_Recv", "MPI_Wait")]
        a = self.lines("A")
        for k, rec in enumerate(a[:9]):
            send, conf = src_of(world[k])
            self.assertEqual(value_of(send), int(rec["value"]), (k, world[k].tags))
            self.assertIn(conf, ("medium", "high"))
        waitall = next(s for s in self.rank0 if s.name == "MPI_Waitall")
        got = sorted(value_of(s) for s, _c in into[idx[id(waitall)]])
        self.assertEqual(got, sorted(int(r["value"]) for r in a[9:]))

    def test_two_threads_per_side_pair_exactly(self):
        spans, into = _edges_into(self.trace)
        idx = {id(s): i for i, s in enumerate(spans)}
        rows = self.lines("C2")
        if not rows:
            self.skipTest("no MPI_THREAD_MULTIPLE")
        for tag in ("100", "101"):
            recvs = sorted((s for s in self.rank0 if s.name == "MPI_Recv" and s.tags.get("tag") == tag),
                           key=lambda s: s.start_ns)
            want = [int(r["value"]) for r in sorted((r for r in rows if r["tag"] == tag), key=lambda r: int(r["k"]))]
            self.assertEqual(len(recvs), 20)
            for r, v in zip(recvs, want):
                (send, conf), = into[idx[id(r)]]
                # the sender's two threads interleave on one counter: the
                # channel order is the sseq order of that tag's sends
                self.assertNotEqual(conf, "low")
                self.assertEqual(send.tags["tag"], tag)
            sends = sorted((s for s in self.mpi if s.tags.get("rank") == "1" and s.tags.get("tag") == tag
                            and s.name == "MPI_Send"), key=lambda s: int(s.tags["sseq"]))
            self.assertEqual([into[idx[id(r)]][0][0] for r in recvs], sends)

    def test_the_contested_channel_is_right_or_low(self):
        spans, into = _edges_into(self.trace)
        idx = {id(s): i for i, s in enumerate(spans)}
        rows = self.lines("C3")
        if not rows:
            self.skipTest("no MPI_THREAD_MULTIPLE")
        by_tid = defaultdict(list)
        for s in self.rank0:
            if s.name == "MPI_Recv" and s.tags.get("tag") == "102":
                by_tid[str(s.tid)].append(s)
        for r in rows:
            span = sorted(by_tid[r["tid"]], key=lambda s: s.start_ns)[int(r["k"])]
            (send, conf), = into[idx[id(span)]]
            if conf != "low":
                self.assertEqual(5000 + int(send.tags["sseq"]), int(r["value"]), r)

    # ── matched probes ──────────────────────────────────────────────────
    def test_matched_probes_link_across_handle_reuse(self):
        probes = [s for s in self.rank0 if s.name == "MPI_Mprobe" and s.tags.get("flag") == "1"
                  and s.tags.get("null") != "1"]
        mrecvs = [s for s in self.rank0 if s.name == "MPI_Mrecv" and s.tags.get("via") == "mprobe"]
        self.assertEqual(len(probes), 5)
        self.assertEqual(len({p.span_id for p in probes}), 5)          # one id per message
        self.assertEqual([m.parent_span_id for m in mrecvs], [p.span_id for p in probes])
        handles = [r["handle"] for r in self.lines("D") if "handle" in r]
        if len(set(handles)) == len(handles):
            print("note: the MPI library did not reuse a message handle in this run", file=sys.stderr)
        spans, into = _edges_into(self.trace)
        idx = {id(s): i for i, s in enumerate(spans)}
        mrecv_vals = [int(ln.split("value=")[1]) for ln in self.out if ln.startswith("D mrecv")]
        self.assertEqual(len(mrecv_vals), 5)
        for p, v in zip(probes, mrecv_vals):
            (send, conf), = into[idx[id(p)]]
            self.assertEqual(1000 * int(send.tags["rank"]) + int(send.tags["sseq"]), v)

    def test_improbe_imrecv_wait(self):
        found = [i for i in self.trace.iter_instants() if i.name == "MPI_Improbe" and i.tags.get("flag") == "1"]
        imrecvs = [s for s in self.rank0 if s.name == "MPI_Imrecv"]
        self.assertEqual(len(imrecvs), 3)
        self.assertEqual([m.parent_span_id for m in imrecvs], [i.tags["mid"] for i in found])
        spans, into = _edges_into(self.trace)
        idx = {id(s): i for i, s in enumerate(spans)}
        waits = [s for s in self.rank0 if s.name == "MPI_Wait" and s.parent_span_id in {m.span_id for m in imrecvs}]
        got = [1000 * int(src.tags["rank"]) + int(src.tags["sseq"]) for w in waits for src, _c in into[idx[id(w)]]]
        want = [int(ln.split("value=")[1]) for ln in self.out if ln.startswith("D imrecv")]
        self.assertEqual(got, want)

    def test_no_proc_and_an_unseen_probe(self):
        nulls = [s for s in self.rank0 if s.name in ("MPI_Mprobe", "MPI_Mrecv") and s.tags.get("null") == "1"]
        self.assertEqual(sorted(s.name for s in nulls), ["MPI_Mprobe", "MPI_Mrecv"])
        unknown = [s for s in self.rank0 if s.name == "MPI_Mrecv" and s.tags.get("mid_unknown") == "1"]
        self.assertEqual(len(unknown), 1)
        untracked = {c.pid: c.value for c in self.trace.iter_counters() if c.name == "mpi_msg_untracked"}
        self.assertEqual(untracked.get(self.rank0[0].pid), 1)
        diag = self.cp.mpi_dependency_diagnostics(self.trace)
        self.assertEqual(diag.get("matched_receive_unlinked"), 1)

    # ── collectives ─────────────────────────────────────────────────────
    def test_collectives_group_by_number_with_every_member(self):
        groups = [(m, c) for m, c in self.cp._rendezvous_groups(
            "mpi", [it for it in (self.cp._rendezvous_item(s, i, "mpi") for i, s in enumerate(self.spans))
                    if it is not None])]
        self.assertTrue(groups)
        lifecycle = []
        for members, conf in groups:
            names = [self.spans[i].name for i in members]
            if self.spans[members[0]].tags.get("type") in ("init", "finalize"):
                lifecycle.append((sorted(set(names)), len(members), conf))     # inferred: "medium"
                continue
            self.assertEqual(conf, "high", names)
        if lifecycle:          # a hook from before the MPI_Init / MPI_Finalize spans has none
            self.assertEqual(sorted(lifecycle), [(["MPI_Finalize"], 4, "medium"), (["MPI_Init_thread"], 4, "medium")])
        diag = self.cp.mpi_dependency_diagnostics(self.trace)
        for k in ("collective_seq_gaps", "collectives_incomplete", "collective_seq_misaligned",
                  "send_seq_gaps", "recv_seq_gaps"):
            self.assertNotIn(k, diag)

    # ── a dropped record ─────────────────────────────────────────────────
    def test_a_dropped_send_record_leaves_every_other_pairing_alone(self):
        from src.core.trace import Trace, TraceMetadata
        victim = next(s for s in self.mpi if s.name == "MPI_Send" and s.tags.get("rank") == "2"
                      and s.tags.get("commid") == "0" and s.tags.get("sseq") == "2")
        cut = Trace(TraceMetadata())
        for s in self.trace.iter_spans():
            if not (s.pid == victim.pid and s.start_ns == victim.start_ns and s.name == victim.name):
                cut.add(s)
        for i in self.trace.iter_instants():
            cut.add(i)
        for c in self.trace.iter_counters():
            cut.add(c)

        def pairing(trace):
            spans, into = _edges_into(trace)
            return {(d.pid, d.start_ns, d.name): (s.pid, s.start_ns, c)
                    for v, lst in into.items() for s, c in lst for d in (spans[v],)}
        before, after = pairing(self.trace), pairing(cut)
        lost = [k for k, (sp, st, _c) in before.items() if (sp, st) == (victim.pid, victim.start_ns)]
        self.assertEqual(len(lost), 1)
        for k, v in before.items():
            if k in lost:
                self.assertNotIn(k, after)
            else:
                self.assertEqual(after.get(k, (None, None, None))[:2], v[:2], k)
        diag = self.cp.mpi_dependency_diagnostics(cut)
        self.assertEqual((diag.get("send_seq_gaps"), diag.get("seq_gaps_placed"), diag.get("send_record_lost")),
                         (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
