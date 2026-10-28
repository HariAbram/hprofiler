"""
MPI semantics in the dependency graph (criticalpath.py's matcher and
collective edges), on synthetic spans in the hook's wire format:

  * a persistent request's MPI_Start is a post: the receiving start's
    completion pairs with the sending start;
  * rooted collectives with their root recorded (wroot=) follow the data
    flow: a bcast's members wait for the root only, a reduce's root for the
    last arriver, its other members for nobody (released by their own
    arrival -- "wait", never "unknown"); without wroot= the rendezvous
    stays, and is flagged;
  * non-blocking collectives: the last poster's call gates each member's
    completing MPI_Wait; a member nobody gates is released by its own post;
  * both pipelines (dict and streamed) build the same edges.
"""
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import cp_model as cm
from src.analysis import criticalpath as cp
from tests.test_cp_model import _trace, mpi


def edges(t, kinds=("p2p", "arrival")):
    spans, preds = cp.build_dependency_graph(t)
    out = Counter()
    for v, pl in preds.items():
        for u, k, _c in pl:
            if k in kinds:
                out[(k, spans[u].name, spans[u].tags.get("rank"), spans[v].name, spans[v].tags.get("rank"))] += 1
    streamed = cp.build_edges(t)
    full = [(v, u, cp._KIND_CODE[k], cp._CONF_CODE[c]) for v, pl in preds.items() for u, k, c in pl]
    assert sorted(streamed.tolist()) == sorted(full), "pipelines differ"
    return out


def coll(name, rank, start, end, typ, **tags):
    return mpi(name, rank + 1, start, end, type=typ, rank=rank, commid=0, **tags)


def modes(t, name, rank):
    res = cm.analyze_model(t, cp.load_or_build_graph(t), sensitivity=0)
    g = res.graph
    return {cm.MODES[g.mode[e]] for e in range(len(g))
            if g.owner[e] >= 0 and g.name(g.owner[e]) == name and g.cat[g.owner[e]] == "mpi"
            and t.store.span_by_seq(int(cp.load_or_build_graph(t).seqs[g.owner[e]])).tags.get("rank") == str(rank)}


class TestPersistentRequests(unittest.TestCase):
    def test_starts_pair_like_isend_and_irecv(self):
        t = _trace([
            mpi("MPI_Start", 1, 100, 110, sid="s1", type="start", rank=0, ptype="send", peer=1, wpeer=1, tag=7,
                commid=0),
            mpi("MPI_Wait", 1, 120, 130, psid="s1", type="wait", rank=0),
            mpi("MPI_Start", 2, 0, 5, sid="r1", type="start", rank=1, ptype="recv", peer=0, wpeer=0, tag=7,
                commid=0),
            mpi("MPI_Wait", 2, 10, 300, psid="r1", type="wait", rank=1),
        ])
        self.assertEqual(edges(t), Counter({("p2p", "MPI_Start", "0", "MPI_Wait", "1"): 1}))
        self.assertEqual(modes(t, "MPI_Wait", 1), {"wait"})


class TestRootedCollectives(unittest.TestCase):
    def test_bcast_members_wait_for_the_root_only(self):
        t = _trace([coll("MPI_Bcast", r, s, 500, "bcast", root=1, wroot=1)
                    for r, s in ((0, 0), (1, 100), (2, 400))])
        self.assertEqual(edges(t), Counter({("arrival", "MPI_Bcast", "1", "MPI_Bcast", "0"): 1,
                                            ("arrival", "MPI_Bcast", "1", "MPI_Bcast", "2"): 1}))
        res = cm.analyze_model(t, cp.load_or_build_graph(t), sensitivity=0)
        self.assertEqual(res.graph_evidence["rooted_collective_edges"], 0)
        self.assertEqual(res.graph_evidence["by_provenance"], {"rooted_flow": 2})

    def test_reduce_root_waits_for_the_last_arriver_and_the_others_for_nobody(self):
        t = _trace([coll("MPI_Reduce", r, s, 500, "reduce", root=0, wroot=0)
                    for r, s in ((0, 0), (1, 100), (2, 400))])
        self.assertEqual(edges(t), Counter({("arrival", "MPI_Reduce", "2", "MPI_Reduce", "0"): 1}))
        res = cm.analyze_model(t, cp.load_or_build_graph(t), sensitivity=0)
        self.assertEqual(res.unmatched_by_category, {})          # rank 1 waited for nobody: "wait"

    def test_without_the_root_the_rendezvous_stays_and_is_flagged(self):
        t = _trace([coll("MPI_Bcast", r, s, 500, "bcast") for r, s in ((0, 0), (1, 100), (2, 400))])
        self.assertEqual(edges(t), Counter({("arrival", "MPI_Bcast", "2", "MPI_Bcast", "0"): 1,
                                            ("arrival", "MPI_Bcast", "2", "MPI_Bcast", "1"): 1}))
        res = cm.analyze_model(t, cp.load_or_build_graph(t), sensitivity=0)
        self.assertEqual(res.graph_evidence["rooted_collective_edges"], 2)


class TestNonblockingCollectives(unittest.TestCase):
    def trace(self, root=None):
        spans = []
        for r, post in ((0, 0), (1, 50), (2, 300)):
            tags = {"root": root, "wroot": root} if root is not None else {}
            typ = "ibcast" if root is not None else "iallreduce"
            name = "MPI_Ibcast" if root is not None else "MPI_Iallreduce"
            spans.append(mpi(name, r + 1, post, post + 5, sid=f"c{r}", type=typ, rank=r, commid=0, **tags))
            spans.append(mpi("MPI_Wait", r + 1, post + 10, 600, psid=f"c{r}", type="wait", rank=r))
        return _trace(spans)

    def test_the_last_poster_gates_every_completion(self):
        self.assertEqual(edges(self.trace()), Counter({
            ("arrival", "MPI_Iallreduce", "2", "MPI_Wait", "0"): 1,
            ("arrival", "MPI_Iallreduce", "2", "MPI_Wait", "1"): 1,
            ("arrival", "MPI_Iallreduce", "2", "MPI_Wait", "2"): 1}))      # its own: released by itself
        t = self.trace()
        res = cm.analyze_model(t, cp.load_or_build_graph(t), sensitivity=0)
        self.assertEqual(res.unmatched_by_category, {})

    def test_a_rooted_one_waits_for_its_root(self):
        self.assertEqual(edges(self.trace(root=1)), Counter({
            ("arrival", "MPI_Ibcast", "1", "MPI_Wait", "0"): 1,
            ("arrival", "MPI_Ibcast", "1", "MPI_Wait", "2"): 1,
            ("arrival", "MPI_Ibcast", "1", "MPI_Wait", "1"): 1}))

    def test_completed_by_test_only_gets_no_edge(self):
        spans = [mpi("MPI_Iallreduce", 1, 0, 5, sid="c0", type="iallreduce", rank=0, commid=0),
                 mpi("MPI_Wait", 1, 10, 600, psid="c0", type="wait", rank=0),
                 mpi("MPI_Iallreduce", 2, 300, 305, sid="c1", type="iallreduce", rank=1, commid=0)]
        self.assertEqual(edges(_trace(spans)), Counter({("arrival", "MPI_Iallreduce", "1", "MPI_Wait", "0"): 1}))


if __name__ == "__main__":
    unittest.main()
