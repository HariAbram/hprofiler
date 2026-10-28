"""
Calibration harness (src/analysis/edge_calibration.py) on simulated MPI
programs whose true dependencies are known -- a demonstration of the
method on synthetic data, not a calibration of hprofiler on real programs.

Each simulated run: 4 ranks exchange messages on a few channels (sender,
receiver, tag), MPI's non-overtaking order deciding the true pairs; some
runs lose the record of one send (the program still sent it, the trace
did not keep it) and flag that process's capture as lossy, as a dropping
transport would. The builders' edges and cp_alternatives' candidates are
labelled by the true pairs; a calibration is fitted on some seeds and
judged on others (ECE, Brier score, log loss), against the constant
predictor of the base rate.

Run directly (tests/validation is not part of the main discovery):
    python3 -m unittest discover -s tests/validation -p "test_*.py" -v
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.analysis import cp_alternatives as alt
from src.analysis import cp_model as cm
from src.analysis import criticalpath as cp
from src.analysis import edge_calibration as ec
from src.core.events import Category, SpanEvent
from src.core.trace import Trace, TraceMetadata


def simulate(seed: int, lose: bool) -> tuple[Trace, set[tuple[int, int, str]]]:
    """A run and its true point-to-point dependencies (send ordinal,
    receive ordinal, "p2p") among the recorded spans."""
    rng = random.Random(seed)
    channels = [(s, d, tag) for s in range(4) for d in range(4) if s != d for tag in (1, 2)
                if rng.random() < 0.4]
    events = []                       # (span, label)
    recv_clock = [0] * 4              # a rank posts its next blocking receive once the last returned
    for (src, dst, tag) in channels:
        n = rng.randrange(2, 5)
        t = rng.randrange(0, 1000)
        for k in range(n):
            t += rng.randrange(50, 400)
            send = SpanEvent("MPI_Send", Category.MPI, t, 10, src + 1, src + 1,
                             tags={"type": "send", "rank": str(src), "peer": str(dst), "wpeer": str(dst),
                                   "tag": str(tag), "commid": "0"})
            post = max(recv_clock[dst], t + rng.choice([-300, -50, 0, 40]))
            end = max(t + 30, post + 5)
            recv_clock[dst] = end
            recv = SpanEvent("MPI_Recv", Category.MPI, post, end - post, dst + 1, dst + 1,
                             tags={"type": "recv", "rank": str(dst), "peer": str(src), "wpeer": str(src),
                                   "tag": str(tag), "commid": "0"})
            events.append((send, ("s", src, dst, tag, k)))
            events.append((recv, ("r", src, dst, tag, k)))
    lost = None
    lossy_pids = set()
    if lose and channels:
        src, dst, tag = rng.choice(channels)
        lost = ("s", src, dst, tag, 0)
        lossy_pids.add(src + 1)
    md = TraceMetadata()
    if lossy_pids:
        md.capture_health = {"transport": {f"{p}/mpi": {"dropped_full": 1} for p in lossy_pids}}
    tr = Trace(md)
    kept = [(s, lab) for s, lab in events if lab != lost]
    rng.shuffle(kept)
    for s, _lab in kept:
        tr.add(s)
    # the store hands back new objects: find each by what identifies it
    spans = list(tr.iter_spans())
    by_key = {}
    for i, s in enumerate(spans):
        key = ("s" if s.name == "MPI_Send" else "r", s.start_ns, s.pid, s.tags["tag"], s.tags["peer"])
        by_key.setdefault(key, i)
    idx = {}
    for s, lab in kept:
        key = ("s" if s.name == "MPI_Send" else "r", s.start_ns, s.pid, s.tags["tag"], s.tags["peer"])
        idx[lab] = by_key[key]
    truth = set()
    for lab, i in idx.items():
        if lab[0] == "s" and ("r",) + lab[1:] in idx:
            truth.add((i, idx[("r",) + lab[1:]], "p2p"))
    return tr, truth


def examples(seeds) -> list[ec.Labeled]:
    out = []
    for seed in seeds:
        tr, truth = simulate(seed, lose=seed % 2 == 1)
        graph = cp.load_or_build_graph(tr)
        res = cm.analyze_model(tr, graph, sensitivity=0)
        a = alt.analyze_alternatives(tr, graph, res, samples=0, one_change=0)
        out += [e for e in ec.labeled_examples(a.evidence, truth) if e.kind == "p2p"]
    return out


class TestCalibrationHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = examples(range(0, 40))
        cls.test = examples(range(40, 60))

    def test_the_fit_beats_the_base_rate_on_held_out_runs(self):
        cal = ec.fit(self.train, dataset="simulated MPI channels (seeds 0-39)", min_examples=20)
        self.assertIn("p2p/posting_order", cal.strata)
        q = ec.evaluate(cal, self.test)
        base = sum(e.label for e in self.train) / len(self.train)
        q0 = ec.brier([base] * q["quantified"],
                      [e.label for e in self.test if cal.probability(e.kind, e.provenance, e.score) is not None])
        print(f"\n  held out: {q['quantified']} quantified, {q['unquantified']} not; ECE {q['ece']:.3f}, "
              f"Brier {q['brier']:.3f} (base rate {q0:.3f}), log loss {q['log_loss']:.3f}")
        self.assertGreater(q["quantified"], 0)
        self.assertLess(q["brier"], q0)

    def test_more_evidence_is_never_less_probable(self):
        cal = ec.fit(self.train, dataset="simulated", min_examples=20)
        for st in cal.strata.values():
            self.assertEqual(st["p"], sorted(st["p"]))

    def test_complete_channels_are_right_and_lossy_ones_are_not_always(self):
        # inferred (complete channel) vs ambiguous (a lost send made alternatives)
        inferred = [e.label for e in self.train if e.score // 100 == ec.STATUS_ORDER["inferred"]]
        ambiguous = [e.label for e in self.train if e.score // 100 == ec.STATUS_ORDER["ambiguous"]]
        self.assertTrue(inferred and all(inferred))
        self.assertTrue(ambiguous)
        self.assertLess(sum(ambiguous) / len(ambiguous), 1.0)


if __name__ == "__main__":
    unittest.main()
