"""
Repeated-trial profiling-accuracy report (not a unittest -- timing noise
makes these measurements, not pass/fail checks; the pass/fail tolerances
live in test_profiling_accuracy.py). Usage:

    python3 tests/integration/accuracy_report.py [--trials 10]

For each ground-truth fixture (tests/fixtures/*_truth.c), runs it N times
unprofiled and N times through the real Runner, then reports:
  - overhead: change in the program's OWN measured elapsed time (first to
    last logged stamp) when profiled, i.e. perturbation of the workload
    itself, excluding hprofiler's startup/teardown;
  - timestamp error: profiled span start/end vs the program's own stamps;
  - missing-event rate: expected vs captured spans;
  - aggregation error: summed measured duration vs summed truth duration;
  - behavior change: truth-measured wait time with vs without profiling;
  - ordering: per-thread event-order inversions vs the program's order.
Measured values come from the hooks; everything here is derived from
those measurements plus the program's own clock.
"""
from __future__ import annotations

import argparse
import collections
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
FIX = REPO / "tests" / "fixtures"

from src.core.runner import Runner  # noqa: E402


def truth_rows(path):
    rows = collections.defaultdict(list)
    for line in open(path):
        p = line.split()
        rows[p[0]].append([int(x) for x in p[1:] if x.lstrip("-").isdigit()])
    return rows


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


# name -> (build cmd, backends, [(span name, truth key, expected count per run)], elapsed(truth))
def fixtures(tmp):
    omp = str(FIX / "omp_truth.c")
    reg_elapsed = lambda t: t["region"][-1][2] - t["region"][0][1]
    return {
        "openmp-llvm (OMPT)": (["clang", "-O2", "-fopenmp=libomp", omp, "-o"], ["openmp"],
                               [("omp_barrier_explicit", "barrier", 24)], reg_elapsed),
        "openmp-gnu (GOMP)": (["gcc", "-O2", "-fopenmp", omp, "-o"], ["openmp"],
                              [("omp_barrier", "barrier", 24)], reg_elapsed),
        "mpi (self)": (["mpicc", "-O2", str(FIX / "mpi_truth.c"), "-o"], ["mpi"],
                       [("MPI_Barrier", "barrier", 8), ("MPI_Waitall", "waitall", 8),
                        ("MPI_Allreduce", "allreduce", 8)],
                       lambda t: t["allreduce"][-1][2] - t["barrier"][0][1]),
        "opencl (CPU)": (["gcc", "-O2", str(FIX / "ocl_truth.c"), "-o", None, "-lOpenCL"], ["opencl"],
                         [],
                         lambda t: t["read_nonblocking_plus_finish"][-1][2] - t["write_blocking"][0][1]),
    }


def match(spans, rows, tid_col):
    """Pair each truth row [idx..., (tid), start, end] with the nearest span
    (same tid when the row has one). Returns (lags, end_leads, dur_errs, missing)."""
    lags, leads, derr, missing = [], [], [], 0
    used = set()
    for row in rows:
        a, b = row[-2], row[-1]
        tid = row[tid_col] if tid_col is not None else None
        cands = [s for s in spans if id(s) not in used and (tid is None or s.tid == tid)
                 and a - 5_000_000 <= s.start_ns <= b]
        if not cands:
            missing += 1
            continue
        s = min(cands, key=lambda s: abs(s.start_ns - a))
        used.add(id(s))
        lags.append(s.start_ns - a)
        leads.append(b - s.end_ns)
        derr.append(s.duration_ns - (b - a))
    return lags, leads, derr, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="hprofiler_accreport_")
    try:
        for name, (cmd, backends, checks, elapsed_fn) in fixtures(tmp).items():
            exe = os.path.join(tmp, name.split()[0])
            cmd = [exe if c is None else c for c in cmd]
            if "-o" in cmd and cmd[cmd.index("-o") + 1:cmd.index("-o") + 2] == []:
                cmd.append(exe)
            elif cmd[-1] == "-o":
                cmd.append(exe)
            if not shutil.which(cmd[0]) or subprocess.run(cmd, capture_output=True).returncode != 0:
                print(f"\n## {name}: SKIPPED (toolchain unavailable)")
                continue
            base_el, prof_el, base_wait, prof_wait = [], [], [], []
            lags, leads, derrs, agg = [], [], [], []
            missing = expected = inversions = pairs = 0
            for i in range(args.trials):
                tp = os.path.join(tmp, f"base{i}.truth")
                subprocess.run([exe], env={**os.environ, "TRUTH_OUT": tp}, capture_output=True, check=True)
                t = truth_rows(tp)
                base_el.append(elapsed_fn(t))
                if "barrier" in t and name.startswith("openmp"):
                    base_wait.append(sum(r[-1] - r[-2] for r in t["barrier"]))
                tp = os.path.join(tmp, f"prof{i}.truth")
                tr = Runner(command=[exe], backends=backends, env_extra={"TRUTH_OUT": tp}).run()
                t = truth_rows(tp)
                prof_el.append(elapsed_fn(t))
                spans = tr.spans
                if "barrier" in t and name.startswith("openmp"):
                    prof_wait.append(sum(r[-1] - r[-2] for r in t["barrier"]))
                for span_name, key, n in checks:
                    ss = [s for s in spans if s.name == span_name]
                    expected += n
                    missing += max(0, n - len(ss))
                    tid_col = 2 if name.startswith("openmp") else None
                    la, le, de, miss = match(ss, t[key], tid_col)
                    lags += la; leads += le; derrs += de
                    truth_sum = sum(r[-1] - r[-2] for r in t[key])
                    diff = sum(s.duration_ns for s in ss) - truth_sum
                    agg.append((diff / truth_sum * 100, diff / max(len(ss), 1), truth_sum / max(len(t[key]), 1)))
                    # ordering: per tid, profiled start order vs truth order
                    by_tid = collections.defaultdict(list)
                    for s in ss:
                        by_tid[s.tid].append(s.start_ns)
                    for v in by_tid.values():
                        pairs += max(0, len(v) - 1)
                        inversions += sum(1 for x, y in zip(v, v[1:]) if y < x)
                if name.startswith("opencl"):
                    k = sorted((s for s in spans if s.tags.get("side") == "gpu" and s.tags.get("type") == "kernel"),
                               key=lambda s: s.start_ns)
                    expected += len(t["kernel_devdur"])
                    missing += max(0, len(t["kernel_devdur"]) - len(k))
                    derrs += [s.duration_ns - r[-1] for s, r in zip(k, t["kernel_devdur"])]
            ov = [(p - b) / b * 100 for p, b in zip(sorted(prof_el), sorted(base_el))]
            print(f"\n## {name}  ({args.trials} trials each)")
            print(f"- program elapsed: unprofiled median {statistics.median(base_el)/1e6:.2f} ms "
                  f"(CV {statistics.pstdev(base_el)/statistics.mean(base_el)*100:.2f}%), profiled median "
                  f"{statistics.median(prof_el)/1e6:.2f} ms (CV {statistics.pstdev(prof_el)/statistics.mean(prof_el)*100:.2f}%)")
            print(f"- overhead on program's own elapsed time: median {statistics.median(ov):+.2f}% "
                  f"(min {min(ov):+.2f}%, max {max(ov):+.2f}%)")
            print(f"- missing events: {missing}/{expected}")
            if lags:
                print(f"- start error (span.start - program stamp): median {statistics.median(lags)/1e3:.2f} us, "
                      f"p95 {pct(lags, .95)/1e3:.2f} us, max {max(lags)/1e3:.2f} us")
                print(f"- end error (program stamp - span.end): median {statistics.median(leads)/1e3:.2f} us, "
                      f"p95 {pct(leads, .95)/1e3:.2f} us, max {max(leads)/1e3:.2f} us")
            if derrs:
                print(f"- duration error (span - truth): median {statistics.median(derrs)/1e3:.2f} us, "
                      f"p5 {pct(derrs, .05)/1e3:.2f} us, p95 {pct(derrs, .95)/1e3:.2f} us")
            if agg:
                rel = [a[0] for a in agg]; per = [a[1] for a in agg]; mean_call = statistics.median(a[2] for a in agg)
                print(f"- aggregation error (sum measured vs sum truth): median {statistics.median(rel):+.3f}% "
                      f"(range [{min(rel):+.3f}%, {max(rel):+.3f}%]); per event {statistics.median(per)/1e3:+.2f} us "
                      f"on a median truth call of {mean_call/1e3:.1f} us")
            if base_wait:
                print(f"- behavior change: program-measured barrier wait unprofiled {statistics.median(base_wait)/1e6:.2f} ms "
                      f"vs profiled {statistics.median(prof_wait)/1e6:.2f} ms (medians)")
            if pairs:
                print(f"- per-thread ordering inversions: {inversions}/{pairs}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
