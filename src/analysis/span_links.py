"""
Which span a span's psid= (parent_span_id) names.

Span ids are joined by value within a process, and every hook numbers its
own spans: traces recorded before the hooks got separate id spaces
(hooks/common/hp_ids.h) can carry one id on two unrelated spans -- the MPI
hook's request ids and the CUDA/ROCm hooks' host-call ids were both
(pid << 32) | n, counted from 1 in the same process (an MPI + CUDA program:
816 shared ids in one LLNL Comb trace), and so were OMPT's parallel-region
ids. A lookup that let the last span with an id win joined a CUDA call to
an MPI request; the call tree, which shared one node per id, even closed
cycles through them.

The resolver keeps every span of an id and, when several share it, picks
the one of the child's own hook family (an MPI completion names an MPI
request, a device span its host call, an OpenMP span its parallel region),
else an enclosing annotation range (a host call's psid= is the NVTX range
open on its thread); among several of that family the latest that started
no later than the child. Nothing is guessed across families.
"""
from __future__ import annotations

from collections import defaultdict

_ANNOTATION_TYPES = frozenset({"nvtx_range", "nvtx_mark", "roctx_range", "roctx_mark"})


def id_family(span) -> str:
    """The family of hooks a span's ids come from. Reads only the category,
    name and type tag, so it works on "light" spans (no tags) too."""
    cat = span.category.value
    if cat == "mpi":
        return "mpi"
    if cat == "nvtx" or span.tags.get("type") in _ANNOTATION_TYPES:
        return "annotation"
    if cat == "openmp" or (cat == "sync" and span.name.startswith(("omp_", "GOMP_"))):
        return "openmp"
    if cat in ("cuda", "rocm", "opencl", "nccl", "memory", "sync"):
        return "gpu"
    return cat


class SpanIdResolver:
    """(pid, span id) -> the value registered for the span carrying it."""

    def __init__(self) -> None:
        self._by: dict[tuple[int, str], list[tuple[int, str, object]]] = defaultdict(list)

    def add(self, pid: int, span_id: str, start_ns: int, family: str, value) -> None:
        if span_id:
            self._by[(pid, span_id)].append((start_ns, family, value))

    def add_span(self, span, value) -> None:
        self.add(span.pid, span.span_id, span.start_ns, id_family(span), value)

    def ambiguous_ids(self) -> int:
        return sum(1 for c in self._by.values() if len(c) > 1)

    def resolve(self, pid: int, parent_id: str, start_ns: int, family: str):
        """The value of the span `parent_id` names for a child of `family`
        that started at `start_ns`, or None (unknown, or ambiguous)."""
        cands = self._by.get((pid, parent_id))
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0][2]
        pick = [c for c in cands if c[1] == family] or [c for c in cands if c[1] == "annotation"]
        if len(pick) == 1:
            return pick[0][2]
        before = [c for c in pick if c[0] <= start_ns]
        return max(before, key=lambda c: c[0])[2] if before else None

    def parent_of(self, child, parent_id: str | None = None):
        return self.resolve(child.pid, parent_id or child.parent_span_id, child.start_ns, id_family(child))
