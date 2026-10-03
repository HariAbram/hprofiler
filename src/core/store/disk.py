"""
DiskTraceStore: an on-disk, indexed trace store for large captures.

Layout (schema version SCHEMA_VERSION; documented in DOCUMENTATION.md,
"Trace store"):

    <name>.hpstore/
        catalog.sqlite          store info, string/code dictionaries,
                                metadata blobs, batch log, and everything
                                derived at finalization (lanes, per-name
                                aggregates, exclusive time, activity index,
                                dependency edges)
        shards/p<pid>.sqlite    one per process: spans, instants, counters

Capture appends events in bounded batches (one transaction per shard per
batch, logged in `batches`); nothing is indexed while capturing.
finalize() builds the indexes and the derived tables. Event ids encode
(shard, kind, row) -- see common.make_eid.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Iterator

import heapq
import numpy as np

from ..events import AnyEvent, Category, CounterEvent, InstantEvent, SpanEvent
from .base import TraceStore, finish_aggregate
from .common import (
    BUCKET_NAMES, CATEGORY_BY_CODE, EDGE_DTYPE, FLAG_DEVICE, FLAG_GPU_MODEL,
    KIND_COUNTER, KIND_INSTANT, KIND_SPAN, MAX_SHARDS, ActivityBuilder, ExclusiveAggregate, LaneInfo,
    LiteSpan,
    SpanFilter, assign_lane_names, exclusive_thread_sweep, lane_base, make_eid, span_bucket,
    span_flags, split_eid,
)

SCHEMA_VERSION = 1
FORMAT_NAME = "hprofiler-trace-store"
# Bumped when derived tables change meaning; a store whose derived version
# differs is re-finalized on open (events are untouched).
DERIVED_VERSION = 1

CATALOG_DDL = """
CREATE TABLE IF NOT EXISTS store_info (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS categories (code INTEGER PRIMARY KEY, value TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS buckets (code INTEGER PRIMARY KEY, value TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS names (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS lane_keys (id INTEGER PRIMARY KEY, cat TEXT NOT NULL, kind TEXT NOT NULL,
                                      ident TEXT NOT NULL, UNIQUE (cat, kind, ident));
CREATE TABLE IF NOT EXISTS shards (shard INTEGER PRIMARY KEY, pid INTEGER UNIQUE NOT NULL,
                                   file TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS batches (id INTEGER PRIMARY KEY, shard INTEGER NOT NULL,
                                    kind INTEGER NOT NULL, first_seq INTEGER NOT NULL,
                                    last_seq INTEGER NOT NULL, rows INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS lanes (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL,
                                  shard INTEGER NOT NULL, lane_key INTEGER NOT NULL, pid INTEGER NOT NULL,
                                  cat TEXT NOT NULL, kind TEXT NOT NULL, ident TEXT NOT NULL,
                                  count INTEGER NOT NULL, first_seq INTEGER NOT NULL,
                                  min_start INTEGER NOT NULL, max_end INTEGER NOT NULL,
                                  max_dur INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS activity (lane INTEGER NOT NULL, level INTEGER NOT NULL, t0 INTEGER NOT NULL,
                                     nbins INTEGER NOT NULL, bin_ns REAL NOT NULL, busy BLOB NOT NULL,
                                     starts BLOB, PRIMARY KEY (lane, level));
CREATE TABLE IF NOT EXISTS agg_names (cat INTEGER NOT NULL, name_id INTEGER NOT NULL,
                                      count INTEGER NOT NULL, total_ns INTEGER NOT NULL,
                                      min_ns INTEGER NOT NULL, max_ns INTEGER NOT NULL,
                                      first_seq INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS exclusive (pid INTEGER NOT NULL, tid INTEGER NOT NULL, cat INTEGER NOT NULL,
                                      name_id INTEGER NOT NULL, bucket INTEGER NOT NULL,
                                      device INTEGER NOT NULL, ns INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS exclusive_threads (pid INTEGER NOT NULL, tid INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS extents (key TEXT PRIMARY KEY, lo INTEGER, hi INTEGER);
CREATE TABLE IF NOT EXISTS edges (version TEXT NOT NULL, ord INTEGER NOT NULL, dst INTEGER NOT NULL,
                                  src INTEGER NOT NULL, kind INTEGER NOT NULL, conf INTEGER NOT NULL,
                                  PRIMARY KEY (version, ord));
"""

SHARD_DDL = """
CREATE TABLE IF NOT EXISTS spans (
    row INTEGER PRIMARY KEY AUTOINCREMENT, seq INTEGER NOT NULL, batch INTEGER NOT NULL,
    start_ns INTEGER NOT NULL, end_ns INTEGER NOT NULL, tid INTEGER NOT NULL,
    cat INTEGER NOT NULL, name_id INTEGER NOT NULL, lane_key INTEGER NOT NULL,
    bucket INTEGER NOT NULL, flags INTEGER NOT NULL,
    stream TEXT, corr TEXT, lid TEXT, type TEXT, side TEXT,
    span_id TEXT, parent_span_id TEXT, tags TEXT NOT NULL, stack TEXT);
CREATE TABLE IF NOT EXISTS instants (
    row INTEGER PRIMARY KEY AUTOINCREMENT, seq INTEGER NOT NULL, batch INTEGER NOT NULL,
    ts_ns INTEGER NOT NULL, tid INTEGER NOT NULL, cat INTEGER NOT NULL, name_id INTEGER NOT NULL,
    tags TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS counters (
    row INTEGER PRIMARY KEY AUTOINCREMENT, seq INTEGER NOT NULL, batch INTEGER NOT NULL,
    ts_ns INTEGER NOT NULL, cat INTEGER NOT NULL, name_id INTEGER NOT NULL,
    value REAL NOT NULL, unit TEXT NOT NULL);
"""

SHARD_INDEXES = """
CREATE INDEX IF NOT EXISTS spans_seq ON spans(seq);
CREATE INDEX IF NOT EXISTS spans_start ON spans(start_ns, seq);
CREATE INDEX IF NOT EXISTS spans_lane ON spans(lane_key, start_ns, seq);
CREATE INDEX IF NOT EXISTS spans_thread ON spans(tid, start_ns, seq);
CREATE INDEX IF NOT EXISTS spans_name ON spans(cat, name_id);
CREATE INDEX IF NOT EXISTS spans_stream ON spans(stream) WHERE stream IS NOT NULL;
CREATE INDEX IF NOT EXISTS spans_span_id ON spans(span_id) WHERE span_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS spans_parent ON spans(parent_span_id) WHERE parent_span_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS spans_corr ON spans(corr) WHERE corr IS NOT NULL;
CREATE INDEX IF NOT EXISTS spans_lid ON spans(lid) WHERE lid IS NOT NULL;
CREATE INDEX IF NOT EXISTS instants_ts ON instants(ts_ns, seq);
CREATE INDEX IF NOT EXISTS counters_name ON counters(name_id, ts_ns, seq);
"""

_SPAN_COLS = ("row, seq, start_ns, end_ns, tid, cat, name_id, tags, stack, span_id, parent_span_id")
_INSERT_SPAN = ("INSERT INTO spans (row, seq, batch, start_ns, end_ns, tid, cat, name_id, lane_key, bucket, "
                "flags, stream, corr, lid, type, side, span_id, parent_span_id, tags, stack) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
_INSERT_INSTANT = "INSERT INTO instants (row, seq, batch, ts_ns, tid, cat, name_id, tags) VALUES (?,?,?,?,?,?,?,?)"
_INSERT_COUNTER = ("INSERT INTO counters (row, seq, batch, ts_ns, cat, name_id, value, unit) "
                   "VALUES (?,?,?,?,?,?,?,?)")
_TABLES = ("spans", "instants", "counters")

_FETCH = 4096


def _dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _tag_col(tags: dict, key: str) -> str | None:
    v = tags.get(key)
    return None if v is None else str(v)


class StoreError(RuntimeError):
    pass


def is_store_path(path: str | os.PathLike) -> bool:
    return (Path(path) / "catalog.sqlite").is_file()


class DiskTraceStore(TraceStore):
    kind = "disk"

    def __init__(self, path: str | os.PathLike, *, create: bool = True, batch_size: int = 8192,
                 max_open_shards: int = 256) -> None:
        self.path = Path(path)
        self.batch_size = batch_size
        self.max_open_shards = max_open_shards
        self._lock = threading.RLock()
        self._derived = {}
        exists = is_store_path(self.path)
        if not exists and not create:
            raise StoreError(f"{self.path}: not a trace store")
        if not exists:
            (self.path / "shards").mkdir(parents=True, exist_ok=True)
        try:
            self._cat = self._connect(self.path / "catalog.sqlite", cache_kib=8192)
            if exists:
                self._check_schema()
            else:
                self._init_catalog()
            self._load_dictionaries()
        except sqlite3.DatabaseError as exc:
            raise StoreError(f"{self.path}: the store catalog is unreadable or corrupted ({exc}); "
                             "re-run the capture, or re-import the JSON export if there is one") from exc
        self._shard_conns: "OrderedDict[int, sqlite3.Connection]" = OrderedDict()
        if exists:
            self._verify_shards()
        self._buf: dict[int, dict[int, list]] = {}
        self._buffered = 0
        self._next_row: dict[tuple[int, int], int] = {}
        self._batch_no = self._info_int("next_batch", 0)
        self._next_seq = self._info_int("next_seq", 0)
        self._finalized = self._info_int("finalized", 0) == 1 and \
            self._info_int("derived_version", 0) == DERIVED_VERSION
        self._content_version = self._info_int("content_version", 0)

    # ── connections / schema ─────────────────────────────────────────────
    @staticmethod
    def _connect(path: Path, cache_kib: int = 2048) -> sqlite3.Connection:
        conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=FILE")       # index-build sorts spill to disk, not RAM
        # Small per-connection page cache: one connection per process
        # shard, so memory must not scale with the number of processes
        # (the OS page cache does the rest).
        conn.execute(f"PRAGMA cache_size=-{cache_kib}")
        return conn

    def _init_catalog(self) -> None:
        c = self._cat
        c.executescript(CATALOG_DDL)
        c.execute("BEGIN")
        c.executemany("INSERT INTO store_info VALUES (?, ?)", [
            ("format", FORMAT_NAME), ("schema_version", str(SCHEMA_VERSION)),
            ("finalized", "0"), ("derived_version", "0"), ("next_seq", "0"), ("next_batch", "0"),
        ])
        c.executemany("INSERT INTO categories VALUES (?, ?)", [(i, v) for i, v in enumerate(CATEGORY_BY_CODE)])
        c.executemany("INSERT INTO buckets VALUES (?, ?)", [(i, v) for i, v in enumerate(BUCKET_NAMES)])
        c.execute("COMMIT")

    def _check_schema(self) -> None:
        info = dict(self._cat.execute("SELECT key, value FROM store_info").fetchall())
        if info.get("format") != FORMAT_NAME:
            raise StoreError(f"{self.path}: not an hprofiler trace store")
        version = int(info.get("schema_version", "0"))
        if version > SCHEMA_VERSION:
            raise StoreError(f"{self.path}: store schema v{version} is newer than this hprofiler "
                             f"(supports v{SCHEMA_VERSION}); upgrade hprofiler to open it")
        if version < SCHEMA_VERSION:
            migrate(self._cat, self.path, version)
        self._cat.executescript(CATALOG_DDL)     # tables added by later minor revisions

    def _info_int(self, key: str, default: int) -> int:
        row = self._cat.execute("SELECT value FROM store_info WHERE key=?", (key,)).fetchone()
        return int(row[0]) if row else default

    def _set_info(self, key: str, value: Any) -> None:
        self._cat.execute("INSERT OR REPLACE INTO store_info VALUES (?, ?)", (key, str(value)))

    def schema_version(self) -> int:
        return self._info_int("schema_version", 0)

    def is_finalized(self) -> bool:
        return self._finalized

    def _load_dictionaries(self) -> None:
        c = self._cat
        stored_cats = dict(c.execute("SELECT code, value FROM categories").fetchall())
        self._cat_value = stored_cats                               # code -> value
        self._cat_code = {v: k for k, v in stored_cats.items()}
        stored_b = dict(c.execute("SELECT code, value FROM buckets").fetchall())
        self._bucket_value = stored_b
        self._bucket_code = {v: k for k, v in stored_b.items()}
        self._names: list[str | None] = []
        self._name_id: dict[str, int] = {}
        for nid, name in c.execute("SELECT id, name FROM names ORDER BY id"):
            while len(self._names) <= nid:
                self._names.append(None)
            self._names[nid] = name
            self._name_id[name] = nid
        self._pending_names: list[tuple[int, str]] = []
        self._lane_keys: dict[tuple[str, str, str], int] = {}
        self._lane_key_by_id: dict[int, tuple[str, str, str]] = {}
        for lid, cat, kind, ident in c.execute("SELECT id, cat, kind, ident FROM lane_keys"):
            self._lane_keys[(cat, kind, ident)] = lid
            self._lane_key_by_id[lid] = (cat, kind, ident)
        self._pending_lane_keys: list[tuple[int, str, str, str]] = []
        self._shard_of_pid: dict[int, int] = {}
        self._pid_of_shard: dict[int, int] = {}
        self._shard_file: dict[int, str] = {}
        for shard, pid, fname in c.execute("SELECT shard, pid, file FROM shards"):
            self._shard_of_pid[pid] = shard
            self._pid_of_shard[shard] = pid
            self._shard_file[shard] = fname

    def _category_code(self, value: str) -> int:
        code = self._cat_code.get(value)
        if code is None:          # a category newer than this store: register it
            code = max(self._cat_value, default=-1) + 1
            self._cat.execute("INSERT INTO categories VALUES (?, ?)", (code, value))
            self._cat_value[code] = value
            self._cat_code[value] = code
        return code

    def _category(self, code: int) -> Category:
        v = self._cat_value.get(code, "other")
        return Category(v) if v in Category._value2member_map_ else Category.OTHER

    def _name_code(self, name: str) -> int:
        nid = self._name_id.get(name)
        if nid is None:
            nid = len(self._names)
            self._names.append(name)
            self._name_id[name] = nid
            self._pending_names.append((nid, name))
        return nid

    def _lane_key(self, base: tuple[str, str, str]) -> int:
        lk = self._lane_keys.get(base)
        if lk is None:
            lk = len(self._lane_keys)
            self._lane_keys[base] = lk
            self._lane_key_by_id[lk] = base
            self._pending_lane_keys.append((lk, *base))
        return lk

    def _shard_for(self, pid: int) -> int:
        shard = self._shard_of_pid.get(pid)
        if shard is None:
            shard = len(self._shard_of_pid)
            if shard >= MAX_SHARDS:
                raise StoreError(f"more than {MAX_SHARDS} processes in one store")
            fname = f"p{pid}.sqlite" if pid >= 0 else f"pm{-pid}.sqlite"
            self._cat.execute("INSERT INTO shards VALUES (?, ?, ?)", (shard, pid, fname))
            self._shard_of_pid[pid] = shard
            self._pid_of_shard[shard] = pid
            self._shard_file[shard] = fname
            conn = self._conn(shard)
            conn.executescript(SHARD_DDL)
        return shard

    _SQLITE_MAGIC = b"SQLite format 3\x00"

    def _verify_shards(self) -> None:
        """Every shard the catalog lists must be present and be an SQLite
        file -- sqlite3.connect() would silently create a missing one empty
        and the first query would fail deep inside a viewer. A zero-length
        shard (capture killed right after creating it) just gets its
        tables."""
        missing, bad = [], []
        for shard, fname in sorted(self._shard_file.items()):
            f = self.path / "shards" / fname
            try:
                with open(f, "rb") as fh:
                    head = fh.read(16)
            except FileNotFoundError:
                missing.append(fname)
                continue
            except OSError as exc:
                bad.append(f"{fname} ({exc.strerror})")
                continue
            if not head:
                try:
                    self._conn(shard).executescript(SHARD_DDL)
                except sqlite3.DatabaseError as exc:
                    bad.append(f"{fname} ({exc})")
            elif head != self._SQLITE_MAGIC:
                bad.append(f"{fname} (not an SQLite file)")
        if missing or bad:
            parts = []
            if missing:
                parts.append(f"{len(missing)} shard file(s) missing: {', '.join(missing[:5])}")
            if bad:
                parts.append(f"{len(bad)} shard file(s) unreadable: {', '.join(bad[:5])}")
            raise StoreError(f"{self.path}: the store is incomplete -- " + "; ".join(parts) +
                             " (copied partially, or files deleted); re-run the capture, or re-import "
                             "the JSON export if there is one")

    def _conn(self, shard: int) -> sqlite3.Connection:
        conn = self._shard_conns.get(shard)
        if conn is not None:
            self._shard_conns.move_to_end(shard)
            return conn
        conn = self._connect(self.path / "shards" / self._shard_file[shard])
        self._shard_conns[shard] = conn
        if len(self._shard_conns) > self.max_open_shards:
            _old, oc = self._shard_conns.popitem(last=False)
            oc.close()
        return conn

    def shards(self) -> list[int]:
        return sorted(self._pid_of_shard)

    # ── writing ──────────────────────────────────────────────────────────
    def _row_for(self, shard: int, kind: int) -> int:
        key = (shard, kind)
        row = self._next_row.get(key)
        if row is None:
            conn = self._conn(shard)
            table = _TABLES[kind]
            mx = conn.execute(f"SELECT COALESCE(MAX(row), 0) FROM {table}").fetchone()[0]
            sq = conn.execute("SELECT seq FROM sqlite_sequence WHERE name=?", (table,)).fetchone()
            row = max(mx, sq[0] if sq else 0) + 1
        self._next_row[key] = row + 1
        return row

    def append(self, event: AnyEvent) -> int:
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            shard = self._shard_for(event.pid)
            bufs = self._buf.setdefault(shard, {KIND_SPAN: [], KIND_INSTANT: [], KIND_COUNTER: []})
            if isinstance(event, SpanEvent):
                kind = KIND_SPAN
                row = self._row_for(shard, kind)
                bufs[kind].append(self._span_row(event, row, seq))
            elif isinstance(event, InstantEvent):
                kind = KIND_INSTANT
                row = self._row_for(shard, kind)
                bufs[kind].append((row, seq, 0, event.timestamp_ns, event.tid,
                                   self._category_code(event.category.value),
                                   self._name_code(event.name), _dumps(event.tags)))
            elif isinstance(event, CounterEvent):
                kind = KIND_COUNTER
                row = self._row_for(shard, kind)
                bufs[kind].append((row, seq, 0, event.timestamp_ns, self._category_code(event.category.value),
                                   self._name_code(event.name), float(event.value), event.unit or ""))
            else:
                raise TypeError(f"not an event: {event!r}")
            event.seq = seq
            event.eid = make_eid(shard, kind, row)
            self._buffered += 1
            if self._derived:
                self._derived = {}
            if self._buffered >= self.batch_size:
                self._flush_locked()
            return event.eid

    def _span_row(self, span: SpanEvent, row: int, seq: int) -> tuple:
        tags = span.tags
        return (row, seq, 0, span.start_ns, span.start_ns + span.duration_ns, span.tid,
                self._category_code(span.category.value), self._name_code(span.name),
                self._lane_key(lane_base(span)), self._bucket_code[span_bucket(span)],
                span_flags(span), _tag_col(tags, "stream"), _tag_col(tags, "corr"),
                _tag_col(tags, "lid"), _tag_col(tags, "type"), _tag_col(tags, "side"),
                span.span_id or None, span.parent_span_id or None, _dumps(tags),
                _dumps(span.stack_frames) if span.stack_frames else None)

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffered and not self._pending_names and not self._pending_lane_keys:
            return
        cat = self._cat
        cat.execute("BEGIN")
        if self._pending_names:
            cat.executemany("INSERT INTO names VALUES (?, ?)", self._pending_names)
            self._pending_names = []
        if self._pending_lane_keys:
            cat.executemany("INSERT INTO lane_keys VALUES (?, ?, ?, ?)", self._pending_lane_keys)
            self._pending_lane_keys = []
        cat.execute("COMMIT")
        batch_log = []
        for shard, bufs in self._buf.items():
            if not any(bufs.values()):
                continue
            conn = self._conn(shard)
            conn.execute("BEGIN")
            for kind, rows in bufs.items():
                if not rows:
                    continue
                batch = self._batch_no
                self._batch_no += 1
                rows = [r[:2] + (batch,) + r[3:] for r in rows]
                conn.executemany((_INSERT_SPAN, _INSERT_INSTANT, _INSERT_COUNTER)[kind], rows)
                batch_log.append((batch, shard, kind, rows[0][1], rows[-1][1], len(rows)))
                bufs[kind] = []
            conn.execute("COMMIT")
        cat.execute("BEGIN")
        if batch_log:
            cat.executemany("INSERT INTO batches VALUES (?, ?, ?, ?, ?, ?)", batch_log)
        self._set_info("next_seq", self._next_seq)
        self._set_info("next_batch", self._batch_no)
        self._bump_content_locked()
        cat.execute("COMMIT")
        self._buffered = 0

    def _bump_content_locked(self) -> None:
        """Events changed: derived tables and persisted edges are stale."""
        self._content_version += 1
        self._set_info("content_version", self._content_version)
        if self._finalized:
            self._finalized = False
            self._set_info("finalized", 0)

    def _mark_dirty(self) -> None:
        self._derived = {}
        self._bump_content_locked()

    def update_span(self, span: SpanEvent) -> None:
        self.update_spans([span])

    def update_spans(self, spans: Iterable[SpanEvent]) -> None:
        with self._lock:
            spans = list(spans)
            # Spans still in the write buffer are replaced there (the
            # Runner attaches call stacks right after each span arrives --
            # flushing for each would turn batches into single rows).
            rest = []
            for span in spans:
                if not self._update_buffered(span):
                    rest.append(span)
            if not rest:
                if self._derived:
                    self._derived = {}
                return
            spans = rest
            self._flush_locked()
            by_shard: dict[int, list] = {}
            for span in spans:
                shard, kind, row = split_eid(span.eid)
                if kind != KIND_SPAN or span.eid < 0:
                    raise ValueError(f"not a stored span: {span!r}")
                r = self._span_row(span, row, span.seq)
                by_shard.setdefault(shard, []).append(r[3:] + (row,))
            if self._pending_names or self._pending_lane_keys:
                self._flush_dictionaries()
            for shard, rows in by_shard.items():
                conn = self._conn(shard)
                conn.execute("BEGIN")
                conn.executemany(
                    "UPDATE spans SET start_ns=?, end_ns=?, tid=?, cat=?, name_id=?, lane_key=?, "
                    "bucket=?, flags=?, stream=?, corr=?, lid=?, type=?, side=?, span_id=?, "
                    "parent_span_id=?, tags=?, stack=? WHERE row=?", rows)
                conn.execute("COMMIT")
            self._mark_dirty()

    def _update_buffered(self, span: SpanEvent) -> bool:
        if span.eid < 0:
            return False
        shard, kind, row = split_eid(span.eid)
        rows = self._buf.get(shard, {}).get(KIND_SPAN)
        if kind != KIND_SPAN or not rows:
            return False
        idx = row - rows[0][0]
        if 0 <= idx < len(rows) and rows[idx][0] == row:
            rows[idx] = self._span_row(span, row, span.seq)
            return True
        return False

    def _flush_dictionaries(self) -> None:
        cat = self._cat
        cat.execute("BEGIN")
        if self._pending_names:
            cat.executemany("INSERT INTO names VALUES (?, ?)", self._pending_names)
            self._pending_names = []
        if self._pending_lane_keys:
            cat.executemany("INSERT INTO lane_keys VALUES (?, ?, ?, ?)", self._pending_lane_keys)
            self._pending_lane_keys = []
        cat.execute("COMMIT")

    def delete_spans(self, eids: Iterable[int]) -> None:
        with self._lock:
            self._flush_locked()
            by_shard: dict[int, list[int]] = {}
            for e in eids:
                shard, kind, row = split_eid(e)
                if kind == KIND_SPAN:
                    by_shard.setdefault(shard, []).append(row)
            for shard, rows in by_shard.items():
                conn = self._conn(shard)
                conn.execute("BEGIN")
                conn.executemany("DELETE FROM spans WHERE row=?", [(r,) for r in rows])
                conn.execute("COMMIT")
            self._mark_dirty()

    # ── reading ──────────────────────────────────────────────────────────
    def _row_to_span(self, shard: int, r: tuple) -> SpanEvent:
        row, seq, start, end, tid, cat, name_id, tags, stack, sid, psid = r
        s = SpanEvent(name=self._names[name_id], category=self._category(cat), start_ns=start,
                      duration_ns=end - start, pid=self._pid_of_shard[shard], tid=tid,
                      tags=json.loads(tags), stack_frames=json.loads(stack) if stack else [],
                      span_id=sid or "", parent_span_id=psid or "")
        s.eid = make_eid(shard, KIND_SPAN, row)
        s.seq = seq
        return s

    def _query_iter(self, shard: int, sql: str, params: tuple, convert) -> Iterator:
        """Rows fetched in chunks under the lock; the lock is not held
        while the caller consumes them."""
        with self._lock:
            cur = self._conn(shard).execute(sql, params)
        while True:
            with self._lock:
                rows = cur.fetchmany(_FETCH)
            if not rows:
                return
            for r in rows:
                yield convert(shard, r)

    def _count(self, table: str) -> int:
        with self._lock:
            self._flush_locked()
            return sum(self._conn(sh).execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                       for sh in self.shards())

    def span_count(self) -> int:
        c = self._cache()
        if "n_spans" not in c:
            c["n_spans"] = self._count("spans")
        return c["n_spans"]

    def instant_count(self) -> int:
        return self._count("instants")

    def counter_count(self) -> int:
        return self._count("counters")

    def _span_where(self, *, tid=None, lane=None, categories=None, window=None, has_stack=None,
                    gpu_model=None, with_ids=None, filt: SpanFilter | None = None,
                    lookback: int = 0) -> tuple[str, list]:
        clauses: list[str] = []
        params: list = []
        if tid is not None:
            clauses.append("tid = ?")
            params.append(tid)
        if lane is not None:
            clauses.append("lane_key = ?")
            params.append(lane.lane_key)
        if categories is not None:
            codes = [self._cat_code[c] for c in categories if c in self._cat_code]
            clauses.append(f"cat IN ({','.join('?' * len(codes))})" if codes else "0")
            params.extend(codes)
        if window is not None:
            a, b = window
            clauses.append("start_ns >= ? AND start_ns <= ? AND end_ns > ?")
            params.extend([a - lookback, b, a])
        if has_stack is not None:
            clauses.append("stack IS NOT NULL" if has_stack else "stack IS NULL")
        if gpu_model is not None:
            clauses.append(f"(flags & {FLAG_GPU_MODEL}) {'!=' if gpu_model else '='} 0")
        if with_ids is not None:
            clauses.append("(span_id IS NOT NULL OR parent_span_id IS NOT NULL)" if with_ids
                           else "(span_id IS NULL AND parent_span_id IS NULL)")
        if filt is not None and not filt.is_empty():
            if filt.min_dur_ns > 0:
                clauses.append("end_ns - start_ns >= ?")
                params.append(int(filt.min_dur_ns))
            if filt.range_start_ns is not None:
                clauses.append("end_ns > ? AND start_ns < ?")
                params.extend([int(filt.range_start_ns), int(filt.range_end_ns)])
            if filt.categories:
                codes = [self._cat_code[c] for c in filt.categories if c in self._cat_code]
                clauses.append(f"cat IN ({','.join('?' * len(codes))})" if codes else "0")
                params.extend(codes)
            if filt.buckets:
                codes = [self._bucket_code[b] for b in filt.buckets if b in self._bucket_code]
                clauses.append(f"bucket IN ({','.join('?' * len(codes))})" if codes else "0")
                params.extend(codes)
            m = filt.name_matcher()
            if m is not None:
                ids = [i for i, n in enumerate(self._names) if n is not None and m(n)]
                if not ids:
                    clauses.append("0")
                elif len(ids) <= 900:
                    clauses.append(f"name_id IN ({','.join('?' * len(ids))})")
                    params.extend(ids)
                else:
                    clauses.append(f"name_id IN ({','.join(str(i) for i in ids)})")
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def iter_spans(self, *, order: str = "seq", pid: int | None = None, tid: int | None = None,
                   lane: str | None = None, categories: Iterable[str] | None = None,
                   window: tuple[int, int] | None = None, has_stack: bool | None = None,
                   gpu_model: bool | None = None, with_ids: bool | None = None,
                   filt: SpanFilter | None = None) -> Iterator[SpanEvent]:
        with self._lock:
            self._flush_locked()
        lane_info = None
        if lane is not None:
            lane_info = self.lane(lane)
            if lane_info is None:
                return iter(())
            pid = lane_info.pid
        if pid is not None:
            shard = self._shard_of_pid.get(pid)
            shards = [] if shard is None else [shard]
        else:
            shards = self.shards()
        lookback = 0
        if window is not None:
            lookback = lane_info.max_dur if lane_info is not None else self._max_dur(tid=tid)
        where, params = self._span_where(tid=tid, lane=lane_info, categories=categories, window=window,
                                         has_stack=has_stack, gpu_model=gpu_model, with_ids=with_ids,
                                         filt=filt, lookback=lookback)
        order_sql = " ORDER BY row" if order == "seq" else " ORDER BY start_ns, seq"
        sql = f"SELECT {_SPAN_COLS} FROM spans{where}{order_sql}"
        iters = [self._query_iter(sh, sql, tuple(params), self._row_to_span) for sh in shards]
        if len(iters) == 1:
            return iters[0]
        key = (lambda s: s.seq) if order == "seq" else (lambda s: (s.start_ns, s.seq))
        return heapq.merge(*iters, key=key)

    def _max_dur(self, tid: int | None = None) -> int:
        c = self._cache()
        if "max_dur" not in c:
            m = 0
            with self._lock:
                for sh in self.shards():
                    v = self._conn(sh).execute("SELECT MAX(end_ns - start_ns) FROM spans").fetchone()[0]
                    m = max(m, v or 0)
            c["max_dur"] = m
        return c["max_dur"]

    def _row_to_instant(self, shard: int, r: tuple) -> InstantEvent:
        row, seq, ts, tid, cat, name_id, tags = r
        e = InstantEvent(name=self._names[name_id], category=self._category(cat), timestamp_ns=ts,
                         pid=self._pid_of_shard[shard], tid=tid, tags=json.loads(tags))
        e.eid = make_eid(shard, KIND_INSTANT, row)
        e.seq = seq
        return e

    def _row_to_counter(self, shard: int, r: tuple) -> CounterEvent:
        row, seq, ts, cat, name_id, value, unit = r
        e = CounterEvent(name=self._names[name_id], category=self._category(cat), timestamp_ns=ts,
                         value=value, unit=unit, pid=self._pid_of_shard[shard])
        e.eid = make_eid(shard, KIND_COUNTER, row)
        e.seq = seq
        return e

    def iter_instants(self) -> Iterator[InstantEvent]:
        with self._lock:
            self._flush_locked()
        sql = "SELECT row, seq, ts_ns, tid, cat, name_id, tags FROM instants ORDER BY row"
        return heapq.merge(*[self._query_iter(sh, sql, (), self._row_to_instant) for sh in self.shards()],
                           key=lambda e: e.seq)

    def iter_counters(self, *, order: str = "seq") -> Iterator[CounterEvent]:
        with self._lock:
            self._flush_locked()
        cols = "row, seq, ts_ns, cat, name_id, value, unit"
        if order != "name":
            sql = f"SELECT {cols} FROM counters ORDER BY row"
            return heapq.merge(*[self._query_iter(sh, sql, (), self._row_to_counter) for sh in self.shards()],
                               key=lambda e: e.seq)

        def by_name() -> Iterator[CounterEvent]:
            with self._lock:
                ids = set()
                for sh in self.shards():
                    ids.update(r[0] for r in self._conn(sh).execute("SELECT DISTINCT name_id FROM counters"))
            for nid in sorted(ids, key=lambda i: self._names[i]):
                sql = f"SELECT {cols} FROM counters WHERE name_id = ? ORDER BY ts_ns, seq"
                yield from heapq.merge(*[self._query_iter(sh, sql, (nid,), self._row_to_counter)
                                         for sh in self.shards()],
                                       key=lambda e: (e.timestamp_ns, e.seq))
        return by_name()

    def event_by_id(self, eid: int) -> AnyEvent | None:
        shard, kind, row = split_eid(eid)
        if shard not in self._pid_of_shard:
            return None
        with self._lock:
            self._flush_locked()
            conn = self._conn(shard)
            if kind == KIND_SPAN:
                r = conn.execute(f"SELECT {_SPAN_COLS} FROM spans WHERE row=?", (row,)).fetchone()
                return self._row_to_span(shard, r) if r else None
            if kind == KIND_INSTANT:
                r = conn.execute("SELECT row, seq, ts_ns, tid, cat, name_id, tags FROM instants WHERE row=?",
                                 (row,)).fetchone()
                return self._row_to_instant(shard, r) if r else None
            r = conn.execute("SELECT row, seq, ts_ns, cat, name_id, value, unit FROM counters WHERE row=?",
                             (row,)).fetchone()
            return self._row_to_counter(shard, r) if r else None

    def span_by_seq(self, seq: int) -> SpanEvent | None:
        with self._lock:
            self._flush_locked()
            for sh in self._shards_for_seq(seq):
                r = self._conn(sh).execute(f"SELECT {_SPAN_COLS} FROM spans WHERE seq=?", (seq,)).fetchone()
                if r:
                    return self._row_to_span(sh, r)
        return None

    def _shards_for_seq(self, seq: int) -> list[int]:
        rows = self._cat.execute(
            "SELECT DISTINCT shard FROM batches WHERE kind=? AND first_seq <= ? AND last_seq >= ?",
            (KIND_SPAN, seq, seq)).fetchall()
        return [r[0] for r in rows]

    def span_seqs(self) -> np.ndarray:
        with self._lock:
            self._flush_locked()
            parts = [np.fromiter((r[0] for r in self._conn(sh).execute("SELECT seq FROM spans")),
                                 dtype=np.int64) for sh in self.shards()]
        if not parts:
            return np.empty(0, dtype=np.int64)
        return np.sort(np.concatenate(parts))

    # ── metadata ─────────────────────────────────────────────────────────
    def save_meta(self, key: str, value: Any) -> None:
        with self._lock:
            self._cat.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, _dumps(value)))

    def load_meta(self, key: str, default: Any = None) -> Any:
        with self._lock:
            r = self._cat.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    # ── derived: SQL versions of the generic algorithms ──────────────────
    def pids(self) -> list[int]:
        with self._lock:
            self._flush_locked()
            return sorted(self._pid_of_shard[sh] for sh in self.shards()
                          if self._conn(sh).execute("SELECT 1 FROM spans LIMIT 1").fetchone())

    def threads(self) -> list[tuple[int, int]]:
        out = []
        with self._lock:
            self._flush_locked()
            for sh in self.shards():
                pid = self._pid_of_shard[sh]
                out.extend((pid, r[0]) for r in self._conn(sh).execute("SELECT DISTINCT tid FROM spans"))
        return sorted(out)

    def span_extent(self, *, timed_only: bool = True) -> tuple[int, int] | None:
        key = ("extent", timed_only)
        c = self._cache()
        if key not in c:
            c[key] = self._persisted_extent(f"spans_{int(timed_only)}") or self._sql_extent(timed_only)
        return c[key]

    def _sql_extent(self, timed_only: bool) -> tuple[int, int] | None:
        lo = hi = None
        where = " WHERE end_ns > start_ns" if timed_only else ""
        with self._lock:
            self._flush_locked()
            for sh in self.shards():
                a, b = self._conn(sh).execute(f"SELECT MIN(start_ns), MAX(end_ns) FROM spans{where}").fetchone()
                if a is not None:
                    lo = a if lo is None else min(lo, a)
                    hi = b if hi is None else max(hi, b)
        return None if lo is None else (lo, hi)

    def event_extent(self) -> tuple[int, int] | None:
        c = self._cache()
        if "event_extent" not in c:
            ext = self._persisted_extent("events")
            if ext is None:
                vals = []
                sp = self._sql_extent(False)
                if sp:
                    vals.extend(sp)
                with self._lock:
                    for sh in self.shards():
                        conn = self._conn(sh)
                        for table in ("instants", "counters"):
                            a, b = conn.execute(f"SELECT MIN(ts_ns), MAX(ts_ns) FROM {table}").fetchone()
                            if a is not None:
                                vals.extend((a, b))
                ext = (min(vals), max(vals)) if vals else None
            c["event_extent"] = ext
        return c["event_extent"]

    def _persisted_extent(self, key: str) -> tuple[int, int] | None:
        if not self.is_finalized():
            return None
        r = self._cat.execute("SELECT lo, hi FROM extents WHERE key=?", (key,)).fetchone()
        return None if r is None or r[0] is None else (r[0], r[1])

    def lane_infos(self) -> list[LaneInfo]:
        c = self._cache()
        if "lanes" not in c:
            lanes = self._load_lanes() if self.is_finalized() else None
            if lanes is None:
                groups = []
                with self._lock:
                    self._flush_locked()
                    for sh in self.shards():
                        pid = self._pid_of_shard[sh]
                        for lk, n, fs, mn, mx, md in self._conn(sh).execute(
                                "SELECT lane_key, COUNT(*), MIN(seq), MIN(start_ns), MAX(end_ns), "
                                "MAX(end_ns - start_ns) FROM spans GROUP BY lane_key"):
                            groups.append({"pid": pid, "base": self._lane_key_by_id[lk], "count": n,
                                           "first_seq": fs, "min_start": mn, "max_end": mx,
                                           "max_dur": md, "lane_key": lk})
                lanes = assign_lane_names(groups)
            c["lanes"] = lanes
        return c["lanes"]

    def _load_lanes(self) -> list[LaneInfo] | None:
        rows = self._cat.execute(
            "SELECT name, cat, kind, ident, pid, count, first_seq, min_start, max_end, max_dur, lane_key "
            "FROM lanes ORDER BY first_seq").fetchall()
        if not rows and self.span_count():
            return None
        return [LaneInfo(name=r[0], cat=r[1], kind=r[2], ident=r[3], pid=r[4], count=r[5],
                         first_seq=r[6], min_start=r[7], max_end=r[8], max_dur=r[9], lane_key=r[10])
                for r in rows]

    def aggregate_stats(self) -> list[dict]:
        c = self._cache()
        if "agg" not in c:
            if self.is_finalized():
                rows_src = self._cat.execute(
                    "SELECT cat, name_id, count, total_ns, min_ns, max_ns, first_seq FROM agg_names").fetchall()
            else:
                rows_src = self._sql_agg()
            rows = []
            for cat, nid, n, tot, mn, mx, fs in sorted(rows_src, key=lambda r: r[6]):
                rows.append({"name": self._names[nid], "category": self._cat_value[cat], "count": n,
                             "total_ns": tot, "min_ns": mn, "max_ns": mx, "_first": fs})
            for r in rows:
                r.pop("_first")
            c["agg"] = finish_aggregate(rows)
        return [dict(r) for r in c["agg"]]

    def _sql_agg(self) -> list[tuple]:
        merged: dict[tuple[int, int], list] = {}
        with self._lock:
            self._flush_locked()
            for sh in self.shards():
                for cat, nid, n, tot, mn, mx, fs in self._conn(sh).execute(
                        "SELECT cat, name_id, COUNT(*), SUM(end_ns - start_ns), MIN(end_ns - start_ns), "
                        "MAX(end_ns - start_ns), MIN(seq) FROM spans GROUP BY cat, name_id"):
                    m = merged.get((cat, nid))
                    if m is None:
                        merged[(cat, nid)] = [cat, nid, n, tot, mn, mx, fs]
                    else:
                        m[2] += n
                        m[3] += tot
                        m[4] = min(m[4], mn)
                        m[5] = max(m[5], mx)
                        m[6] = min(m[6], fs)
        return [tuple(v) for v in merged.values()]

    def _compute_exclusive(self) -> ExclusiveAggregate:
        if self.is_finalized():
            agg = ExclusiveAggregate()
            for pid, tid, cat, nid, b, dev, ns in self._cat.execute(
                    "SELECT pid, tid, cat, name_id, bucket, device, ns FROM exclusive"):
                agg.rows[(pid, tid, self._cat_value[cat], self._names[nid],
                          self._bucket_value[b], bool(dev))] = ns
            agg.threads = {tuple(r) for r in self._cat.execute("SELECT pid, tid FROM exclusive_threads")}
            return agg
        annot = self._bucket_code["Annotation"]
        agg = ExclusiveAggregate()
        with self._lock:
            self._flush_locked()
            for sh in self.shards():
                pid = self._pid_of_shard[sh]
                conn = self._conn(sh)
                for tid, cat, nid, b, ns in conn.execute(
                        f"SELECT tid, cat, name_id, bucket, SUM(end_ns - start_ns) FROM spans "
                        f"WHERE (flags & {FLAG_DEVICE}) != 0 AND end_ns > start_ns AND bucket != ? "
                        f"GROUP BY tid, cat, name_id, bucket", (annot,)):
                    agg.add((pid, tid, self._cat_value[cat], self._names[nid], self._bucket_value[b], True), ns)
                for (tid,) in conn.execute(
                        f"SELECT DISTINCT tid FROM spans WHERE (flags & {FLAG_DEVICE}) = 0 "
                        f"AND end_ns > start_ns AND bucket != ?", (annot,)).fetchall():
                    agg.threads.add((pid, tid))
        cats = {code: self._category(code) for code in self._cat_value}
        names, buckets = self._names, self._bucket_value
        for pid, tid in sorted(agg.threads):
            shard = self._shard_of_pid[pid]
            sql = (f"SELECT start_ns, end_ns, seq, cat, name_id, bucket FROM spans WHERE tid = ? "
                   f"AND (flags & {FLAG_DEVICE}) = 0 AND end_ns > start_ns AND bucket != ? "
                   f"ORDER BY start_ns, seq")

            def lite(_shard, r, pid=pid, tid=tid):
                return LiteSpan(r[0], r[1], r[2], pid, tid, names[r[4]], cats[r[3]], buckets[r[5]])
            exclusive_thread_sweep(self._query_iter(shard, sql, (tid, annot), lite), agg)
        return agg

    def _build_activity(self, lane: str) -> list[dict] | None:
        info = self.lane(lane)
        if info is None:
            return None
        if self.is_finalized():
            rows = self._cat.execute(
                "SELECT a.level, a.t0, a.nbins, a.bin_ns, a.busy, a.starts FROM activity a "
                "JOIN lanes l ON l.id = a.lane WHERE l.name = ? ORDER BY a.level", (lane,)).fetchall()
            if rows:
                levels = []
                for level, t0, nb, bin_ns, busy, starts in rows:
                    lv = {"nbins": nb, "bin_ns": bin_ns, "t0": t0,
                          "busy": np.frombuffer(zlib.decompress(busy), dtype=np.float32)}
                    if starts is not None:
                        lv["starts"] = np.frombuffer(zlib.decompress(starts), dtype=np.int64)
                    levels.append(lv)
                return levels
        grid = self.activity_grid()
        if grid is None:
            return None
        b = ActivityBuilder(grid[0], grid[1])
        add = b.add
        for start, end in self._query_iter(
                self._shard_of_pid[info.pid],
                "SELECT start_ns, end_ns FROM spans WHERE lane_key = ? ORDER BY start_ns, seq",
                (info.lane_key,), lambda _sh, r: r):
            add(start, end)
        levels = b.finish()
        for lv in levels:
            lv["t0"] = grid[0]
        levels[0]["starts"] = b.starts
        return levels

    def iter_spans_light(self, *, pid: int | None = None, tid: int | None = None) -> Iterator[SpanEvent]:
        with self._lock:
            self._flush_locked()
        shards = self.shards() if pid is None else (
            [self._shard_of_pid[pid]] if pid in self._shard_of_pid else [])
        where, params = self._span_where(tid=tid)
        sql = f"SELECT seq, start_ns, end_ns, tid, cat, name_id, span_id, parent_span_id FROM spans{where} ORDER BY row"
        names, category, pid_of = self._names, self._category, self._pid_of_shard

        def light(shard, r):
            s = SpanEvent(name=names[r[5]], category=category(r[4]), start_ns=r[1], duration_ns=r[2] - r[1],
                          pid=pid_of[shard], tid=r[3], span_id=r[6] or "", parent_span_id=r[7] or "")
            s.seq = r[0]
            return s
        iters = [self._query_iter(sh, sql, tuple(params), light) for sh in shards]
        if len(iters) == 1:
            return iters[0]
        return heapq.merge(*iters, key=lambda s: s.seq)

    def iter_intervals(self, *, pid: int | None = None, categories: Iterable[str] | None = None,
                       exclude_types: Iterable[str] | None = None,
                       timed_only: bool = True) -> Iterator[tuple[int, int]]:
        with self._lock:
            self._flush_locked()
        shards = self.shards() if pid is None else (
            [self._shard_of_pid[pid]] if pid in self._shard_of_pid else [])
        where, params = self._span_where(categories=categories)
        clauses = [where[len(" WHERE "):]] if where else []
        if timed_only:
            clauses.append("end_ns > start_ns")
        excl = sorted(set(exclude_types or ()))
        if excl:
            clauses.append(f"(type IS NULL OR type NOT IN ({','.join('?' * len(excl))}))")
            params = list(params) + excl
        sql = "SELECT start_ns, end_ns FROM spans" + (" WHERE " + " AND ".join(clauses) if clauses else "") + \
              " ORDER BY start_ns, seq"
        iters = [self._query_iter(sh, sql, tuple(params), lambda _sh, r: r) for sh in shards]
        if len(iters) == 1:
            return iters[0]
        return heapq.merge(*iters)

    def interval_arrays(self, *, categories: Iterable[str] | None = None,
                        chunk: int = 65536) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        with self._lock:
            self._flush_locked()
        where, params = self._span_where(categories=categories)
        for sh in self.shards():
            with self._lock:
                cur = self._conn(sh).execute(f"SELECT start_ns, end_ns FROM spans{where}", params)
            while True:
                with self._lock:
                    rows = cur.fetchmany(chunk)
                if not rows:
                    break
                a = np.asarray(rows, dtype=np.int64)
                yield a[:, 0].copy(), a[:, 1].copy()

    def first_with_tag(self, key: str) -> dict[str, SpanEvent]:
        c = self._cache()
        if ("first_with_tag", key) not in c:
            # Only rows whose tag text contains the key are decoded.
            needle = json.dumps(key) + ":"
            best: dict[str, tuple[int, SpanEvent]] = {}
            with self._lock:
                self._flush_locked()
            for sh in self.shards():
                sql = f"SELECT {_SPAN_COLS} FROM spans WHERE instr(tags, ?) > 0 ORDER BY row"
                for span in self._query_iter(sh, sql, (needle,), self._row_to_span):
                    if span.tags.get(key) and (span.name not in best or span.seq < best[span.name][0]):
                        best[span.name] = (span.seq, span)
            c[("first_with_tag", key)] = {n: sp for n, (_q, sp) in sorted(best.items(), key=lambda kv: kv[1][0])}
        return c[("first_with_tag", key)]

    def window_columns(self, lane: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
        info = self.lane(lane)
        if info is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), []
        where, params = self._span_where(lane=info, window=(start, end), lookback=info.max_dur)
        with self._lock:
            self._flush_locked()
            rows = self._conn(self._shard_of_pid[info.pid]).execute(
                f"SELECT start_ns, end_ns, name_id FROM spans{where} ORDER BY start_ns, seq", params).fetchall()
        if not rows:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), []
        a = np.asarray(rows, dtype=np.int64)
        names = self._names
        return a[:, 0].copy(), a[:, 1].copy(), [names[i] for i in a[:, 2].tolist()]

    def span_by_span_id(self, span_id: str) -> SpanEvent | None:
        best = None
        with self._lock:
            self._flush_locked()
            for sh in self.shards():
                r = self._conn(sh).execute(f"SELECT {_SPAN_COLS} FROM spans WHERE span_id = ? "
                                           f"ORDER BY seq DESC LIMIT 1", (span_id,)).fetchone()
                if r is not None and (best is None or r[1] > best[1][1]):
                    best = (sh, r)
        return None if best is None else self._row_to_span(*best)

    def count_window(self, lane: str, start: int, end: int, *, filt: SpanFilter | None = None,
                     cap: int | None = None) -> int:
        info = self.lane(lane)
        if info is None:
            return 0
        where, params = self._span_where(lane=info, window=(start, end), filt=filt, lookback=info.max_dur)
        sql = f"SELECT COUNT(*) FROM (SELECT 1 FROM spans{where}" + (f" LIMIT {cap + 1})" if cap else ")")
        with self._lock:
            self._flush_locked()
            return self._conn(self._shard_of_pid[info.pid]).execute(sql, params).fetchone()[0]

    def lane_count(self, lane: str, filt: SpanFilter | None = None) -> int:
        info = self.lane(lane)
        if info is None:
            return 0
        if filt is None or filt.is_empty():
            return info.count
        where, params = self._span_where(lane=info, filt=filt)
        with self._lock:
            return self._conn(self._shard_of_pid[info.pid]).execute(
                f"SELECT COUNT(*) FROM spans{where}", params).fetchone()[0]

    def names(self) -> list[str]:
        return [n for n in self._names if n is not None]

    # ── finalization ─────────────────────────────────────────────────────
    def finalize(self, progress=None) -> None:
        """Build indexes and persist every derived table. Idempotent."""
        with self._lock:
            self._flush_locked()
            if self.is_finalized():
                return
            for i, sh in enumerate(self.shards()):
                self._conn(sh).executescript(SHARD_INDEXES)
                if progress:
                    progress("indexes", i + 1, len(self.shards()))
            self._derived = {}
            cat = self._cat
            cat.execute("BEGIN")
            for table in ("lanes", "activity", "agg_names", "exclusive", "exclusive_threads", "extents"):
                cat.execute(f"DELETE FROM {table}")
            cat.execute("COMMIT")
            lanes = self.lane_infos()
            agg_rows = self._sql_agg()
            extents = {"spans_1": self._sql_extent(True), "spans_0": self._sql_extent(False)}
            self._derived["extent", True] = extents["spans_1"]
            self._derived["extent", False] = extents["spans_0"]
            extents["events"] = self.event_extent()
        excl = self.exclusive_aggregate()
        lane_ids = {ln.name: i for i, ln in enumerate(lanes)}
        activity_rows = []
        for i, ln in enumerate(lanes):
            levels = self._build_activity(ln.name) or []
            for lvl_no, lv in enumerate(levels):
                starts = lv.get("starts")
                activity_rows.append((lane_ids[ln.name], lvl_no, lv["t0"], lv["nbins"], lv["bin_ns"],
                                      zlib.compress(np.asarray(lv["busy"], dtype=np.float32).tobytes(), 6),
                                      None if starts is None else
                                      zlib.compress(np.asarray(starts, dtype=np.int64).tobytes(), 6)))
            if progress:
                progress("activity", i + 1, len(lanes))
        with self._lock:
            cat = self._cat
            cat.execute("BEGIN")
            cat.executemany("INSERT INTO lanes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [
                (lane_ids[ln.name], ln.name, self._shard_of_pid[ln.pid], ln.lane_key, ln.pid, ln.cat,
                 ln.kind, ln.ident, ln.count, ln.first_seq, ln.min_start, ln.max_end, ln.max_dur)
                for ln in lanes])
            cat.executemany("INSERT INTO activity VALUES (?,?,?,?,?,?,?)", activity_rows)
            cat.executemany("INSERT INTO agg_names VALUES (?,?,?,?,?,?,?)", agg_rows)
            cat.executemany("INSERT INTO exclusive VALUES (?,?,?,?,?,?,?)", [
                (pid, tid, self._cat_code[c_], self._name_code(n), self._bucket_code[b], int(d), ns)
                for (pid, tid, c_, n, b, d), ns in excl.rows.items()])
            cat.executemany("INSERT INTO exclusive_threads VALUES (?, ?)", sorted(excl.threads))
            cat.executemany("INSERT INTO extents VALUES (?, ?, ?)",
                            [(k, v[0] if v else None, v[1] if v else None) for k, v in extents.items()])
            self._set_info("finalized", 1)
            self._set_info("derived_version", DERIVED_VERSION)
            cat.execute("COMMIT")
            self._finalized = True
            if self._pending_names:
                self._flush_dictionaries()
            self._derived["excl"] = excl

    def compact(self) -> None:
        """Checkpoint write-ahead logs so the store is a set of plain files
        (safe to copy or archive)."""
        with self._lock:
            for conn in [self._cat, *self._shard_conns.values()]:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    # ── dependency edges ─────────────────────────────────────────────────
    def save_edges(self, version: str, edges: np.ndarray) -> None:
        """Persist dependency edges; valid until the events change."""
        tagged = f"{version}@{self._content_version}"
        with self._lock:
            cat = self._cat
            cat.execute("BEGIN")
            cat.execute("DELETE FROM edges")
            step = 65536                       # bounded: never every edge as Python tuples at once
            for a in range(0, len(edges), step):
                cat.executemany("INSERT INTO edges VALUES (?,?,?,?,?,?)",
                                ((tagged, a + i, d, s_, k, c)
                                 for i, (d, s_, k, c) in enumerate(edges[a:a + step].tolist())))
            cat.execute("COMMIT")

    def load_edges(self, version: str, kinds: Iterable[int] | None = None) -> np.ndarray | None:
        with self._lock:
            self._flush_locked()
            tagged = f"{version}@{self._content_version}"
            if self._cat.execute("SELECT 1 FROM edges WHERE version=? LIMIT 1", (tagged,)).fetchone() is None:
                return None
            where, params = "version=?", [tagged]
            if kinds is not None:
                kinds = [int(k) for k in kinds]
                where += f" AND kind IN ({','.join('?' * len(kinds))})" if kinds else " AND 0"
                params += kinds
            cur = self._cat.execute(f"SELECT dst, src, kind, conf FROM edges WHERE {where} ORDER BY ord", params)
            parts = []
            while True:
                rows = cur.fetchmany(65536)
                if not rows:
                    break
                a = np.array(rows, dtype=np.int64)
                part = np.empty(len(a), dtype=EDGE_DTYPE)
                part["dst"], part["src"], part["kind"], part["conf"] = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
                parts.append(part)
        if not parts:
            return np.empty(0, dtype=EDGE_DTYPE)
        return np.concatenate(parts)

    def close(self) -> None:
        with self._lock:
            self._flush_locked()
            for conn in self._shard_conns.values():
                conn.close()
            self._shard_conns.clear()
            self._cat.close()


def migrate(conn: sqlite3.Connection, path: Path, from_version: int) -> None:
    """Upgrade an older store in place. Version 1 is the first schema;
    future versions add their steps here (see DOCUMENTATION.md, "Trace
    store", for the migration policy)."""
    raise StoreError(f"{path}: store schema v{from_version} cannot be migrated to v{SCHEMA_VERSION}")
