from __future__ import annotations

"""Persistent, bounded NVMe tier for paged K/V images and recurrent checkpoints (round 3).

Opt-in with EXL3_NVME_TIER=<absolute directory on a dedicated filesystem>. Unset, this module is never imported.

Two layers:

- ``SegmentStore`` (stdlib only, CPU-testable) owns every byte of file I/O: checksummed records in append-only
  segment files, the in-memory hash-chain index rebuilt from the records at open, the byte cap, eviction and
  space reclaim, namespace identity and GC. Files are mutated by one thread (the writer thread, or the
  constructor before the writer starts); readers only read immutable records.
- ``DiskPageCache`` (torch) stages exact page images and checkpoint tensors between the caches and the store.

Rules the generator thread relies on (tests enforce them):

- The generator thread never opens, reads, writes, fsyncs or unlinks a file and never serializes a checkpoint.
  It issues tensor copies, dict lookups and non-blocking queue puts. It WAITS only when a starting job needs pages
  or a checkpoint from disk: those reads run on reader threads and the job cannot start without them.
- A tier failure never raises into the generator: the writer counts errors and disables writes after repeated
  failures; a read that fails its checksum is a miss.
- Chains are written proactively when a checkpoint is admitted: a few pages per iteration while jobs run, and by
  a background drain while the generator is idle, so a ``docker rm -f`` after the "drained" log line finds every
  admitted chain complete on disk.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from collections import defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import fcntl
import queue
import shutil
import struct
import threading
import time
import zlib

try:  # SegmentStore is importable in CPU-only test environments without torch.
    import torch
except ImportError:  # pragma: no cover
    torch = None

from ..constants import PAGE_SIZE

try:
    from ..tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
except Exception:  # pragma: no cover - CPU tests load this module without the tokenizer package
    FIRST_MM_EMBEDDING_INDEX = 1000000000


# Record format. Version 3: round 2's 128-byte header; checkpoint payloads are raw tensors (no pickle).
FORMAT_VERSION = 3
MAGIC = b"EXL3NV03"
ALIGNMENT = 4096
KIND_PAGE = 1
KIND_CHECKPOINT = 2
HEADER = struct.Struct("<8sBBHIQQQ16s16s16sI36x")  # exactly 128 bytes
NO_HASH = bytes(16)

# Space reclaim: a non-active segment is forwarded (live records copied to the active segment, file deleted)
# while its live fraction is below RECLAIM_MAX_LIVE_FRAC (rewrite <= 1x the reclaimed bytes). At the cap, after
# each eviction the least-live segment is forwarded once it is below RECLAIM_FORCE_LIVE_FRAC (rewrite <= 3x),
# which bounds how much live data eviction removes before physical bytes can drop.
RECLAIM_MAX_LIVE_FRAC = 0.5
RECLAIM_FORCE_LIVE_FRAC = 0.75
READ_FD_CACHE = 8
COPY_CHUNK = 16 * 1024**2
WRITER_MAX_CONSECUTIVE_ERRORS = 8


def _align(n: int, a: int = ALIGNMENT) -> int:
    return (n + a - 1) // a * a


def _hash16(value: bytes | None) -> bytes:
    if value is None:
        return NO_HASH
    if len(value) != 16:
        raise ValueError("page hashes must be 16 bytes")
    return value


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys = True, separators = (",", ":")).encode("utf8")


def _bytes_view(p) -> memoryview:
    mv = p if isinstance(p, memoryview) else memoryview(p)
    return mv if (mv.ndim == 1 and mv.format == "B") else mv.cast("B")


def _fadvise_dontneed(fd: int, offset: int, length: int):
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


@dataclass
class DiskEntry:
    kind: int
    key: bytes
    prev: bytes | None
    segment: int
    offset: int
    payload_len: int
    record_len: int
    serial: int
    created_ns: int
    last_access: int
    hits: int = 0


class _ReadFds:
    """Refcounted O_RDONLY fds per segment, shared by concurrent readers (pread never moves a shared position).
    Segment numbers are never reused, so a dropped (compacted-away) segment is simply gone: a reader that still
    holds its fd finishes on the unlinked file, a new acquire fails and the caller retries at the new location."""

    def __init__(self, capacity: int, opener):
        self.capacity = capacity
        self._opener = opener
        self._entries: dict[int, list] = {}  # segment -> [fd, refcount, stale]
        self._lock = threading.Lock()

    def acquire(self, number: int) -> int:
        with self._lock:
            e = self._entries.get(number)
            if e is not None and e[2]:
                raise FileNotFoundError(f"segment {number} was compacted away")
            if e is None:
                while len(self._entries) >= self.capacity:
                    for num, ent in self._entries.items():
                        if ent[1] == 0:
                            os.close(ent[0])
                            del self._entries[num]
                            break
                    else:
                        break
                e = [self._opener(number), 0, False]
                self._entries[number] = e
            e[1] += 1
            return e[0]

    def release(self, number: int):
        with self._lock:
            e = self._entries.get(number)
            if e is not None:
                e[1] -= 1
                if e[1] == 0 and e[2]:
                    os.close(e[0])
                    del self._entries[number]

    def drop(self, number: int):
        with self._lock:
            e = self._entries.get(number)
            if e is None:
                return
            e[2] = True
            if e[1] == 0:
                os.close(e[0])
                del self._entries[number]

    def close_all(self):
        with self._lock:
            for e in self._entries.values():
                try:
                    os.close(e[0])
                except OSError:
                    pass
            self._entries.clear()


class SegmentStore:
    """
    Append-only checksummed segment store with an in-memory hash-chain index.

    Threading: methods that write, truncate or unlink files (``store_page``, ``store_checkpoint``,
    ``reclaim_dirty``, ``close``) are called from one thread (the writer). Reads (``read_into``,
    ``fetch_checkpoint``) and index queries may run on any thread. ``self.lock`` guards the index and the byte
    accounting and is never held across file I/O, so readers and the generator thread never wait behind a write,
    an fsync or a compaction.

    Cap: ``disk_bytes()`` (namespace file + all segment bytes) is <= ``capacity_bytes`` whenever an append
    returns. While a segment is being forwarded it can exceed the cap by that segment's live bytes (less than
    RECLAIM_FORCE_LIVE_FRAC of one segment); segments are at most capacity / 8.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        namespace: dict,
        capacity_bytes: int,
        min_free_pct: float = 15.0,
        segment_bytes: int = 256 * 1024**2,
        admission_min_pages: int = 32,
        ttl_seconds: int = 0,
        free_space_fn = None,
        gc_stale_namespaces: bool = True,
    ):
        if capacity_bytes <= 0:
            raise ValueError("disk cache capacity must be positive")
        if not 0 <= min_free_pct < 100:
            raise ValueError("disk cache minimum free percentage must be in [0, 100)")
        if segment_bytes < ALIGNMENT:
            raise ValueError("segment size must be at least 4096 bytes")
        self.root = Path(root)
        self.root.mkdir(parents = True, exist_ok = True)
        self._lock_file = open(self.root / ".nvme-tier.lock", "a+b")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_file.close()
            raise RuntimeError(f"disk cache path is already owned by another process: {self.root}") from exc
        self.namespace = dict(namespace)
        self.namespace["tier_format_version"] = FORMAT_VERSION
        self.namespace_digest = hashlib.sha256(_json_bytes(self.namespace)).hexdigest()
        self.path = self.root / f"ns-{self.namespace_digest}"
        self.path.mkdir(mode = 0o700, parents = True, exist_ok = True)
        self.capacity_bytes = int(capacity_bytes)
        self.min_free_pct = float(min_free_pct)
        # Reclaim granularity is one segment: keep at least eight segments under the cap
        self.segment_bytes = max(ALIGNMENT, min(int(segment_bytes), _align(self.capacity_bytes // 8)))
        self.admission_min_pages = int(admission_min_pages)
        self.ttl_seconds = int(ttl_seconds)
        self.free_space_fn = free_space_fn or self._filesystem_free_pct
        self.lock = threading.RLock()

        self.pages: dict[bytes, DiskEntry] = {}
        self.checkpoints: dict[bytes, DiskEntry] = {}
        self.eligible: set[bytes] = set()
        self._children: dict[bytes, int] = defaultdict(int)
        self._leaves: set[bytes] = set()
        self._seg_phys: dict[int, int] = {}
        self._seg_live: dict[int, int] = {}
        self._phys_bytes = 0
        self._live_bytes_total = 0
        self._ns_bytes = 0
        self.serial = 0
        self.active_segment = 0
        self.active_offset = 0
        self._active_fd: int | None = None
        self._read_fds = _ReadFds(READ_FD_CACHE, self._open_read_fd)
        self.metrics = defaultdict(int)
        self.last_read_failure = None
        for k in (
            "page_writes", "checkpoint_writes", "bytes_written", "dedup_hits", "page_hits", "page_misses",
            "checkpoint_hits", "checkpoint_misses", "restored_bytes", "pruned_pages", "pruned_checkpoints",
            "evicted_interior_checkpoints", "stranded_pruned", "capacity_refusals", "free_space_refusals",
            "torn_records", "namespace_gc", "admission_rejects", "compactions", "compaction_rewrite_bytes",
            "compaction_reclaim_bytes", "read_retries", "quarantined", "write_verify_failures",
            "read_fail_open", "read_fail_header", "read_fail_identity", "read_fail_short", "read_fail_digest",
            "read_fail_io",
        ):
            self.metrics[k] = 0

        self._write_namespace()
        self._phys_bytes = self._ns_bytes
        if gc_stale_namespaces:
            self._gc_other_namespaces()
        self._rebuild()
        # At open there is no VRAM contribution to any chain: anchors whose page records are incomplete are
        # residue of a writer that was killed before it finished copying the chain out
        self.prune_stranded()
        self._expire()
        # A lowered cap applies at open: shrink before the first append
        if self._phys_bytes > self.capacity_bytes:
            self._ensure_room(0)

    # ---------- namespace and recovery ----------

    def _write_namespace(self):
        target = self.path / "namespace.json"
        content = _json_bytes({
            "digest": self.namespace_digest,
            "descriptor": self.namespace,
            "last_open_ns": time.time_ns(),
        }) + b"\n"
        tmp = self.path / "namespace.json.tmp"
        with open(tmp, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
        self._ns_bytes = target.stat().st_size

    def _gc_other_namespaces(self):
        for candidate in self.root.glob("ns-*"):
            if candidate == self.path or not candidate.is_dir():
                continue
            # Only directories this implementation created (a self-describing namespace.json) are removed
            marker = candidate / "namespace.json"
            try:
                data = json.loads(marker.read_text("utf8"))
            except (OSError, ValueError):
                continue
            if data.get("digest") != candidate.name[3:]:
                continue
            shutil.rmtree(candidate, ignore_errors = True)
            self.metrics["namespace_gc"] += 1

    def _segment_path(self, number: int) -> Path:
        return self.path / f"segment-{number:08d}.bin"

    def _pack_header(self, kind, payload_len, serial, created_ns, key, prev, digest, crc = 0):
        return HEADER.pack(
            MAGIC, FORMAT_VERSION, kind, 0, HEADER.size, payload_len, serial, created_ns,
            key, _hash16(prev), digest, crc,
        )

    def _decode_header(self, raw: bytes):
        if len(raw) != HEADER.size:
            return None
        values = HEADER.unpack(raw)
        magic, version, kind, flags, header_len, payload_len, serial, created_ns, key, prev, digest, crc = values
        if magic != MAGIC or version != FORMAT_VERSION or flags or header_len != HEADER.size:
            return None
        if kind not in (KIND_PAGE, KIND_CHECKPOINT):
            return None
        zeroed = self._pack_header(kind, payload_len, serial, created_ns, key,
                                   None if prev == NO_HASH else prev, digest, 0)
        if zlib.crc32(zeroed) & 0xFFFFFFFF != crc:
            return None
        return kind, payload_len, serial, created_ns, key, (None if prev == NO_HASH else prev), digest

    def _digest_range(self, fd: int, offset: int, length: int) -> bytes:
        hasher = hashlib.blake2b(digest_size = 16)
        done = 0
        while done < length:
            chunk = os.pread(fd, min(COPY_CHUNK, length - done), offset + done)
            if not chunk:
                return b""
            hasher.update(chunk)
            done += len(chunk)
        return hasher.digest()

    def _rebuild(self):
        """
        Re-index every valid record. Segments other than the newest were fsynced when they were rotated out, so
        only their headers are read (payload digests are verified on every read). The newest segment is the only
        one a killed writer can have torn, so its payloads are verified in full. The first invalid record of a
        segment truncates the segment there.
        """
        numbers = []
        for p in self.path.glob("segment-*.bin"):
            try:
                numbers.append(int(p.stem.split("-")[1]))
            except (IndexError, ValueError):
                continue
        numbers.sort()
        newest = numbers[-1] if numbers else None
        for number in numbers:
            path = self._segment_path(number)
            fd = os.open(path, os.O_RDONLY)
            try:
                size = os.fstat(fd).st_size
                offset = 0
                while offset < size:
                    decoded = self._decode_header(os.pread(fd, HEADER.size, offset))
                    if decoded is None:
                        self.metrics["torn_records"] += 1
                        break
                    kind, payload_len, serial, created_ns, key, prev, digest = decoded
                    record_len = _align(HEADER.size + payload_len)
                    if payload_len > self.capacity_bytes or offset + record_len > size:
                        self.metrics["torn_records"] += 1
                        break
                    if number == newest and self._digest_range(fd, offset + HEADER.size, payload_len) != digest:
                        self.metrics["torn_records"] += 1
                        break
                    entry = DiskEntry(kind, key, prev, number, offset, payload_len, record_len,
                                      serial, created_ns, serial)
                    table = self.pages if kind == KIND_PAGE else self.checkpoints
                    table[key] = entry  # a later copy (compaction forward) supersedes an earlier one
                    self.serial = max(self.serial, serial)
                    offset += record_len
                _fadvise_dontneed(fd, 0, size)
            finally:
                os.close(fd)
            self._seg_phys[number] = offset
            self._phys_bytes += offset
            if offset < size:
                with open(path, "r+b") as f:
                    f.truncate(offset)
                    f.flush()
                    os.fsync(f.fileno())
        if numbers:
            self.active_segment = newest
            self.active_offset = self._seg_phys.get(newest, 0)
        self._reindex_all()

    def _reindex_all(self):
        self._children = defaultdict(int)
        self._leaves = set()
        self._seg_live = {n: 0 for n in self._seg_phys}
        self._live_bytes_total = 0
        for e in self.pages.values():
            self._seg_live[e.segment] = self._seg_live.get(e.segment, 0) + e.record_len
            self._live_bytes_total += e.record_len
            if e.prev is not None:
                self._children[e.prev] += 1
        for key in self.pages:
            if not self._children.get(key):
                self._leaves.add(key)
        for e in self.checkpoints.values():
            self._seg_live[e.segment] = self._seg_live.get(e.segment, 0) + e.record_len
            self._live_bytes_total += e.record_len

    # ---------- accounting ----------

    def disk_bytes(self) -> int:
        """Physical namespace bytes, maintained incrementally."""
        return self._phys_bytes

    def disk_bytes_stat(self) -> int:
        """The same quantity from stat() on every file (tests and debugging)."""
        total = self._ns_bytes
        for p in self.path.glob("segment-*.bin"):
            total += p.stat().st_size
        return total

    def live_bytes(self) -> int:
        return self._live_bytes_total + self._ns_bytes

    def _filesystem_free_pct(self) -> float:
        s = os.statvfs(self.path)
        return 100.0 * s.f_bavail / s.f_blocks if s.f_blocks else 0.0

    # ---------- writer-thread file operations ----------

    def _rotate(self):
        if self._active_fd is not None:
            os.fsync(self._active_fd)
            os.close(self._active_fd)
            self._active_fd = None
        self.active_segment += 1
        self.active_offset = 0

    def _ensure_active_fd(self) -> int:
        if self._active_fd is None:
            path = self._segment_path(self.active_segment)
            existing = path.stat().st_size if path.exists() else 0
            if existing < self.active_offset:
                self.active_offset = existing
            self._active_fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
        return self._active_fd

    def _open_read_fd(self, number: int) -> int:
        return os.open(self._segment_path(number), os.O_RDONLY)

    def _append(self, kind: int, key: bytes, prev: bytes | None, payload) -> DiskEntry | None:
        parts = [_bytes_view(p) for p in (payload if isinstance(payload, (list, tuple)) else [payload])]
        payload_len = sum(p.nbytes for p in parts)
        record_len = _align(HEADER.size + payload_len)
        if record_len + self._ns_bytes > self.capacity_bytes:
            self.metrics["capacity_refusals"] += 1
            return None
        if self.free_space_fn() < self.min_free_pct:
            self.metrics["free_space_refusals"] += 1
            return None
        if not self._ensure_room(record_len):
            self.metrics["capacity_refusals"] += 1
            return None
        if self.active_offset and self.active_offset + record_len > self.segment_bytes:
            self._rotate()
        with self.lock:
            self.serial += 1
            serial = self.serial
        created = time.time_ns()
        offset = self.active_offset
        segment = self.active_segment
        hasher = hashlib.blake2b(digest_size = 16)
        for p in parts:
            hasher.update(p)
        digest = hasher.digest()
        header0 = self._pack_header(kind, payload_len, serial, created, key, prev, digest, 0)
        crc = zlib.crc32(header0) & 0xFFFFFFFF
        header = self._pack_header(kind, payload_len, serial, created, key, prev, digest, crc)
        pad = record_len - HEADER.size - payload_len
        buffers = [header] + parts + ([bytes(pad)] if pad else [])
        fd = self._ensure_active_fd()
        os.lseek(fd, offset, os.SEEK_SET)
        written = os.writev(fd, buffers)
        if written != record_len:
            # active_offset is unchanged, so the next append overwrites the partial record
            raise OSError(f"short write to disk cache segment ({written} of {record_len} bytes)")
        _fadvise_dontneed(fd, offset, record_len)
        with self.lock:
            self.active_offset = offset + record_len
            self._seg_phys[segment] = self._seg_phys.get(segment, 0) + record_len
            self._phys_bytes += record_len
            entry = DiskEntry(kind, key, prev, segment, offset, payload_len, record_len, serial, created, serial)
            if kind == KIND_PAGE:
                self._insert_page(entry)
            else:
                self._insert_checkpoint_entry(entry)
        self.metrics["bytes_written"] += record_len
        return entry

    def _insert_page(self, entry: DiskEntry):
        old = self.pages.get(entry.key)
        if old is not None:
            self._drop_page_entry(old, count = False)
        self.pages[entry.key] = entry
        self._seg_live[entry.segment] = self._seg_live.get(entry.segment, 0) + entry.record_len
        self._live_bytes_total += entry.record_len
        if entry.prev is not None:
            self._children[entry.prev] += 1
            self._leaves.discard(entry.prev)
        if not self._children.get(entry.key):
            self._leaves.add(entry.key)

    def _insert_checkpoint_entry(self, entry: DiskEntry):
        old = self.checkpoints.get(entry.key)
        if old is not None:
            self._drop_checkpoint_entry(old, count = False)
        self.checkpoints[entry.key] = entry
        self._seg_live[entry.segment] = self._seg_live.get(entry.segment, 0) + entry.record_len
        self._live_bytes_total += entry.record_len

    def _drop_checkpoint_entry(self, entry: DiskEntry, count: bool = True):
        if self.checkpoints.get(entry.key) is not entry:
            return
        del self.checkpoints[entry.key]
        if entry.segment in self._seg_live:
            self._seg_live[entry.segment] -= entry.record_len
        self._live_bytes_total -= entry.record_len
        if count:
            self.metrics["pruned_checkpoints"] += 1

    def _drop_page_entry(self, entry: DiskEntry, count: bool = True):
        """Remove one page from the index, at any position in its chain. Its children keep their count under the
        removed key, so a re-inserted page is not mistaken for a leaf."""
        if self.pages.get(entry.key) is not entry:
            return
        del self.pages[entry.key]
        if entry.segment in self._seg_live:
            self._seg_live[entry.segment] -= entry.record_len
        self._live_bytes_total -= entry.record_len
        self._leaves.discard(entry.key)
        if entry.prev is not None:
            c = self._children.get(entry.prev, 0) - 1
            if c > 0:
                self._children[entry.prev] = c
            else:
                self._children.pop(entry.prev, None)
                if entry.prev in self.pages:
                    self._leaves.add(entry.prev)
        if count:
            self.metrics["pruned_pages"] += 1

    def _ensure_room(self, incoming: int) -> bool:
        """
        Make room for one record under the cap (writer thread). Index mutations take the lock briefly; file I/O
        (compaction) runs outside it. Order: forward a segment whose live fraction is below
        RECLAIM_MAX_LIVE_FRAC; otherwise evict one victim (interior checkpoints first, then radix leaves) and
        forward the least-live segment once it is below RECLAIM_FORCE_LIVE_FRAC.
        """
        if self._phys_bytes + incoming <= self.capacity_bytes:
            return True
        victims = None
        guard = 0
        limit = len(self.pages) + len(self.checkpoints) + 2 * len(self._seg_phys) + 64
        while self._phys_bytes + incoming > self.capacity_bytes:
            guard += 1
            if guard > limit:
                return False
            number = self._pick_segment(RECLAIM_MAX_LIVE_FRAC)
            if number is not None:
                self._compact_forward(number)
                continue
            if victims is None:
                victims = self._victims()
            victim = next(victims, None)
            if victim is None:
                number = self._pick_segment(1.0, require_dead = True)
                if number is None:
                    return False
                self._compact_forward(number)
                continue
            with self.lock:
                self._evict(victim)
            number = self._pick_segment(RECLAIM_FORCE_LIVE_FRAC)
            if number is not None:
                self._compact_forward(number)
        return True

    def _pick_segment(self, max_live_frac: float, require_dead: bool = False) -> int | None:
        """Non-active segment with the lowest live fraction below max_live_frac (ties: most dead bytes)."""
        best, best_key = None, None
        with self.lock:
            for number, phys in self._seg_phys.items():
                if number == self.active_segment or phys == 0:
                    continue
                live = self._seg_live.get(number, 0)
                frac = live / phys
                if frac >= max_live_frac or (require_dead and live >= phys):
                    continue
                key = (frac, -(phys - live))
                if best_key is None or key < best_key:
                    best, best_key = number, key
        return best

    def _superseded_checkpoints(self) -> set[bytes]:
        """Checkpoints with a descendant checkpoint on disk (interior points of a stored conversation). Each stored
        page is walked at most once."""
        sup = set()
        visited = set()
        for key in self.checkpoints:
            anchor = self.pages.get(key)
            h = anchor.prev if anchor is not None else None
            while h is not None and h not in visited:
                visited.add(h)
                if h in self.checkpoints:
                    sup.add(h)
                e = self.pages.get(h)
                h = e.prev if e is not None else None
        return sup

    def _victims(self):
        with self.lock:
            sup = self._superseded_checkpoints()
            order = sorted((self.checkpoints[k] for k in sup), key = lambda e: (e.last_access, e.serial))
        for e in order:
            yield ("cp", e)
        while True:
            with self.lock:
                key = self._choose_leaf_key()
                e = self.pages.get(key) if key is not None else None
            if e is None:
                return
            yield ("page", e)

    def _evict(self, victim):
        kind, e = victim
        if kind == "cp":
            if self.checkpoints.get(e.key) is e:
                self._drop_checkpoint_entry(e)
                self.metrics["evicted_interior_checkpoints"] += 1
        else:
            self._remove_leaf_index_only(e)

    def _compact_forward(self, number: int):
        """
        Forward a non-active segment's live records verbatim to the active segment and delete the source. The
        index is snapshotted under the lock, bytes are copied outside it, and the new locations are committed under
        the lock before the unlink: a reader holding an old location either reads the still-open file or fails its
        header check and retries at the new location. The destination is fsynced before the unlink, so a crash in
        between leaves both copies and the rebuild keeps the later one.
        """
        path = self._segment_path(number)
        if number == self.active_segment:
            return
        with self.lock:
            snapshot = sorted(
                ((e, e.offset) for t in (self.pages, self.checkpoints) for e in t.values() if e.segment == number),
                key = lambda x: x[1],
            )
            phys = self._seg_phys.get(number, 0)
        live = sum(e.record_len for e, _ in snapshot)
        rewritten = 0
        if live and path.exists():
            if self.active_offset and self.active_offset + live > self.segment_bytes:
                self._rotate()
            fd = self._ensure_active_fd()
            start = self.active_offset
            src_fd = os.open(path, os.O_RDONLY)
            try:
                pos = start
                for e, off in snapshot:
                    self._copy_range(src_fd, off, fd, pos, e.record_len)
                    pos += e.record_len
                _fadvise_dontneed(src_fd, 0, phys)
            finally:
                os.close(src_fd)
            os.fsync(fd)
            _fadvise_dontneed(fd, start, live)
            with self.lock:
                cursor = start
                moved = 0
                for e, off in snapshot:
                    table = self.pages if e.kind == KIND_PAGE else self.checkpoints
                    if table.get(e.key) is e and e.segment == number and e.offset == off:
                        e.segment = self.active_segment
                        e.offset = cursor
                        moved += e.record_len
                    cursor += e.record_len
                self.active_offset = start + live
                self._phys_bytes += live
                self._seg_phys[self.active_segment] = self._seg_phys.get(self.active_segment, 0) + live
                self._seg_live[self.active_segment] = self._seg_live.get(self.active_segment, 0) + moved
            rewritten = live
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        with self.lock:
            self._phys_bytes -= phys
            self._seg_phys.pop(number, None)
            self._seg_live.pop(number, None)
        self._read_fds.drop(number)
        self.metrics["compactions"] += 1
        self.metrics["compaction_rewrite_bytes"] += rewritten
        self.metrics["compaction_reclaim_bytes"] += phys - rewritten

    @staticmethod
    def _copy_range(src_fd: int, src_off: int, dst_fd: int, dst_off: int, length: int):
        done = 0
        cfr = getattr(os, "copy_file_range", None)
        while done < length:
            n = 0
            if cfr is not None:
                try:
                    n = cfr(src_fd, dst_fd, length - done, src_off + done, dst_off + done)
                except OSError:
                    cfr = None
                    n = 0
            if n <= 0:
                buf = os.pread(src_fd, min(COPY_CHUNK, length - done), src_off + done)
                if not buf:
                    raise IOError("short read while compacting disk cache")
                n = os.pwrite(dst_fd, buf, dst_off + done)
            done += n

    def reclaim_dirty(self, budget: int = 1):
        """Forward up to `budget` mostly-dead segments (writer thread, when its queue is empty)."""
        for _ in range(budget):
            number = self._pick_segment(RECLAIM_MAX_LIVE_FRAC)
            if number is None:
                return
            with self.lock:
                dead = self._seg_phys.get(number, 0) - self._seg_live.get(number, 0)
            if dead < self.segment_bytes // 8:
                return
            self._compact_forward(number)

    # ---------- radix index, admission and pruning ----------

    def admit_chain(self, hashes: list[bytes]) -> bool:
        with self.lock:
            if len(hashes) < self.admission_min_pages:
                self.metrics["admission_rejects"] += 1
                return False
            self.eligible.update(hashes)
            return True

    def should_store_page(self, key: bytes) -> bool:
        return key in self.eligible and key not in self.pages

    def store_page(self, key: bytes, prev: bytes | None, payload, force: bool = False, verify: bool = False) -> bool:
        with self.lock:
            if key in self.pages:
                self.metrics["dedup_hits"] += 1
                return True
            if not force and key not in self.eligible:
                self.metrics["admission_rejects"] += 1
                return False
        entry = self._append(KIND_PAGE, key, prev, payload)
        if entry is None:
            return False
        if verify and not self._verified(entry, self.pages, self._drop_page_entry):
            return False
        self.metrics["page_writes"] += 1
        return True

    def _verified(self, entry: DiskEntry, table: dict, drop) -> bool:
        """Read-back check of a record just appended: the file must hold exactly the bytes that were hashed."""
        why = self.verify_record(entry)
        if not why:
            return True
        self.metrics["write_verify_failures"] += 1
        with self.lock:
            drop(entry)
        if self.metrics["write_verify_failures"] <= 10:
            print(f" -- nvme tier: write verify failed for {'page' if table is self.pages else 'ckpt'} "
                  f"{entry.key[:4].hex()} at segment {entry.segment} offset {entry.offset}: '{why}'", flush = True)
        return False

    def store_checkpoint(self, key: bytes, chain: list[bytes], payload, verify: bool = False) -> bool:
        with self.lock:
            if key in self.checkpoints:
                self.metrics["dedup_hits"] += 1
                self.eligible.update(chain)
                return True
            if not chain or chain[-1] != key:
                return False
            if len(chain) < self.admission_min_pages:
                self.metrics["admission_rejects"] += 1
                return False
            self.eligible.update(chain)
        prev = chain[-2] if len(chain) > 1 else None
        entry = self._append(KIND_CHECKPOINT, key, prev, payload)
        if entry is None:
            return False
        if verify and not self._verified(entry, self.checkpoints, self._drop_checkpoint_entry):
            return False
        self.metrics["checkpoint_writes"] += 1
        return True

    def prev_of(self, key: bytes) -> bytes | None:
        entry = self.pages.get(key)
        return entry.prev if entry is not None else None

    def _chain_complete(self, anchor: bytes) -> bool:
        h = anchor
        seen = set()
        while h is not None:
            if h in seen:
                return False
            seen.add(h)
            entry = self.pages.get(h)
            if entry is None:
                return False
            h = entry.prev
        return True

    def has_checkpoint(self, key: bytes) -> bool:
        return key in self.checkpoints

    def has_resumable_checkpoint(self, key: bytes) -> bool:
        with self.lock:
            return key in self.checkpoints and self._chain_complete(key)

    def longest_resumable_prefix(self, hashes: list[bytes]) -> int:
        """Pages through the deepest checkpoint whose whole chain is on disk, O(prefix pages)."""
        with self.lock:
            deepest = 0
            expected_prev = None
            for i, key in enumerate(hashes):
                entry = self.pages.get(key)
                if entry is None or entry.prev != expected_prev:
                    break
                expected_prev = key
                if key in self.checkpoints:
                    deepest = i + 1
            return deepest

    def _choose_leaf_key(self) -> bytes | None:
        best_key, best_rank = None, None
        for key in self._leaves:
            e = self.pages.get(key)
            if e is None:
                continue
            rank = (e.last_access, e.serial)
            if best_rank is None or rank < best_rank:
                best_key, best_rank = key, rank
        return best_key

    def _remove_leaf_index_only(self, entry: DiskEntry):
        """Logical eviction of one radix leaf (index only). A checkpoint anchored at it goes with it."""
        entry = self.pages.get(entry.key)
        if entry is not None:
            self._drop_page_entry(entry)
            cp = self.checkpoints.get(entry.key)
            if cp is not None:
                self._drop_checkpoint_entry(cp)

    def prune_one_leaf(self) -> bytes | None:
        with self.lock:
            key = self._choose_leaf_key()
            if key is None:
                return None
            victim = self.pages[key]
            self._remove_leaf_index_only(victim)
            return victim.key

    def prune_stranded(self) -> tuple[int, int]:
        """Drop anchors whose chain is incomplete on disk, then pages that lead to no surviving anchor, tail-first.
        Only valid when no chain is being copied out (at open)."""
        with self.lock:
            bad_cp = [key for key in self.checkpoints if not self._chain_complete(key)]
            for key in bad_cp:
                self._drop_checkpoint_entry(self.checkpoints[key])
            anchored = set()
            for key in self.checkpoints:
                h = key
                while h is not None and h not in anchored:
                    anchored.add(h)
                    entry = self.pages.get(h)
                    h = entry.prev if entry is not None else None
            stranded = set(self.pages) - anchored
            page_count = 0
            while stranded:
                leaves = [self.pages[h] for h in stranded if not self._children.get(h)]
                if not leaves:
                    for h in list(stranded):  # a hash cycle cannot be emptied leaf-first
                        self._drop_page_entry(self.pages[h])
                        page_count += 1
                    break
                for victim in sorted(leaves, key = lambda e: (e.last_access, e.serial)):
                    stranded.discard(victim.key)
                    self._drop_page_entry(victim)
                    page_count += 1
            self.metrics["stranded_pruned"] += len(bad_cp) + page_count
            self._rebuild_eligible()
            return page_count, len(bad_cp)

    def _rebuild_eligible(self):
        self.eligible.clear()
        for anchor in self.checkpoints:
            h = anchor
            walked = set()
            while h is not None and h not in walked:
                walked.add(h)
                self.eligible.add(h)
                entry = self.pages.get(h)
                if entry is None:
                    break
                h = entry.prev

    def _expire(self):
        if not self.ttl_seconds:
            return
        cutoff = time.time_ns() - self.ttl_seconds * 1_000_000_000
        with self.lock:
            expired = [e for e in self.checkpoints.values() if e.created_ns < cutoff]
            for e in expired:
                self._drop_checkpoint_entry(e)
        if expired:
            self.prune_stranded()

    # ---------- reads (any thread; the lock only guards the location lookup) ----------

    def _locate(self, table: dict, key: bytes):
        with self.lock:
            e = table.get(key)
            return (e, e.segment, e.offset) if e is not None else None

    def _read_at(self, e: DiskEntry, segment: int, offset: int, buf) -> str:
        """Read one record's payload into buf and verify it. Returns "" on success, else the failed check:
        "open", "header", "identity", "short", "digest" or "io"."""
        try:
            fd = self._read_fds.acquire(segment)
        except OSError:
            return "open"
        try:
            decoded = self._decode_header(os.pread(fd, HEADER.size, offset))
            if decoded is None:
                return "header"
            kind, payload_len, _, _, key, _, digest = decoded
            if kind != e.kind or key != e.key or payload_len != e.payload_len:
                return "identity"
            mv = _bytes_view(buf)[:payload_len]
            got = 0
            while got < payload_len:
                n = os.preadv(fd, [mv[got:]], offset + HEADER.size + got)
                if n <= 0:
                    return "short"
                got += n
            if hashlib.blake2b(mv, digest_size = 16).digest() != digest:
                return "digest"
            _fadvise_dontneed(fd, offset, e.record_len)
            return ""
        except OSError:
            return "io"
        finally:
            self._read_fds.release(segment)

    def verify_record(self, e: DiskEntry) -> str:
        """Re-read a record just written and check its digest (writer thread). "" = the file holds what was hashed."""
        buf = bytearray(e.payload_len)
        return self._read_at(e, e.segment, e.offset, buf)

    def _read_record(self, table: dict, key: bytes, buf_fn):
        """Locate and read one record. A record moved by compaction mid-read is re-read at its new location; one
        that fails twice at the same location is corrupt and leaves the index (with a checkpoint anchored on it)."""
        last = None
        buf = None
        for _ in range(3):
            loc = self._locate(table, key)
            if loc is None:
                return None, None
            if buf is None:
                buf = buf_fn(loc[0].payload_len)
            why = self._read_at(loc[0], loc[1], loc[2], buf)
            if not why:
                return loc[0], buf
            self.metrics["read_retries"] += 1
            self.metrics["read_fail_" + why] += 1
            self.last_read_failure = (
                "page" if table is self.pages else "ckpt", key[:4].hex(), loc[1], loc[2], why
            )
            if last is not None and (last[0] is loc[0] and last[1:] == loc[1:]):
                break
            last = loc
        with self.lock:
            e2 = table.get(key)
            if e2 is not None and last is not None and e2 is last[0] and e2.segment == last[1] and e2.offset == last[2]:
                if table is self.pages:
                    self._drop_page_entry(e2)
                    cp = self.checkpoints.get(key)
                    if cp is not None:
                        self._drop_checkpoint_entry(cp)
                else:
                    self._drop_checkpoint_entry(e2)
                self.metrics["quarantined"] += 1
                if self.metrics["quarantined"] <= 10:
                    kind, k8, seg, off, why = self.last_read_failure
                    print(f" -- nvme tier: quarantined {kind} {k8} at segment {seg} offset {off}: read check "
                          f"'{why}' failed twice", flush = True)
        return None, None

    def read_into(self, key: bytes, buf) -> int | None:
        """Read a stored page payload into buf (writable, at least the payload length)."""
        e, _ = self._read_record(self.pages, key, lambda n: buf)
        if e is None:
            self.metrics["page_misses"] += 1
            return None
        self._mark_hit(e)
        self.metrics["page_hits"] += 1
        self.metrics["restored_bytes"] += e.payload_len
        return e.payload_len

    def fetch_page(self, key: bytes) -> bytes | None:
        e, buf = self._read_record(self.pages, key, bytearray)
        if e is None:
            self.metrics["page_misses"] += 1
            return None
        self._mark_hit(e)
        self.metrics["page_hits"] += 1
        self.metrics["restored_bytes"] += e.payload_len
        return bytes(buf)

    def fetch_checkpoint(self, key: bytes) -> bytearray | None:
        e, buf = self._read_record(self.checkpoints, key, bytearray)
        if e is None:
            self.metrics["checkpoint_misses"] += 1
            return None
        self._mark_hit(e)
        self.metrics["checkpoint_hits"] += 1
        self.metrics["restored_bytes"] += e.payload_len
        return buf

    def _mark_hit(self, entry: DiskEntry):
        with self.lock:
            entry.hits += 1
            self.serial += 1
            entry.last_access = self.serial

    def close(self):
        if self._active_fd is not None:
            try:
                os.fsync(self._active_fd)
                os.close(self._active_fd)
            except OSError:
                pass
            self._active_fd = None
        self._read_fds.close_all()
        if self._lock_file is not None:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None

    def __contains__(self, key: bytes):
        return key in self.pages

    def __len__(self):
        return len(self.pages)


# ----------------------------------------------------------------------------------------------------------------
# Identity
# ----------------------------------------------------------------------------------------------------------------

def _checkpoint_identity(directory: str) -> dict:
    """Model identity without mtime (an rsync or touch of the model directory must not wipe the tier): names, sizes
    and sampled head/tail bytes of the weights, JSON configs in full."""
    root = Path(directory)
    digest = hashlib.sha256()
    files = []
    for pattern in ("config.json", "quantization_config.json", "*.index.json", "*.safetensors"):
        files.extend(root.glob(pattern))
    for p in sorted(set(files), key = lambda x: x.name):
        try:
            st = p.stat()
            digest.update(p.name.encode("utf8"))
            digest.update(struct.pack("<Q", st.st_size))
            with open(p, "rb") as f:
                if p.name.endswith(".json"):
                    digest.update(f.read())
                else:
                    digest.update(f.read(65536))
                    if st.st_size > 65536:
                        f.seek(max(0, st.st_size - 65536))
                        digest.update(f.read(65536))
        except OSError:
            continue
    return {"directory": str(root.resolve()), "sampled_sha256": digest.hexdigest()}


_ENGINE_SUFFIXES = (".py", ".cu", ".cuh", ".cpp", ".h", ".c")
_ENV_EXCLUDE_PREFIXES = ("EXL3_NVME_TIER",)


def engine_identity(package_dir: str | os.PathLike | None = None, environ = None) -> dict:
    """
    What else decides the bytes of a page or checkpoint for given tokens: the engine's sources (Python, and the
    CUDA/C++ sources the extension is built from) and every EXL3_* switch (kernel selections such as the E3
    prefill change numerics). A different image or flag set is a different namespace, and the old namespace is
    removed at open. This file is excluded (its record format is versioned), as are the EXL3_NVME_TIER* knobs.
    """
    root = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parents[1]
    me = Path(__file__).resolve()
    digest = hashlib.sha256()
    count = 0
    for p in sorted(root.rglob("*")):
        if p.suffix not in _ENGINE_SUFFIXES or "__pycache__" in p.parts or not p.is_file():
            continue
        if p.resolve() == me:
            continue
        try:
            digest.update(str(p.relative_to(root)).encode("utf8"))
            digest.update(hashlib.sha256(p.read_bytes()).digest())
            count += 1
        except OSError:
            continue
    env = os.environ if environ is None else environ
    flags = sorted(
        [k, v] for k, v in env.items()
        if k.startswith("EXL3_") and not k.startswith(_ENV_EXCLUDE_PREFIXES)
    )
    return {"sources_sha256": digest.hexdigest(), "source_files": count, "env": flags}


# ----------------------------------------------------------------------------------------------------------------
# Checkpoint payloads: raw tensor bytes behind a JSON table (no pickle, no torch.save)
# ----------------------------------------------------------------------------------------------------------------

_CP_PREFIX = struct.Struct("<4sI")
_CP_MAGIC = b"RCP1"
_CP_ALIGN = 64


def _enc_key(k):
    if isinstance(k, bool) or k is None:
        raise TypeError(f"unsupported checkpoint key {k!r}")
    if isinstance(k, str):
        return ["s", k]
    if isinstance(k, int):
        return ["i", k]
    if isinstance(k, tuple):
        return ["t", [_enc_key(x) for x in k]]
    raise TypeError(f"unsupported checkpoint key {k!r}")


def _dec_key(v):
    tag, val = v
    if tag == "s":
        return val
    if tag == "i":
        return int(val)
    if tag == "t":
        return tuple(_dec_key(x) for x in val)
    raise ValueError(f"bad checkpoint key tag {tag!r}")


def serialize_stash(stashed: dict) -> list:
    """Payload parts of a stash dict ({"position": int, "checkpoint_size": int, layer key: (tensor, ...)}): a
    prefix + JSON table, then each tensor's raw bytes, 64-byte aligned. Tensor parts are views of the stash's own
    CPU memory, so nothing is copied before the write."""
    if "tp_handle" in stashed:
        raise TypeError("tensor-parallel checkpoints live in the workers and cannot be persisted")
    scalars = []
    tensors = []
    blobs = []
    offset = 0
    for k, v in stashed.items():
        if isinstance(v, (int, float, str)) and not isinstance(v, bool):
            scalars.append([_enc_key(k), v])
            continue
        single = not isinstance(v, (tuple, list))
        group = (v,) if single else v
        entries = []
        for t in group:
            if not isinstance(t, torch.Tensor) or t.device.type != "cpu":
                raise TypeError(f"checkpoint entry {k!r} is not a group of CPU tensors")
            t = t.contiguous()
            nbytes = t.numel() * t.element_size()
            entries.append([str(t.dtype).replace("torch.", ""), list(t.shape), offset, nbytes])
            blobs.append((offset, t, nbytes))
            offset = _align(offset + nbytes, _CP_ALIGN)
        tensors.append([_enc_key(k), entries, single])
    table = _json_bytes({"scalars": scalars, "tensors": tensors, "data_bytes": offset})
    head_len = _align(_CP_PREFIX.size + len(table), _CP_ALIGN)
    head = bytearray(head_len)
    _CP_PREFIX.pack_into(head, 0, _CP_MAGIC, len(table))
    head[_CP_PREFIX.size:_CP_PREFIX.size + len(table)] = table
    parts = [head]
    cursor = 0
    for off, t, nbytes in blobs:
        if off > cursor:
            parts.append(bytes(off - cursor))
        if nbytes:
            parts.append(t.reshape(-1).view(torch.uint8).numpy())
        cursor = off + nbytes
    if offset > cursor:
        parts.append(bytes(offset - cursor))
    return parts


def deserialize_stash(buf) -> dict:
    """Inverse of serialize_stash. The tensors are views of `buf` (which they keep alive): bytes in == bytes out."""
    mv = memoryview(buf).cast("B")
    magic, table_len = _CP_PREFIX.unpack_from(mv, 0)
    if magic != _CP_MAGIC:
        raise ValueError("not a recurrent checkpoint payload")
    table = json.loads(bytes(mv[_CP_PREFIX.size:_CP_PREFIX.size + table_len]))
    base = _align(_CP_PREFIX.size + table_len, _CP_ALIGN)
    if base + table["data_bytes"] > len(mv):
        raise ValueError("truncated recurrent checkpoint payload")
    out = {}
    for k, v in table["scalars"]:
        out[_dec_key(k)] = v
    for k, entries, single in table["tensors"]:
        group = []
        for dtype, shape, off, nbytes in entries:
            dt = getattr(torch, dtype)
            if nbytes:
                t = torch.frombuffer(buf, dtype = torch.uint8, count = nbytes, offset = base + off)
                t = t.view(dt).view(shape)
            else:
                t = torch.empty(shape, dtype = dt)
            group.append(t)
        out[_dec_key(k)] = group[0] if single else tuple(group)
    return out


def own_stash_tensors(stashed: dict) -> int:
    """
    Replace, in place, every CPU tensor of a stash dict that is a view of other memory by a copy; returns how many.

    A layer's stash() is expected to return a snapshot, and for device-resident state `.cpu()` is one. For state
    that lives on the host it is not: `.cpu()` of a CPU tensor returns the tensor itself. PLELayerState keeps its
    carried token-id context on the CPU (modules/ple.py alloc) and stashes `id_state[slot, :ctx].cpu()`
    (modules/ple.py stash), i.e. a live view of the slot, which every later forward on that slot overwrites in
    place (modules/ple.py forward: `id_state[s, :ctx].copy_(...)`). A checkpoint holding that view does not hold
    the checkpoint-time context, and a writer hashing then writing it races the next forward (R526: every
    persisted checkpoint failed its digest on read). Tensors that own their storage (the `.cpu()` copies of GPU
    state, the tip round's snapshots, deserialized checkpoints marked tier-owned) are left alone.
    """
    n = 0
    for k, v in list(stashed.items()):
        single = isinstance(v, torch.Tensor)
        if not single and not isinstance(v, tuple):
            continue
        group = (v,) if single else v
        fixed = []
        changed = False
        for t in group:
            if isinstance(t, torch.Tensor) and t.device.type == "cpu" and t._base is not None:
                t = t.clone()
                changed = True
                n += 1
            fixed.append(t)
        if changed:
            stashed[k] = fixed[0] if single else tuple(fixed)
    return n


class _StashedState:
    """Duck-typed recurrent state for RecurrentCache.put(): put() only calls .stash(); the tip policy reads
    .position and .checkpoint_size. Its tensors are views of a buffer read from disk that nothing else writes."""

    tier_owned = True

    def __init__(self, stashed: dict):
        self.stashed = stashed
        self.position = stashed["position"]
        self.checkpoint_size = stashed["checkpoint_size"]

    def stash(self):
        return self.stashed


# ----------------------------------------------------------------------------------------------------------------
# Staging slabs
# ----------------------------------------------------------------------------------------------------------------

class SlabPool:
    """Bounded pool of pinned staging slabs. A transfer leaves CUDA events on its slab; the next acquirer waits on
    them only when it reuses that slab. try_acquire() never waits."""

    def __init__(self, slab_size: int, count: int, view_builder):
        self.slab_size = slab_size
        self._view_builder = view_builder
        self._free: deque = deque()
        self._recycle: deque = deque()
        self._cond = threading.Condition()
        self.unpinned_fallback = False
        self.count = count
        for _ in range(count):
            self._free.append(self._make())

    def _make(self):
        try:
            slab = torch.empty((self.slab_size,), dtype = torch.uint8, pin_memory = True)
        except (RuntimeError, AssertionError):
            # No CUDA (CPU tests): pageable memory, copies are synchronous, the record format is unaffected
            self.unpinned_fallback = True
            slab = torch.empty((self.slab_size,), dtype = torch.uint8, pin_memory = False)
        return slab, self._view_builder(slab)

    def _collect(self):
        while self._recycle:
            slab, views, events = self._recycle[0]
            if all(e.query() for e in events):
                self._recycle.popleft()
                self._free.append((slab, views))
            else:
                break

    def acquire(self):
        """Blocking acquire (reader threads only)."""
        with self._cond:
            while True:
                self._collect()
                if self._free:
                    return self._free.popleft()
                if self._recycle:
                    slab, views, events = self._recycle.popleft()
                    self._cond.release()
                    try:
                        for e in events:
                            e.synchronize()
                    finally:
                        self._cond.acquire()
                    return slab, views
                self._cond.wait(0.05)

    def try_acquire(self):
        with self._cond:
            self._collect()
            if self._free:
                return self._free.popleft()
            return None

    def release(self, slab, views, events = None):
        events = list(events or [])
        with self._cond:
            if events and not all(e.query() for e in events):
                self._recycle.append((slab, views, events))
            else:
                self._free.append((slab, views))
            self._cond.notify_all()

    def available(self) -> int:
        with self._cond:
            return len(self._free) + len(self._recycle)


# ----------------------------------------------------------------------------------------------------------------
# Restore batch: bounded parallel page reads that live only inside one PageTable.allocate_pages call
# ----------------------------------------------------------------------------------------------------------------

class _RestoreBatch:

    def __init__(self, tier: "DiskPageCache", keys: list[bytes]):
        self.tier = tier
        self.keys = keys
        self.keyset = set(keys)
        self.futures: dict[bytes, object] = {}
        self.next = 0
        self.window = tier._read_window
        self._refill()

    def __contains__(self, key):
        return key in self.keyset

    def _submit_next(self):
        k = self.keys[self.next]
        self.next += 1
        self.futures[k] = self.tier._read_pool.submit(self.tier._read_page_task, k)

    def _refill(self):
        while len(self.futures) < self.window and self.next < len(self.keys):
            self._submit_next()

    def take(self, key: bytes, page_index: int) -> dict | None:
        """Wait for one page's read, copy it into VRAM page `page_index` (async H2D on the current stream) and
        return its token ids, or None (miss)."""
        if key not in self.keyset:
            return None
        while key not in self.futures and self.next < len(self.keys):
            self._submit_next()
        fut = self.futures.pop(key, None)
        if fut is None:
            return None
        self._refill()
        try:
            result = fut.result()
        except Exception:
            result = None
        if result is None:
            self.tier.metrics["restore_failures"] += 1
            return None
        slab, views, tokens = result
        for view, (tensor, _, _, _) in zip(views, self.tier.segments):
            tensor[page_index].copy_(view, non_blocking = True)
        self.tier._pool.release(slab, views, self.tier._events())
        self.tier.metrics["restored_pages"] += 1
        return {"tokens": tokens}

    def finish(self):
        """Release every read this allocation did not consume."""
        for fut in self.futures.values():
            if fut.cancel():
                continue
            try:
                result = fut.result()
            except Exception:
                result = None
            if result is not None:
                self.tier._pool.release(result[0], result[1], None)
        self.futures.clear()
        self.next = len(self.keys)


# ----------------------------------------------------------------------------------------------------------------
# Torch adapter
# ----------------------------------------------------------------------------------------------------------------

def _env_num(name, default, cast):
    try:
        return cast(os.environ.get(name, default))
    except ValueError:
        return default


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class DiskPageCache:
    """
    Persists exact target+draft page images and recurrent checkpoints.

    Generator-thread entry points (pump, on_busy, on_idle, store, persist_checkpoint, on_checkpoint_evicted,
    prepare_resume, begin_restore/end_restore) never do file I/O or serialization and never wait on the writer.
    """

    TOKEN_PREFIX = struct.Struct("<I")
    TOKEN_BUDGET = PAGE_SIZE * 8

    @classmethod
    def install(cls, generator, cache, draft_cache, path: str):
        """Build the tier for a Generator from the EXL3_NVME_TIER* settings, or print why it stays off."""
        tag = " -- nvme tier:"
        if generator.recurrent_cache is None:
            print(f"{tag} disabled (model has no recurrent states)", flush = True)
            return None
        caches = [cache] + ([draft_cache] if draft_cache is not None else [])
        if any(getattr(c.model, "loaded_tp", False) for c in caches):
            print(f"{tag} disabled (tensor-parallel caches are not supported)", flush = True)
            return None
        try:
            tier = cls(
                caches,
                path,
                max_size = int(_env_num("EXL3_NVME_TIER_GB", 128, float) * 1024**3),
                min_free_pct = _env_num("EXL3_NVME_TIER_MIN_FREE_PCT", 15.0, float),
                segment_size = _env_num("EXL3_NVME_TIER_SEGMENT_MB", 256, int) * 1024**2,
                staging_size = _env_num("EXL3_NVME_TIER_STAGING_MB", 512, int) * 1024**2,
                admission_min_pages = _env_num("EXL3_NVME_TIER_MIN_PAGES", 32, int),
                ttl_seconds = int(_env_num("EXL3_NVME_TIER_TTL_HOURS", 0, float) * 3600),
                pump_pages = _env_num("EXL3_NVME_TIER_PUMP_PAGES", 4, int),
                pump_pct = _env_num("EXL3_NVME_TIER_PUMP_PCT", 1.0, float),
                verify = _env_num("EXL3_NVME_TIER_VERIFY", 1, int),
                log_lookups = _env_num("EXL3_NVME_TIER_LOG_LOOKUPS", 1, int) != 0,
                scan_checkpoints = _env_num("EXL3_NVME_TIER_SCAN", 1, int) != 0,
                extra_identity = {
                    "max_chunk_size": generator.max_chunk_size,
                    "recurrent_checkpoint_interval": generator.recurrent_checkpoint_interval,
                    "recurrent_checkpoint_interval_pp": generator.recurrent_checkpoint_interval_pp,
                },
                log_secs = _env_num("EXL3_NVME_TIER_LOG_SECS", 60.0, float),
            )
        except Exception as e:
            print(f"{tag} disabled ({type(e).__name__}: {e})", flush = True)
            return None
        tier.attach(generator.pagetable, generator.recurrent_cache)
        generator.pagetable.disk_tier = tier
        generator.recurrent_cache.disk_tier = tier
        b = tier.backend
        print(
            f"{tag} ON at {b.path} (cap {b.capacity_bytes / 1024**3:.1f} GiB, on disk "
            f"{b.disk_bytes() / 1024**3:.2f} GiB: {len(b.pages)} pages, {len(b.checkpoints)} checkpoints, "
            f"{b.metrics['torn_records']} torn, {b.metrics['stranded_pruned']} stranded pruned, "
            f"{b.metrics['namespace_gc']} stale namespaces removed; page image {tier.slab_size / 1024**2:.2f} MiB, "
            f"{tier.staging_slots} staging slabs, admission >= {b.admission_min_pages} pages)",
            flush = True,
        )
        return tier

    def __init__(
        self,
        caches,
        path: str,
        max_size: int,
        min_free_pct: float = 15.0,
        segment_size: int = 256 * 1024**2,
        staging_size: int = 512 * 1024**2,
        admission_min_pages: int = 32,
        ttl_seconds: int = 0,
        require_dedicated_filesystem: bool = True,
        pump_pages: int = 4,
        pump_pct: float = 1.0,
        verify: int = 1,
        log_lookups: bool = True,
        scan_checkpoints: bool = True,
        extra_identity: dict | None = None,
        engine: dict | None = None,
        log_secs: float = 60.0,
    ):
        if torch is None:
            raise RuntimeError("DiskPageCache requires torch")
        if not os.path.isabs(path):
            raise ValueError("EXL3_NVME_TIER must be an absolute path")
        Path(path).mkdir(parents = True, exist_ok = True)
        # Inside a container "/" is the overlay rootfs, so this rejects only paths that are not a mount of their
        # own; that the mount is a dedicated host filesystem is the launcher's responsibility
        if require_dedicated_filesystem and os.stat(path).st_dev == os.stat("/").st_dev:
            raise ValueError("disk KV cache path must be on a filesystem distinct from the root filesystem")
        if staging_size <= 0 or staging_size > 2 * 1024**3:
            raise ValueError("disk KV pinned staging must be in (0, 2 GiB]")

        self.caches = caches
        self.pagetable = None
        self.recurrent_cache = None
        self.segments = []
        offset = 0
        layout = []
        models = []
        for cache in caches:
            cfg = cache.model.config
            models.append({
                "architecture": getattr(cfg, "arch_string", None) or type(cache.model).__name__,
                "checkpoint": _checkpoint_identity(cfg.directory),
                "max_num_tokens": getattr(cache, "max_num_tokens", None),
            })
            for layer_key, layer in cache.layers.items():
                for tensor_idx, tensor in enumerate(layer.get_tensors()):
                    if tensor is None:
                        continue
                    page_shape = tuple(tensor.shape[1:])
                    nbytes = tensor[0].numel() * tensor.element_size()
                    self.segments.append((tensor, offset, page_shape, tensor.dtype))
                    layout.append({
                        "layer": repr(layer_key), "class": type(layer).__name__, "tensor": tensor_idx,
                        "k_bits": getattr(layer, "k_bits", None), "v_bits": getattr(layer, "v_bits", None),
                        "dtype": str(tensor.dtype), "shape": list(page_shape), "bytes": nbytes,
                    })
                    offset = _align(offset + nbytes, 256)
        self.slab_size = _align(offset)
        if not self.slab_size:
            raise ValueError("no paged cache tensors found for disk tier")
        self.devices = sorted(
            {tensor.device for tensor, _, _, _ in self.segments if tensor.device.type == "cuda"},
            key = lambda d: d.index,
        )

        recurrent_layout = []
        for key, layer in caches[0].recurrent_layers.items():
            recurrent_layout.append({
                "layer": repr(key), "type": type(layer).__name__,
                "checkpoint_bytes": layer.get_checkpoint_size(),
            })
        namespace = {
            "page_size": PAGE_SIZE,
            "models": models,
            "page_layout": layout,
            "page_image_bytes": self.slab_size,
            "recurrent_layout": recurrent_layout,
            "engine": engine if engine is not None else engine_identity(),
            "generator": extra_identity or {},
        }
        self.backend = SegmentStore(
            path, namespace, max_size, min_free_pct, segment_size, admission_min_pages, ttl_seconds,
        )
        self.metrics = self.backend.metrics
        for k in (
            "write_queue_skips", "eviction_store_skips", "mm_refusals", "placeholder_refusals", "chain_refusals",
            "tp_refusals", "writer_errors", "restore_failures", "restored_pages", "restored_checkpoints",
            "checkpoint_restore_failures", "proactive_copies", "proactive_skips", "idle_copies",
            "eviction_copies", "evict_persist", "lookups", "lookup_hit", "lookup_no-disk-pages",
            "lookup_no-disk-ckpt", "lookup_ram-as-deep", "lookup_in-ram", "lookup_load-failed", "lookup_position",
            "lookup_put-dropped", "restore_aborts", "stash_mutations", "stash_views_copied", "pump_copies",
            "pump_seconds", "scan_ok", "scan_bad",
        ):
            self.metrics[k] += 0

        # Record payload: [page image (slab_size)][u32 token bytes][token ids int64]
        self.record_len = _align(self.slab_size + self.TOKEN_PREFIX.size + self.TOKEN_BUDGET)
        self.staging_slots = min(16, staging_size // self.record_len)
        if self.staging_slots < 2:
            raise ValueError("disk KV staging budget must hold at least two page images")
        self._pool = SlabPool(self.record_len, self.staging_slots, self._build_views)
        self.pump_pages = max(0, int(pump_pages))
        self.pump_frac = max(0.0, float(pump_pct)) / 100.0
        self._last_pump = None
        self._pump_credit = 0.0
        self._page_d2h = self.slab_size / 20e9  # seconds of D2H per page at ~20 GB/s
        self._page_cost = 3e-4 + self._page_d2h  # refined from measured copy launches
        self.verify = int(verify)
        self.log_lookups = bool(log_lookups)
        self.last_miss = None
        self._resume_key = None

        self.writes_disabled = False
        self._writer_consecutive_errors = 0
        self._write_queue = queue.Queue(maxsize = self.staging_slots + 4)
        self._queued_cps: set[bytes] = set()
        self._stop = object()
        self._writer = threading.Thread(target = self._writer_main, name = "exl3-nvme-writer", daemon = True)
        self._writer.start()
        self._read_workers = max(1, min(4, self.staging_slots - 1))
        self._read_window = self._read_workers
        self._read_pool = ThreadPoolExecutor(max_workers = self._read_workers, thread_name_prefix = "exl3-nvme-read")
        self._pending_copy: deque[bytes] = deque()
        self._pending_set: set[bytes] = set()
        self._open_batch: _RestoreBatch | None = None
        self._mm_free: dict[bytes, bool] = {}

        # Idle drain: while the generator is idle (no iterate() running, pages cannot move) a background thread
        # copies the pending chain pages on side streams. on_busy() stops it and makes the compute streams wait,
        # on the GPU, for its in-flight copies before any page can be reused.
        self._idle = False
        self._idle_lock = threading.Lock()
        self._idle_wake = threading.Event()
        self._idle_ready: dict = {}
        self._idle_inflight: list = []
        self._side_streams = {}
        self._closing = False
        self._drained_logged = True
        self._drainer = threading.Thread(target = self._drainer_main, name = "exl3-nvme-drain", daemon = True)
        self._drainer.start()

        self.log_secs = log_secs
        self._last_log = time.monotonic()
        self._last_logged = None

        # Open scan: read and digest-check every stored checkpoint once, in the background, and say how many are
        # intact. Older segments are only header-checked at open; this makes a bad checkpoint visible at boot
        # instead of at the first revisit (a failing record is quarantined exactly as an on-demand read would).
        self._scan_thread = None
        if scan_checkpoints and self.backend.checkpoints:
            self._scan_thread = threading.Thread(target = self._scan_checkpoints, name = "exl3-nvme-scan",
                                                 daemon = True)
            self._scan_thread.start()

    def _build_views(self, slab):
        views = []
        for tensor, off, shape, dtype in self.segments:
            nbytes = tensor[0].numel() * tensor.element_size()
            views.append(slab[off:off + nbytes].view(dtype).view(shape))
        return views

    def attach(self, pagetable, recurrent_cache):
        self.pagetable = pagetable
        self.recurrent_cache = recurrent_cache

    def __contains__(self, phash):
        return phash in self.backend

    def __len__(self):
        return len(self.backend)

    @property
    def entries(self):
        return self.backend.pages

    def prev_of(self, phash):
        return self.backend.prev_of(phash)

    def has_checkpoint(self, phash):
        return self.backend.has_checkpoint(phash)

    def pending_copies(self) -> int:
        return len(self._pending_copy)

    # ---------- generator thread: copies and non-blocking queue puts ----------

    def _events(self, streams: dict | None = None):
        events = []
        for device in self.devices:
            stream = streams[device] if streams is not None else torch.cuda.current_stream(device)
            with torch.cuda.device(device):
                event = torch.cuda.Event()
                event.record(stream)
            events.append(event)
        return events

    def _enqueue(self, item) -> bool:
        try:
            self._write_queue.put_nowait(item)
            return True
        except queue.Full:
            return False

    def _copy_out(self, key: bytes, page, streams: dict | None = None) -> str:
        """Copy one live page's image into a staging slab and queue its record. Never waits: returns "ok" or
        "busy" (no free slab, or the queue is full)."""
        sv = self._pool.try_acquire()
        if sv is None:
            return "busy"
        slab, views = sv
        for view, (tensor, _, _, _) in zip(views, self.segments):
            if streams is not None and tensor.device.type == "cuda":
                with torch.cuda.stream(streams[tensor.device]):
                    view.copy_(tensor[page.page_index], non_blocking = True)
            else:
                view.copy_(tensor[page.page_index], non_blocking = True)
        token_buf = page.sequence.contiguous().view(torch.uint8).numpy().tobytes()
        events = self._events(streams)
        if not self._enqueue(("page", key, page.prev_hash, token_buf, slab, views, events)):
            self._pool.release(slab, views, events)
            self.metrics["write_queue_skips"] += 1
            return "busy"
        return "ok"

    def _live_complete(self, key: bytes):
        page = self.pagetable.get_live_page(key) if self.pagetable is not None else None
        if page is not None and page.phash == key:
            return page
        return None

    def pump(self, budget: int | None = None):
        """Top of Generator.iterate: end an idle drain, release a restore batch a failed allocation left open,
        then copy out a few pages of admitted chains (async D2H on the compute stream, so any later overwrite of
        the page is stream-ordered after the copy)."""
        self.on_busy()
        if self._open_batch is not None:
            self._open_batch.finish()
            self._open_batch = None
        if self.writes_disabled:
            if self._pending_copy:
                self._pending_copy.clear()
                self._pending_set.clear()
            return
        now = time.perf_counter()
        last, self._last_pump = self._last_pump, now
        timed = budget is None
        if timed:
            # Time budget: each iteration earns pump_frac of its own duration (capped, so an idle gap or a long
            # prefill step cannot bank a burst); a page costs its measured host time for the copy launches plus
            # its D2H time on the compute stream (estimated from the page size). The idle drain does the bulk.
            budget = self.pump_pages
            if last is not None:
                self._pump_credit = min(
                    self._pump_credit + min(now - last, 0.1) * self.pump_frac, budget * self._page_cost
                )
        while self._pending_copy and budget > 0 and (not timed or self._pump_credit >= self._page_cost):
            key = self._pending_copy[0]
            page = self._live_complete(key)
            if page is not None and self.backend.should_store_page(key):
                t0 = time.perf_counter()
                if self._copy_out(key, page) == "busy":
                    break
                cost = time.perf_counter() - t0 + self._page_d2h
                self._page_cost = 0.8 * self._page_cost + 0.2 * cost
                self.metrics["pump_seconds"] += cost
                if timed:
                    self._pump_credit -= cost
                self.metrics["proactive_copies"] += 1
                self.metrics["pump_copies"] += 1
                budget -= 1
            else:
                self.metrics["proactive_skips"] += 1
            self._pending_copy.popleft()
            self._pending_set.discard(key)
        self._maybe_log()

    def store(self, page, serial: int = 0, protect = None):
        """PageTable.evict hook: the page's VRAM is about to be reused. Copy it out if it belongs to an admitted
        chain and is not on disk yet; skip, never wait, when no slab is free."""
        if self.writes_disabled or not self.backend.should_store_page(page.phash):
            return
        if self._copy_out(page.phash, page) == "ok":
            self.metrics["eviction_copies"] += 1
        else:
            self.metrics["eviction_store_skips"] += 1

    store_page = store

    # ---------- checkpoint admission ----------

    def _chain_mm_free(self, chain: list[bytes]) -> bool:
        """A chain is persistable only if none of its pages holds a multimodal placeholder id. Those ids come from a
        per-process counter (tokenizer/mm_embedding.py), so after a restart the same ids stand for other images
        and the page hash would alias different K/V."""
        memo = self._mm_free
        if len(memo) > 1_000_000:
            memo.clear()
        pt = self.pagetable
        for h in chain:
            ok = memo.get(h)
            if ok is None:
                page = pt.get_live_page(h) if pt is not None else None
                if page is not None:
                    ok = not bool((page.sequence >= FIRST_MM_EMBEDDING_INDEX).any())
                elif h in self.backend.pages:
                    ok = True  # stored pages were admitted as text-only
                elif pt is not None and pt.cpu_tier is not None and h in pt.cpu_tier:
                    ok = not bool((pt.cpu_tier.entries[h]["tokens"] >= FIRST_MM_EMBEDDING_INDEX).any())
                else:
                    ok = False
                memo[h] = ok
            if not ok:
                return False
        return True

    def own_stash(self, state, stashed: dict):
        """RecurrentCache.put hook, before the stash is stored: make it a snapshot (see own_stash_tensors)."""
        if getattr(state, "tier_owned", False):
            return
        n = own_stash_tensors(stashed)
        if n:
            self.metrics["stash_views_copied"] += n

    def persist_checkpoint(self, phash: bytes, stashed: dict) -> bool:
        """RecurrentCache.put hook: admit the checkpoint's chain, hand the checkpoint to the writer and queue the
        chain's pages that are not on disk for proactive copy-out."""
        from .pagetable import is_content_hash
        if self.writes_disabled or self.pagetable is None:
            return False
        if self.backend.has_checkpoint(phash) or phash in self._queued_cps:
            return True
        if not is_content_hash(phash):
            self.metrics["placeholder_refusals"] += 1
            return False
        if "tp_handle" in stashed:
            self.metrics["tp_refusals"] += 1
            return False
        chain = self.pagetable.chain_to_root(phash)
        if not chain:
            self.metrics["chain_refusals"] += 1
            return False
        if len(chain) < self.backend.admission_min_pages:
            self.metrics["admission_rejects"] += 1
            return False
        if not self._chain_mm_free(chain):
            self.metrics["mm_refusals"] += 1
            return False
        if not self._enqueue(("checkpoint", phash, chain, stashed)):
            self.metrics["write_queue_skips"] += 1
            return False
        self._queued_cps.add(phash)
        self.backend.admit_chain(chain)
        pages = self.backend.pages
        for k in chain:
            if k not in pages and k not in self._pending_set:
                self._pending_set.add(k)
                self._pending_copy.append(k)
        self._drained_logged = False
        return True

    def on_checkpoint_evicted(self, key: bytes, stashed: dict):
        """RecurrentCache.on_evict hook (the base LRU loop and the tip policy's evictions): the RAM copy is going
        away; persist it now if the put-time attempt did not (queue was full, chain was not complete then)."""
        if self.writes_disabled or self.backend.has_checkpoint(key) or key in self._queued_cps:
            return
        if self.persist_checkpoint(key, stashed):
            self.metrics["evict_persist"] += 1

    # ---------- restore (the generator thread waits only for reads a starting job needs) ----------

    def prepare_resume(self, page_hashes, recurrent_cache, pagetable, recurrent_pages: list, restore_limit: int) -> int:
        """
        Sequence.allocate_pages hook. If the deepest resumable checkpoint of this prompt (every page before it
        live in VRAM, in the CPU tier or on disk) is on disk only and deeper than every checkpoint in RAM, load it
        into the RAM cache now, before any page is allocated. A failed load changes nothing. Returns the new
        restore limit and appends the checkpoint's page index to recurrent_pages.

        Every call is counted with its outcome ("lookups" and the lookup_* counters in the summary line); a
        lookup that finds any of its pages on disk also prints one line saying what happened.
        """
        self._resume_key = None
        backend = self.backend
        if not backend.pages:
            return restore_limit
        self.metrics["lookups"] += 1
        cpu_tier = pagetable.cpu_tier
        deepest = None
        on_disk = 0
        walked = 0
        for i, h in enumerate(page_hashes):
            live = pagetable.get_live_page(h) is not None or (cpu_tier is not None and h in cpu_tier)
            disk = h in backend.pages
            if not (live or disk):
                break
            walked = i + 1
            on_disk += disk and not live
            if h in backend.checkpoints:
                deepest = i
        if on_disk == 0 and deepest is None:
            self._lookup_outcome("no-disk-pages", walked, len(page_hashes), None, restore_limit, quiet = True)
            return restore_limit
        if deepest is None:
            return self._lookup_outcome("no-disk-ckpt", walked, len(page_hashes), None, restore_limit)
        if deepest + 1 <= restore_limit:
            return self._lookup_outcome("ram-as-deep", walked, len(page_hashes), deepest, restore_limit,
                                        quiet = on_disk == 0)
        key = page_hashes[deepest]
        if key in recurrent_cache:
            return self._lookup_outcome("in-ram", walked, len(page_hashes), deepest, restore_limit,
                                        quiet = on_disk == 0)
        stashed, why = self.load_checkpoint(key)
        if stashed is None:
            self.metrics["checkpoint_restore_failures"] += 1
            return self._lookup_outcome("load-" + why, walked, len(page_hashes), deepest, restore_limit)
        if stashed.get("position") != (deepest + 1) * PAGE_SIZE:
            self.metrics["checkpoint_restore_failures"] += 1
            return self._lookup_outcome(
                f"position {stashed.get('position')}!={(deepest + 1) * PAGE_SIZE}", walked, len(page_hashes),
                deepest, restore_limit,
            )
        recurrent_cache.put(key, _StashedState(stashed))
        if recurrent_cache.get(key) is None:
            self.metrics["checkpoint_restore_failures"] += 1
            return self._lookup_outcome("put-dropped", walked, len(page_hashes), deepest, restore_limit)
        self.metrics["restored_checkpoints"] += 1
        self._resume_key = key
        # The put may have evicted a shallower checkpoint of this same prompt; the allocation must not fall back to it
        recurrent_pages[:] = [i for i in recurrent_pages if page_hashes[i] in recurrent_cache]
        recurrent_pages.append(deepest)
        return self._lookup_outcome("hit", walked, len(page_hashes), deepest, deepest + 1)

    def _lookup_outcome(self, outcome: str, walked: int, total: int, deepest, limit: int, quiet: bool = False):
        name = outcome.split(" ")[0]
        self.metrics["lookup_" + ("load-failed" if name.startswith("load-") else name)] += 1
        if outcome != "hit":
            self.last_miss = f"{outcome} (prefix {walked}/{total} pages, disk ckpt at {deepest})"
        if not quiet and self.log_lookups:
            print(f" -- nvme tier: lookup {outcome}: {walked}/{total} prompt pages found (VRAM/CPU/disk), deepest "
                  f"disk checkpoint {'-' if deepest is None else deepest + 1} pages, restore limit {limit}",
                  flush = True)
        return limit

    def begin_restore(self, page_hashes, pagetable) -> _RestoreBatch | None:
        """PageTable.allocate_pages hook: start bounded parallel reads, in allocation order, of the pages that are
        on disk and neither in VRAM nor in the CPU tier."""
        if self._open_batch is not None:
            self._open_batch.finish()
            self._open_batch = None
        cpu_tier = pagetable.cpu_tier
        keys = [
            h for h in page_hashes
            if h not in pagetable.referenced_pages and h not in pagetable.unreferenced_pages and
            not (cpu_tier is not None and h in cpu_tier) and h in self.backend.pages
        ]
        if not keys:
            return None
        self._open_batch = _RestoreBatch(self, keys)
        return self._open_batch

    def abort_restore(self, ops: list, reason: str = "page read failed"):
        """PageTable.allocate_pages hook, when a disk page could not be restored: return this allocation to the
        state it would have had without the tier. Pages already copied from disk go back to the state of a freshly
        claimed page (nothing valid, prefill rewrites them) and a checkpoint loaded from disk for this allocation
        leaves the RAM cache, so the job resumes from what VRAM/RAM alone hold (usually a cold prefill)."""
        for op in ops:
            op.kv_position = 0
            op.prev_hash = None
        key = self._resume_key
        self._resume_key = None
        rc = self.recurrent_cache
        dropped = False
        if key is not None and rc is not None and key in rc:
            rc.pop(key)
            rc.update_total_size()
            dropped = True
        self.metrics["restore_aborts"] += 1
        if self.metrics["restore_aborts"] <= 20:
            lf = self.backend.last_read_failure
            print(f" -- nvme tier: restore aborted ({reason}{'' if lf is None else ': ' + lf[4]}); "
                  f"{len(ops)} restored pages reset{', disk checkpoint dropped' if dropped else ''}; "
                  f"the job resumes from VRAM/RAM only", flush = True)

    def end_restore(self, batch: _RestoreBatch | None):
        if batch is not None:
            batch.finish()
            if self._open_batch is batch:
                self._open_batch = None

    def _read_page_task(self, key: bytes):
        """Reader thread: fill a slab straight from NVMe (preadv into the pinned buffer) and parse the token ids."""
        slab, views = self._pool.acquire()
        try:
            np_buf = slab.numpy()
            payload_len = self.backend.read_into(key, memoryview(np_buf))
            token_len = None
            if payload_len is not None:
                token_len, = self.TOKEN_PREFIX.unpack_from(np_buf, self.slab_size)
            if token_len != PAGE_SIZE * 8 or payload_len != self.slab_size + self.TOKEN_PREFIX.size + token_len:
                self._pool.release(slab, views, None)
                return None
            start = self.slab_size + self.TOKEN_PREFIX.size
            token_ids = torch.frombuffer(
                memoryview(np_buf)[start:start + token_len], dtype = torch.long
            ).view(1, PAGE_SIZE).clone()
        except BaseException:
            self._pool.release(slab, views, None)
            raise
        return slab, views, token_ids

    def _load_checkpoint_task(self, phash: bytes):
        payload = self.backend.fetch_checkpoint(phash)
        if payload is None:
            lf = self.backend.last_read_failure
            return None, "read:" + (lf[4] if lf is not None and lf[1] == phash[:4].hex() else "missing")
        try:
            return deserialize_stash(payload), ""
        except Exception as e:
            return None, f"decode:{type(e).__name__}"

    def load_checkpoint(self, phash: bytes):
        """Read + decode on a reader thread; the caller waits. Returns (stash or None, failure reason)."""
        try:
            return self._read_pool.submit(self._load_checkpoint_task, phash).result()
        except Exception as e:
            return None, f"error:{type(e).__name__}"

    # ---------- idle drain ----------

    def on_idle(self):
        """End of Generator.on_queue_drained: pages cannot move until the next on_busy(), so the drainer may copy
        the pending chain pages."""
        if self._closing:
            return
        ready = {}
        for d in self.devices:
            with torch.cuda.device(d):
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream(d))
            ready[d] = ev
        with self._idle_lock:
            self._idle_ready = ready
            self._idle = True
        self._last_pump = None
        self._pump_credit = 0.0
        self._idle_wake.set()

    def on_busy(self):
        """Start of iterate() and of on_queue_drained(): stop the drainer; the compute streams wait, on the GPU,
        for the copies it has in flight."""
        if not self._idle:
            return
        with self._idle_lock:
            self._idle = False
            self._idle_wake.clear()
            for d, ev in self._idle_inflight:
                torch.cuda.current_stream(d).wait_event(ev)
            self._idle_inflight.clear()

    def _side_stream(self, device):
        s = self._side_streams.get(device)
        if s is None:
            s = torch.cuda.Stream(device = device)
            self._side_streams[device] = s
        return s

    def _drain_step(self, state: dict) -> str:
        """One page under the idle lock: "idle-ended", "empty", "busy", "copied" or "skipped"."""
        with self._idle_lock:
            if not self._idle or self._closing:
                return "idle-ended"
            if self.writes_disabled:
                self._pending_copy.clear()
                self._pending_set.clear()
            if not self._pending_copy:
                return "empty"
            streams = None
            if self.devices:
                streams = {d: self._side_stream(d) for d in self.devices}
                if not state.get("waited"):
                    for d, ev in self._idle_ready.items():
                        streams[d].wait_event(ev)
                    state["waited"] = True
            key = self._pending_copy[0]
            page = self._live_complete(key)
            if page is not None and self.backend.should_store_page(key):
                if self._copy_out(key, page, streams) == "busy":
                    return "busy"
                if streams is not None:
                    for d in self.devices:
                        with torch.cuda.device(d):
                            ev = torch.cuda.Event()
                            ev.record(streams[d])
                        self._idle_inflight.append((d, ev))
                    if len(self._idle_inflight) > 64:
                        self._idle_inflight = [x for x in self._idle_inflight if not x[1].query()]
                self.metrics["idle_copies"] += 1
                self.metrics["proactive_copies"] += 1
                result = "copied"
            else:
                self.metrics["proactive_skips"] += 1
                result = "skipped"
            self._pending_copy.popleft()
            self._pending_set.discard(key)
            return result

    def _drainer_main(self):
        while True:
            self._idle_wake.wait()
            if self._closing:
                return
            state = {}
            while True:
                r = self._drain_step(state)
                if r in ("idle-ended", "empty"):
                    break
                if r == "busy":
                    time.sleep(0.002)
            if r == "empty" and not self._drained_logged:
                # Let the writer finish what is queued, then say so: the restart test keys on this line
                while self._write_queue.unfinished_tasks and self._idle and not self._closing:
                    time.sleep(0.01)
                if self._idle and not self._closing and not self._pending_copy:
                    self._drained_logged = True
                    print(f" -- nvme tier: drained ({self.summary()})", flush = True)
            with self._idle_lock:
                if not self._closing and (not self._idle or not self._pending_copy or self.writes_disabled):
                    self._idle_wake.clear()

    # ---------- writer thread ----------

    def _writer_main(self):
        while True:
            task = self._write_queue.get()
            if task is self._stop:
                self._write_queue.task_done()
                return
            kind = task[0]
            try:
                if self.writes_disabled:
                    continue
                if kind == "page":
                    _, key, prev, token_buf, slab, views, events = task
                    for event in events:
                        event.synchronize()
                    image = memoryview(slab.numpy())[:self.slab_size]
                    self.backend.store_page(key, prev, [image, self.TOKEN_PREFIX.pack(len(token_buf)), token_buf],
                                            verify = self.verify >= 2)
                elif kind == "checkpoint":
                    _, key, chain, stashed = task
                    try:
                        payload = self._snapshot_checkpoint(key, stashed)
                        if payload is not None:
                            self.backend.store_checkpoint(key, chain, payload, verify = self.verify >= 1)
                    finally:
                        self._queued_cps.discard(key)
                if self._write_queue.qsize() == 0:
                    self.backend.reclaim_dirty(1)
                self._writer_consecutive_errors = 0
            except Exception as exc:
                self.metrics["writer_errors"] += 1
                self._writer_consecutive_errors += 1
                if self.metrics["writer_errors"] <= 5:
                    print(f" -- nvme tier: writer error {type(exc).__name__}: {exc}", flush = True)
                if self._writer_consecutive_errors >= WRITER_MAX_CONSECUTIVE_ERRORS:
                    self.writes_disabled = True
                    print(" -- nvme tier: writes disabled after repeated errors (reads continue)", flush = True)
            finally:
                if kind == "page":
                    self._pool.release(task[4], task[5], None)
                self._write_queue.task_done()

    def _scan_checkpoints(self):
        b = self.backend
        t0 = time.monotonic()
        keys = list(b.checkpoints)
        ok = bad = 0
        reasons = defaultdict(int)
        for key in keys:
            if self._closing:
                return
            e, buf = b._read_record(b.checkpoints, key, bytearray)
            if e is not None:
                try:
                    st = deserialize_stash(buf)
                    ok += 1
                    del st
                except Exception as exc:
                    bad += 1
                    reasons["decode:" + type(exc).__name__] += 1
            elif b.last_read_failure is not None and b.last_read_failure[1] == key[:4].hex():
                bad += 1
                reasons[b.last_read_failure[4]] += 1
            del buf
        self.metrics["scan_ok"] += ok
        self.metrics["scan_bad"] += bad
        print(f" -- nvme tier: open scan: {ok}/{len(keys)} checkpoints intact"
              + (f", {bad} failed ({', '.join(f'{k} {v}' for k, v in reasons.items())})" if bad else "")
              + f" in {time.monotonic() - t0:.1f} s", flush = True)

    def _snapshot_checkpoint(self, key: bytes, stashed: dict):
        """Writer thread: serialize a stash into one buffer the tier owns, so the bytes hashed and written cannot
        change underneath the write. With verification on, hash the source again after the copy: a difference
        means something wrote into the stash's tensors after RecurrentCache.put, and the checkpoint is not stored."""
        views = [_bytes_view(p) for p in serialize_stash(stashed)]
        owned = bytearray(sum(v.nbytes for v in views))
        pos = 0
        for v in views:
            owned[pos:pos + v.nbytes] = v
            pos += v.nbytes
        if self.verify >= 1:
            h = hashlib.blake2b(digest_size = 16)
            for v in views:
                h.update(v)
            if h.digest() != hashlib.blake2b(owned, digest_size = 16).digest():
                self.metrics["stash_mutations"] += 1
                if self.metrics["stash_mutations"] <= 10:
                    print(f" -- nvme tier: checkpoint {key[:4].hex()} changed while it was being serialized; "
                          f"not stored", flush = True)
                return None
        return owned

    def flush(self, timeout: float | None = None) -> bool:
        """Wait until the writer has consumed the queue (tests and close; never the generator thread)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._write_queue.unfinished_tasks:
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.002)
        return True

    # ---------- reporting ----------

    def summary(self) -> str:
        b = self.backend
        m = self.metrics.copy()  # other threads add to the counters
        return (
            f"{len(b.pages)} pages, {len(b.checkpoints)} ckpts, {b.disk_bytes() / 1024**3:.2f}/"
            f"{b.capacity_bytes / 1024**3:.1f} GiB; wrote {m['page_writes']} pages + {m['checkpoint_writes']} ckpts "
            f"({m['bytes_written'] / 1024**3:.2f} GiB); restored {m['restored_pages']} pages + "
            f"{m['restored_checkpoints']} ckpts; pending {len(self._pending_copy)}; skips queue "
            f"{m['write_queue_skips']} evict {m['eviction_store_skips']}; refused mm {m['mm_refusals']} cap "
            f"{m['capacity_refusals']} free {m['free_space_refusals']}; evicted {m['pruned_pages']} pages "
            f"{m['pruned_checkpoints']} ckpts (interior {m['evicted_interior_checkpoints']}); compactions "
            f"{m['compactions']}; errors {m['writer_errors']}; lookups {m['lookups']}: hit {m['lookup_hit']}, "
            f"load-failed {m['lookup_load-failed']}, "
            f"no-ckpt {m['lookup_no-disk-ckpt']}, ram {m['lookup_ram-as-deep'] + m['lookup_in-ram']}, "
            f"position {m['lookup_position']}, aborts {m['restore_aborts']}; integrity: read-fail "
            f"{', '.join(f'{k[10:]} {v}' for k, v in sorted(m.items()) if k.startswith('read_fail_') and v) or '0'}, "
            f"quarantined {m['quarantined']}, write-verify {m['write_verify_failures']}, stash-mutated "
            f"{m['stash_mutations']}, views-copied {m['stash_views_copied']}; pump {m['pump_copies']} pages ({self._page_cost * 1e3:.2f} ms/page), idle "
            f"{m['idle_copies']}"
            + (f"; last miss: {self.last_miss}" if self.last_miss else "")
            + ("; WRITES DISABLED" if self.writes_disabled else "")
        )

    def _maybe_log(self):
        if self.log_secs <= 0:
            return
        now = time.monotonic()
        if now - self._last_log < self.log_secs:
            return
        self._last_log = now
        m = self.metrics
        snap = (m["page_writes"], m["checkpoint_writes"], m["restored_pages"], m["restored_checkpoints"],
                len(self._pending_copy))
        if snap != self._last_logged:
            self._last_logged = snap
            print(f" -- nvme tier: {self.summary()}", flush = True)

    def close(self):
        if self._closing:
            return
        with self._idle_lock:
            self._closing = True
            self._idle = False
        self._idle_wake.set()
        self._drainer.join(timeout = 10)
        if self._scan_thread is not None:
            self._scan_thread.join(timeout = 30)
        for d, ev in self._idle_inflight:
            ev.synchronize()
        self._idle_inflight.clear()
        if self._open_batch is not None:
            self._open_batch.finish()
            self._open_batch = None
        self._write_queue.put(self._stop)
        self._writer.join()
        self._read_pool.shutdown(wait = True, cancel_futures = True)
        self.backend.close()
