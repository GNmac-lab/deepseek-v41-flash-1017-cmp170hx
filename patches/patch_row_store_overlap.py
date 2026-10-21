#!/usr/bin/env python3
"""Overlapped Engram row reads (zebgop dsv41reap-pp overlay/engram_ssd/row_store.cpp).

Profiled 2026-10-19 (TP4, warm prose): the on-stream host callback that reads one rank's Engram rows was the main
reason ranks arrived late at all-reduces (0.4-0.9 ms/step per rank). Only 0.4% of the 189 GiB tables is in page
cache, and a lookup of <= 64 rows read them one after another, so each miss (weight row and scale row) cost a full
NVMe latency. Now: copy every row that is already cached with preadv2(RWF_NOWAIT), start the reads of all misses
together (POSIX_FADV_WILLNEED), then wait for them, so a lookup costs about one device latency. The rank's scale
rows (1/32 of the table) are read ahead into page cache at open, so a miss is one read instead of two.
POSIX_FADV_RANDOM stops readahead around 256-byte rows. Rows are byte-identical; DSV41_ENGRAM_OVERLAP=0 = old path.
usage: patch_row_store_overlap.py <engram_ssd/row_store.cpp>
"""
import sys

p = sys.argv[1]
s = open(p).read()
if "RWF_NOWAIT" in s:
    print("already patched"); sys.exit(0)


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:60], s.count(old))
    s = s.replace(old, new, 1)


rep("#include <sys/stat.h>\n", "#include <sys/stat.h>\n#include <sys/uio.h>\n")
rep("namespace {\n", "namespace {\n\nconstexpr uint64_t kChunk = 64;  // rows per job\n", )
rep("  Pool *pool = nullptr;\n",
    "  Pool *pool = nullptr;\n  bool overlap = true;\n  std::atomic<bool> nowait{true};\n")
rep('''  pread_all(s->fd, sc, 8, s->soff + uint64_t(id) * 8);
}
''', '''  pread_all(s->fd, sc, 8, s->soff + uint64_t(id) * 8);
}

// Copy [len] bytes at [off] only if they are already in page cache, without blocking (RWF_NOWAIT; on a miss
// the kernel also starts the read). Returns false on a miss.
bool read_cached(Store *s, void *out, size_t len, uint64_t off) {
  if (!s->nowait.load(std::memory_order_relaxed)) return false;
  struct iovec v{out, len};
  for (;;) {
    ssize_t got = preadv2(s->fd, &v, 1, off_t(off), RWF_NOWAIT);
    if (got == ssize_t(len)) return true;
    if (got >= 0 || errno == EAGAIN) return false;  // not (fully) cached
    if (errno == EINTR) continue;
    if (errno == EOPNOTSUPP || errno == ENOSYS || errno == EINVAL) {
      s->nowait.store(false, std::memory_order_relaxed);
      return false;
    }
    fail("preadv2 failed");
  }
}

struct Miss { uint8_t *dst; size_t len; uint64_t off; };

// Rows [lo, hi) of one lookup (hi - lo <= kChunk). Cached rows are copied first; then the reads of all misses
// are started together and waited for, so the misses cost about one device latency instead of one each.
void fetch_range(Store *s, const int64_t *ids, uint8_t *weights, uint8_t *scales, uint64_t lo, uint64_t hi) {
  if (!s->overlap) {
    for (uint64_t i = lo; i < hi; ++i) fetch_one(s, ids[i], weights + i * 256, scales + i * 8);
    return;
  }
  Miss miss[2 * kChunk];
  int nm = 0;
  for (uint64_t i = lo; i < hi; ++i) {
    const int64_t id = ids[i];
    uint8_t *w = weights + i * 256, *sc = scales + i * 8;
    if (id < 0 || uint64_t(id) < s->row_lo || uint64_t(id) >= s->row_hi) {
      std::memset(w, 0, 256); std::memset(sc, 0, 8); continue;
    }
    if (uint64_t(id) >= s->rows) fail("row id out of bounds");
    const uint64_t woff = s->woff + uint64_t(id) * 256, soff = s->soff + uint64_t(id) * 8;
    if (!read_cached(s, w, 256, woff)) miss[nm++] = {w, 256, woff};
    if (!read_cached(s, sc, 8, soff)) miss[nm++] = {sc, 8, soff};
  }
  for (int k = 0; k < nm; ++k)
    posix_fadvise(s->fd, off_t(miss[k].off), off_t(miss[k].len), POSIX_FADV_WILLNEED);
  for (int k = 0; k < nm; ++k) pread_all(s->fd, miss[k].dst, miss[k].len, miss[k].off);
}

// Read the scale rows of [lo, hi) ahead into page cache (8 B per row, 1/32 of the table).
void prefetch_scales(Store *s, uint64_t lo, uint64_t hi) {
  if (s->overlap && hi > lo)
    posix_fadvise(s->fd, off_t(s->soff + lo * 8), off_t((hi - lo) * 8), POSIX_FADV_WILLNEED);
}
''')
rep('''  s->rows = rows; s->woff = woff; s->soff = soff; s->row_lo = 0; s->row_hi = rows;
''', '''  s->rows = rows; s->woff = woff; s->soff = soff; s->row_lo = 0; s->row_hi = rows;
  const char *ov = std::getenv("DSV41_ENGRAM_OVERLAP");
  s->overlap = !(ov && ov[0] == '0');
  if (s->overlap) posix_fadvise(s->fd, 0, 0, POSIX_FADV_RANDOM);
''')
rep('''  s->row_lo = lo; s->row_hi = hi;
}''', '''  s->row_lo = lo; s->row_hi = hi;
  prefetch_scales(s, lo, hi);
}''')
rep('''  const uint64_t chunk = 64;  // rows per job
''', '''  const uint64_t chunk = kChunk;
''')
rep('''  if (njobs == 1) {
    for (uint64_t i = 0; i < n; ++i)
      fetch_one(s, work->ids[i], work->weights + i * 256, work->scales + i * 8);
    return;
  }''', '''  if (njobs == 1) {
    fetch_range(s, work->ids, work->weights, work->scales, 0, n);
    return;
  }''')
rep('''    const uint64_t lo = uint64_t(j) * chunk, hi = std::min<uint64_t>(n, lo + chunk);
    for (uint64_t i = lo; i < hi; ++i)
      fetch_one(s, work->ids[i], work->weights + i * 256, work->scales + i * 8);
''', '''    const uint64_t lo = uint64_t(j) * chunk, hi = std::min<uint64_t>(n, lo + chunk);
    fetch_range(s, work->ids, work->weights, work->scales, lo, hi);
''')
open(p, "w").write(s)
print("patched", p)
