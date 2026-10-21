#!/usr/bin/env python3
"""512-column tiles for exllamav3's coop MoE down projection (exllamav3_ext/quant/exl3_moe_coop_kernel.cuh and
exl3_moe_coop.cu). Apply after patch_coop_branchfree.py.

The down GEMV has a short k (I = 512 per TP rank on DeepSeek-V4.1-Flash), so the wide geometry (4 warps along n,
4-way in-block k split, 128 columns) spends a large share of each block on its prologue and the cross-warp
reduction. With 16 warps along n (one k split, 512 columns per block) the reduction disappears: offline
(offline kernel bench, CMP 170HX) down 102.5 -> 87.2 us. Gate/up keeps the 128-column tile (wider is slower there).
The k partial sums are added in a different order, so results differ from the 128-column tile at fp32 rounding
level (max 1.2e-7 offline), not in precision. Layers whose padded output width is not a multiple of 512 fall back
to the narrow tile. usage: patch_coop_down_wide16.py <exl3_moe_coop_kernel.cuh> <exl3_moe_coop.cu>
"""
import sys

kp, lp = sys.argv[1], sys.argv[2]
k = open(kp).read()
l = open(lp).read()
if "patch_coop_down_wide16" in k:
    print("already patched"); sys.exit(0)
assert "patch_coop_branchfree" in k, "apply patch_coop_branchfree.py first"


def rep(s, old, new, n=1):
    assert s.count(old) == n, (old[:70], s.count(old))
    return s.replace(old, new)


# geometry helpers: int G (0 narrow, 1 wide, 2 = 16 warps along n); bool callers convert to 0 / 1
k = rep(k, "template <bool WIDE> constexpr int tile_wn() { return WIDE ? 4 : 1; }",
        "// dsv41 (patch_coop_down_wide16.py): geometry 2 = 16 warps along n, no in-block k split (down stage)\n"
        "template <int WIDE> constexpr int tile_wn() { return WIDE == 2 ? 16 : WIDE ? 4 : 1; }")
for f in ("tile_wk", "tile_cols", "tile_gpc", "tile_cpb"):
    k = rep(k, f"template <bool WIDE> constexpr int {f}()", f"template <int WIDE> constexpr int {f}()")
# gemv_tile takes the geometry as an int
k = rep(k, """template <int bits, int cb, bool WIDE, bool HALF = false>
__device__ __forceinline__ void gemv_tile""", """template <int bits, int cb, int WIDE, bool HALF = false>
__device__ __forceinline__ void gemv_tile""")
# down kernel: wide -> geometry 2
b0 = k.index("void exl3_moe_coop_b_kernel(const MoeCoopParams p)")
kb = k[b0:]
kb = rep(kb, """    constexpr int TCOLS = tile_cols<WIDE>();
    constexpr int GPC = tile_gpc<WIDE>();
    constexpr int CPB = tile_cpb<WIDE>();""", """    constexpr int GB = WIDE ? 2 : 0;  // dsv41: 512-column tiles for the short-k down stage
    constexpr int TCOLS = tile_cols<GB>();
    constexpr int GPC = tile_gpc<GB>();
    constexpr int CPB = tile_cpb<GB>();""")
kb = rep(kb, "        gemv_tile<bits, cb, WIDE, HALF>(B32, (const half2*) p.act_out,",
         "        gemv_tile<bits, cb, GB, HALF>(B32, (const half2*) p.act_out,")
k = k[:b0] + kb
static = "static_assert(THREADS == MOE_COOP_THREADS"
assert static in k
k = k.replace(static, "static_assert(tile_wk<2>() * tile_cols<2>() == WK * COLS, \"down tile reduction buffer\");\n" + static, 1)
# launcher: grid of the down stage uses 512-column groups; widths not divisible by 512 take the narrow tile
l = rep(l, "    const bool wide_b = moe_coop_pick_wide(p.I / 16, slots, device);",
        "    // dsv41 (patch_coop_down_wide16.py): the wide down tile is 512 columns; other widths take the narrow tile\n"
        "    const bool wide_b = moe_coop_pick_wide(p.I / 16, slots, device) && p.Ho % 512 == 0;")
l = rep(l, "    const int base_b = slots * (p.Ho / (wide_b ? 128 : MOE_COOP_COLS));",
        "    const int base_b = slots * (p.Ho / (wide_b ? 512 : MOE_COOP_COLS));")
open(kp, "w").write(k)
open(lp, "w").write(l)
print("patched", kp, lp)
