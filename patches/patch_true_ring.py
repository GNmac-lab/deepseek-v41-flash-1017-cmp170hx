#!/usr/bin/env python3
"""dsv41 v34: a cut-through neighbour-ring variant for the vendored FlashInfer PCIe IPC all-reduce (world size 4).

The stock kStagedRing kernel is a two-shot whose 6 push passes are separated by group-wide barriers; on this rig's
root complex each barrier costs more than the 15 KB it fences (6 x 5120 bf16: 43 us). Here every rank only writes to
rank + 1 and nothing waits on a barrier: each 16-byte pack carries its own arrival signal (the sentinel protocol the
staged kernels already use), so a pack moves to the next hop the moment it lands. Microbench (bench/ring_ar.cu,
4 GPUs, graph-replayed): 1 / 6 / 48 tokens = 14 / 18 / 118 us vs 11.5 / 42.6 / 156.8 us stock.

Registered as IpcVariant.TRUE_RING = 4, so vLLM's startup tune() measures it next to the stock kernels and keeps
whichever is fastest per batch (the tiny batches stay on kStaged). PCIE_IPC_TUNE_VERSION is bumped so tune caches
written without the new variant are not reused.
usage: patch_true_ring.py <flashinfer package dir>"""
import sys
from pathlib import Path

fi = Path(sys.argv[1])


def patch(p, pairs, mark):
    s = p.read_text()
    if mark in s:
        print("already patched", p)
        return
    for old, new in pairs:
        assert s.count(old) == 1, (p, s.count(old), old[:90])
        s = s.replace(old, new)
    p.write_text(s)
    print("patched", p)


KERNEL = r"""
// dsv41 (ds41-patches/patch_true_ring.py): cut-through neighbour ring, world_size 4, 16-bit payloads, kBlock region.
// Every rank only ever writes to rank + 1, and there are no barriers: each pack carries its own arrival signal (the
// sentinel protocol above), so it moves on to the next hop as soon as it lands. Reduce-scatter and all-gather are one
// 2*(WorldSize-1)-hop chain per pack, run stage-major so every pack is in flight at once. Chunk c is summed in ring
// order starting at rank c + 1 (bf16 adds, as in the staged kernels). Each slot is written by exactly one sender and
// read once by its owner, which resets it right away, so the region is clean on exit. The epoch double buffer is safe
// without barriers: a rank starts call n + 2 only after call n + 1's all-gather reached it, which its neighbour sent
// after finishing call n.
template <typename T, int WorldSize, bool UsePdl>
__global__ __launch_bounds__(1024, 1) void ipc_true_ring_param_kernel(
    const PushOneshotParamData<T> __grid_constant__ params) {
  static_assert(sizeof(T) == 2, "16-bit payloads only");
  pdl_grid_sync_const<UsePdl>();
  int32_t* epoch_slot = params.epoch_slots + blockIdx.x;
  const int epoch = advance_scratch_epoch(params.scratch_state, epoch_slot);
  const int rank = params.rank;
  const int part = params.num_packs / WorldSize;
  const int slot_stride = part + params.num_packs % WorldSize;  // the last chunk is the largest
  uint4 const* input = reinterpret_cast<uint4 const*>(params.input);
  uint4* output = reinterpret_cast<uint4*>(params.output);
  uint4* local = reinterpret_cast<uint4*>(params.tmp_ptrs[rank]) + epoch * params.epoch_stride_packs;
  uint4* remote =
      reinterpret_cast<uint4*>(params.tmp_ptrs[(rank + 1) % WorldSize]) + epoch * params.epoch_stride_packs;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const uint4 reset = {0u, 0u, 0u, 0u};
  // Reduce-scatter. Step 0 sends this rank's input of chunk (rank - 1); step s receives the partial sum of chunk
  // (rank - 1 - s), adds this rank's input and forwards it. Step WorldSize - 1 completes this rank's own chunk.
  {
    const int c = (rank + WorldSize - 1) % WorldSize;
    const int base = c * part, len = c == WorldSize - 1 ? params.num_packs - base : part;
    for (int i = tid; i < len; i += stride) {
      store_u4_volatile(remote, i, clear_pos_zero_u4_16(input[base + i]));
    }
  }
#pragma unroll
  for (int s = 1; s < WorldSize; ++s) {
    const int c = (rank + 2 * WorldSize - 1 - s) % WorldSize;
    const int base = c * part, len = c == WorldSize - 1 ? params.num_packs - base : part;
    uint4* in_slot = local + (s - 1) * slot_stride;
    for (int i = tid; i < len; i += stride) {
      uint4 v;
      do {
        v = load_u4_volatile(in_slot, i);
      } while (has_pos_zero_u4_16(v));
      store_u4_volatile(in_slot, i, reset);
      v = packed_add_u4<T>(v, input[base + i]);
      if (s == WorldSize - 1) {
        output[base + i] = v;
      }
      store_u4_volatile(remote + s * slot_stride, i, clear_pos_zero_u4_16(v));
    }
  }
  // All-gather. Step t receives the final chunk (rank - 1 - t) and forwards it unless this is its last hop.
#pragma unroll
  for (int t = 0; t < WorldSize - 1; ++t) {
    const int c = (rank + 2 * WorldSize - 1 - t) % WorldSize;
    const int base = c * part, len = c == WorldSize - 1 ? params.num_packs - base : part;
    uint4* in_slot = local + (WorldSize - 1 + t) * slot_stride;
    for (int i = tid; i < len; i += stride) {
      uint4 v;
      do {
        v = load_u4_volatile(in_slot, i);
      } while (has_pos_zero_u4_16(v));
      store_u4_volatile(in_slot, i, reset);
      output[base + i] = v;
      if (t < WorldSize - 2) {
        store_u4_volatile(remote + (WorldSize + t) * slot_stride, i, v);
      }
    }
  }
  debug_commit_per_block_epoch(epoch_slot, epoch);
  pdl_grid_release_const<UsePdl>();
}

template <typename Kernel, typename... Args>
inline cudaError_t launch("""

H = fi / "data/include/flashinfer/comm/pcie_ipc_all_reduce.cuh"
patch(H, [
    ("""  kFlatStaged = 3,  // staged pushes without the island decomposition
};

constexpr int kVariantCount = 4;""", """  kFlatStaged = 3,  // staged pushes without the island decomposition
  kTrueRing = 4,    // dsv41: cut-through neighbour ring without barriers; world_size 4
};

constexpr int kVariantCount = 5;"""),
    ("""
template <typename Kernel, typename... Args>
inline cudaError_t launch(""", KERNEL),
    ("""      case Variant::kStagedRing:                                                           \\
        return FI_PCIE_IPC_LAUNCH((ipc_rsag_ring_push_param_kernel<T, 4, PDL>), PDL);      \\
      default:                                                                             \\
""", """      case Variant::kStagedRing:                                                           \\
        return FI_PCIE_IPC_LAUNCH((ipc_rsag_ring_push_param_kernel<T, 4, PDL>), PDL);      \\
      case Variant::kTrueRing:                                                             \\
        if constexpr (sizeof(T) == 2) {                                                    \\
          return FI_PCIE_IPC_LAUNCH((ipc_true_ring_param_kernel<T, 4, PDL>), PDL);         \\
        } else {                                                                           \\
          return FI_PCIE_IPC_LAUNCH((ipc_rsag_ring_push_param_kernel<T, 4, PDL>), PDL);    \\
        }                                                                                  \\
      default:                                                                             \\
"""),
], "ipc_true_ring_param_kernel")

C = fi / "data/csrc/pcie_ipc_all_reduce.cu"
patch(C, [
    ("""  TVM_FFI_ICHECK(!(algo == fi::Variant::kFlatStaged && h->world_size != 8))""",
     """  TVM_FFI_ICHECK(!(algo == fi::Variant::kTrueRing && h->world_size != 4))
      << "kTrueRing (dsv41) is world_size 4 only";
  TVM_FFI_ICHECK(!(algo == fi::Variant::kFlatStaged && h->world_size != 8))"""),
], "kTrueRing (dsv41)")

P = fi / "comm/pcie_ipc_policy.py"
patch(P, [
    ("""    FLAT_STAGED = 3
""", """    FLAT_STAGED = 3
    # dsv41 (ds41-patches/patch_true_ring.py): cut-through neighbour ring, world size 4 only
    TRUE_RING = 4
"""),
    ("""    if config.variant == IpcVariant.FLAT_STAGED and world_size != 8:
        return False
""", """    if config.variant == IpcVariant.FLAT_STAGED and world_size != 8:
        return False
    # Every block spins on packs written by the same block index of the previous rank, so all blocks must be
    # resident at once; 16 is far below occupancy even beside the decode kernels.
    if config.variant == IpcVariant.TRUE_RING and (world_size != 4 or config.blocks > 16):
        return False
"""),
], "TRUE_RING")

T = fi / "comm/pcie_ipc_tuning.py"
patch(T, [("PCIE_IPC_TUNE_VERSION = 1\n", "PCIE_IPC_TUNE_VERSION = 2  # dsv41: + IpcVariant.TRUE_RING\n")],
      "PCIE_IPC_TUNE_VERSION = 2")
