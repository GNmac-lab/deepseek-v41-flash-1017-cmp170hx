#!/usr/bin/env python3
"""dsv41 v38 trial bundle (both off by default, for an in-image A/B):
  1. DSV41_DRAFT_HEAD_FP8=1: DSpark's sharded draft argmax reads a draft-only FP8 copy (128x128 block scales) of the
     vocab head shard it shares with the target, via the FP8 GEMV kernel (240 -> 135 us per call at 5 rows). The
     target still scores every draft with its own bf16 head, so outputs are unchanged; only acceptance could move.
  2. DSV41_ROUTER_SPLIT / DSV41_ROUTER_BN: split-K and column tile of the small-batch router GEMM (default 8 / 64);
     more blocks may win a larger share of the SMs beside the concurrent shared-expert GEMM.
usage: patch_v38.py <site-packages> <dsv41_fp8_gemv.py>"""
import shutil
import sys

sp, gemv = sys.argv[1], sys.argv[2]
shutil.copy(gemv, f"{sp}/hybrid/dsv41_fp8_gemv.py")


def patch(p, pairs, mark):
    s = open(p).read()
    if mark in s:
        print("already patched", p)
        return
    for old, new in pairs:
        assert s.count(old) == 1, (p, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w").write(s)
    print("patched", p)


patch(f"{sp}/hybrid/dspark_proposer.py", [
    ("""        local = lp._get_logits(m.model.norm(sample_hidden), head, None, skip_gather=True)
""", """        xh = m.model.norm(sample_hidden)
        q8 = self._dsv41_draft_head_fp8(head) if (_DSV41_DRAFT_HEAD_FP8 and xh.dim() == 2 and xh.shape[0] <= 16) else None
        if q8 is not None:
            # dsv41 (ds41-patches/patch_v38.py): draft-only FP8 copy of the shared head shard; the target still scores
            # every draft with its bf16 head, so outputs are unchanged
            from hybrid import dsv41_fp8_gemv as _g8

            local = _g8.fp8_linear(xh.contiguous(), q8[0], q8[1])
        else:
            local = lp._get_logits(xh, head, None, skip_gather=True)
"""),
    ("""    def _sample_sequential_sharded(self, num_reqs: int, sample_hidden: torch.Tensor):
""", """    def _dsv41_draft_head_fp8(self, head):
        \"\"\"dsv41 (ds41-patches/patch_v38.py): (e4m3 codes [N, K] as uint8, fp32 scales [N/128, K/128]) of the head
        shard, built once on first use in row chunks (small transient memory).\"\"\"
        q8 = getattr(self, "_dsv41_q8", None)
        if q8 is not None:
            return q8 or None
        w = head.weight
        if w.dim() != 2 or w.dtype != torch.bfloat16 or w.shape[1] % 128:
            self._dsv41_q8 = False
            return None
        from hybrid import dsv41_fp8_gemv as _g8

        N, K = w.shape
        nb = (N + 127) // 128
        q = torch.empty(N, K, dtype=torch.uint8, device=w.device)
        s = torch.empty(nb, K // 128, dtype=torch.float32, device=w.device)
        for b0 in range(0, nb, 32):
            r0, r1 = b0 * 128, min(N, (b0 + 32) * 128)
            blk = w[r0:r1].float()
            pad = (-blk.shape[0]) % 128
            if pad:
                blk = torch.nn.functional.pad(blk, (0, 0, 0, pad))
            v = blk.view(-1, 128, K // 128, 128)
            sc = v.abs().amax(dim=(1, 3)).clamp_min(1e-12) / 448.0
            s[b0:b0 + sc.shape[0]] = sc
            qq = (v / sc[:, None, :, None]).to(torch.float8_e4m3fn).view(-1, K)[: r1 - r0]
            q[r0:r1] = qq.view(torch.uint8)
        _g8._TUNED.setdefault((N, K), (64, 128, 2, 4, 3))
        self._dsv41_q8 = (q, s)
        print(f"dsv41 draft head fp8: {N}x{K} ({q.numel() / 2**20:.0f} MiB)", flush=True)
        return self._dsv41_q8

    def _sample_sequential_sharded(self, num_reqs: int, sample_hidden: torch.Tensor):
"""),
    ("\nimport torch\n", "\nimport torch\nimport os as _dsv41_os38\n_DSV41_DRAFT_HEAD_FP8 = _dsv41_os38.environ.get(\"DSV41_DRAFT_HEAD_FP8\", \"0\") == \"1\"\n"),
], "patch_v38.py")

patch(f"{sp}/vllm/model_executor/layers/fused_moe/router/dsv41_router_mm.py", [
    ("_BN, _BK, _SPLIT, _WARPS = 64, 128, 8, 4\n",
     "import os as _dsv41_os38\n"
     "# dsv41 (ds41-patches/patch_v38.py): DSV41_ROUTER_BN / DSV41_ROUTER_SPLIT (K must divide by SPLIT * 128)\n"
     "_BN, _BK, _SPLIT, _WARPS = (int(_dsv41_os38.environ.get(\"DSV41_ROUTER_BN\", \"64\")), 128,\n"
     "                            int(_dsv41_os38.environ.get(\"DSV41_ROUTER_SPLIT\", \"8\")), 4)\n"),
], "DSV41_ROUTER_SPLIT")
