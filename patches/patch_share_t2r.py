#!/usr/bin/env python3
"""Share the per-token request map across attention groups (gpu_model_runner.py, hybrid/dspark_proposer.py).

CommonAttentionMetadata.token_to_req_indices() builds the token -> request index map from query_start_loc (a few
small kernels) and caches it on that metadata object, but every KV-cache group builds from its own shallow copy, so
on DeepSeek-V4.1 the map was rebuilt ~33 times per decode step for the target and ~25 times for the DSpark drafter
(profiled 2026-10-20, ~125 us of GPU plus launch gaps). The map depends only on query_start_loc, which all groups
share. Now the first group's map is handed to the later groups: the target loop puts it back on cm_base before the
next copy (cudagraph capture runs the same loop, so captured graphs read the same buffer that is refreshed first
every step); the drafter loop passes it to later per-group metadata with the same query_start_loc tensor.
Same values. DSV41_SHARE_T2R=0 = stock. usage: patch_share_t2r.py <gpu_model_runner.py> <dspark_proposer.py>
"""
import sys

rp, dp = sys.argv[1], sys.argv[2]
r = open(rp).read()
d = open(dp).read()
if "DSV41_SHARE_T2R" in r:
    print("already patched"); sys.exit(0)

old = '''                else:
                    _build_attn_group_metadata(kv_cache_gid, attn_gid, cm)
'''
assert r.count(old) == 1, r.count(old)
r = r.replace(old, old + '''
            # dsv41 (ds41-patches/patch_share_t2r.py): later groups reuse this group's token -> request map
            if (
                _DSV41_SHARE_T2R
                and ubatch_slices is None
                and getattr(cm, "_token_to_req_indices_cache", None) is not None
                and getattr(cm_base, "_token_to_req_indices_cache", None) is None
            ):
                cm_base._token_to_req_indices_cache = cm._token_to_req_indices_cache
''')
anchor = "class GPUModelRunner("
assert r.count(anchor) == 1
r = r.replace(anchor, '_DSV41_SHARE_T2R = os.environ.get("DSV41_SHARE_T2R", "1") != "0"\n\n\n' + anchor)

old_d = '''        per_group, per_layer = [], {}
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            cad = self._per_gid_cad.get(gid, common_attn_metadata)
'''
assert d.count(old_d) == 1
d = d.replace(old_d, '''        per_group, per_layer = [], {}
        # dsv41 (ds41-patches/patch_share_t2r.py): one token -> request map for all draft groups
        _share = os.environ.get("DSV41_SHARE_T2R", "1") != "0"
        _first = None
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            cad = self._per_gid_cad.get(gid, common_attn_metadata)
            if (
                _share and _first is not None
                and getattr(cad, "_token_to_req_indices_cache", None) is None
                and getattr(_first, "_token_to_req_indices_cache", None) is not None
                and cad.query_start_loc.data_ptr() == _first.query_start_loc.data_ptr()
                and cad.num_actual_tokens == _first.num_actual_tokens
            ):
                cad._token_to_req_indices_cache = _first._token_to_req_indices_cache
''')
old_d2 = '''            md = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=cad, draft_index=draft_index
            )
'''
assert d.count(old_d2) == 1
d = d.replace(old_d2, old_d2 + '''            if _first is None:
                _first = cad
''')
open(rp, "w").write(r)
open(dp, "w").write(d)
print("patched", rp, dp)
