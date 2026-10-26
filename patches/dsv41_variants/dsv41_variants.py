# dsv41: runtime variants = small modules that replace engine functions in each worker, each measured on its own.
#
# Each module in vllm/dsv41_variants_d/ defines SETATTR, a [{"target", "source"}] list of attributes to replace,
# and may self-test on import. apply() runs once per worker after the model is loaded and before graph capture, so the
# variants are in place for every captured graph. DSV41_VARIANTS lists the modules (comma separated); empty = none,
# which is the A side of a cold A/B on the same image. patches/dsv41_variants/patch_variants.py installs this.
import importlib.util
import os
import sys

from vllm.logger import init_logger

logger = init_logger(__name__)
DEFAULT = "dsv41_topk_meta,dsv41_h2d_lean,dsv41_ragged_lean,dsv41_gate,dsv41_ce_ring,dsv41_split_reduce,dsv41_moe_try,dsv41_engram_prefetch,dsv41_mhc_pcfg,dsv41_moe_v2try,dsv41_prefill_pack,dsv41_attn_slices"
DIR = os.path.join(os.path.dirname(__file__), "dsv41_variants_d")
_DONE = False


def _resolve(target: str):
    parts = target.split(".")
    for i in range(len(parts) - 1, 0, -1):
        try:
            owner = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for p in parts[i:-1]:
            owner = getattr(owner, p)
        return owner, parts[-1]
    raise ImportError(target)


def apply() -> None:
    global _DONE
    if _DONE:
        return
    _DONE = True
    names = [n.strip() for n in os.environ.get("DSV41_VARIANTS", DEFAULT).split(",") if n.strip()]
    for name in names:
        spec = importlib.util.spec_from_file_location(name, os.path.join(DIR, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)          # self-tests raise here, which stops the worker
        for s in mod.SETATTR:
            owner, attr = _resolve(s["target"])
            src, sattr = _resolve(s["source"])
            setattr(owner, attr, getattr(src, sattr))
    logger.info("dsv41 variants applied: %s", ", ".join(names) or "none")
