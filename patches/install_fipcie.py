#!/usr/bin/env python3
"""Vendor FlashInfer's PCIe IPC all-reduce (flashinfer 0.7.0.post1, taken from the Morrowmake 1.8.0 image) into the
image's flashinfer 0.6.18, which lacks it. vLLM already has the wrapper (flashinfer_pcie_ipc_all_reduce.py) and turns it
on with VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1.
usage: install_fipcie.py <flashinfer package dir> <extracted source dir>"""
import shutil
import sys
from pathlib import Path

dst, src = Path(sys.argv[1]), Path(sys.argv[2])
for rel in ("comm/pcie_ipc_ar.py", "comm/pcie_ipc_policy.py", "comm/pcie_ipc_topology.py", "comm/pcie_ipc_tuning.py",
            "data/csrc/pcie_ipc_all_reduce.cu", "data/include/flashinfer/comm/pcie_ipc_all_reduce.cuh"):
    (dst / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / rel, dst / rel)

# JIT spec
jc = dst / "jit/comm.py"
s = jc.read_text()
if "def gen_pcie_ipc_comm_module" not in s:
    t = (src / "jit/comm.py").read_text()
    i = t.index("def gen_pcie_ipc_comm_module")
    j = t.index("\ndef ", i + 10)
    jc.write_text(s.rstrip("\n") + "\n\n\n" + t[i:j].rstrip("\n") + "\n")

# trace template (reference + init + template)
tc = dst / "trace/templates/comm.py"
s = tc.read_text()
if "pcie_ipc_all_reduce_trace" not in s:
    t = (src / "trace/templates/comm.py").read_text()
    i = t.index("def _pcie_ipc_all_reduce_reference")
    j = t.index("pcie_ipc_all_reduce_trace = TraceTemplate(")
    k = t.index("\n)\n", j) + 3
    tc.write_text(s.rstrip("\n") + "\n\n\n" + t[i:k])

# exports
ci = dst / "comm/__init__.py"
s = ci.read_text()
if "PcieIpcAllReduceWorkspace" not in s:
    s = s.rstrip("\n") + """


# ds41: vendored FlashInfer PCIe IPC all-reduce (0.7.0.post1)
from .pcie_ipc_ar import PcieIpcAllReduceWorkspace as PcieIpcAllReduceWorkspace  # noqa: E402
from .pcie_ipc_ar import gen_pcie_ipc_comm_module as gen_pcie_ipc_comm_module  # noqa: E402
from .pcie_ipc_ar import get_pcie_ipc_comm_module as get_pcie_ipc_comm_module  # noqa: E402
from .pcie_ipc_policy import IpcLaunchConfig as PcieIpcLaunchConfig  # noqa: E402
from .pcie_ipc_policy import IpcVariant as PcieIpcVariant  # noqa: E402
from .pcie_ipc_tuning import PCIE_IPC_CUSTOM_OP as PCIE_IPC_CUSTOM_OP  # noqa: E402
from .pcie_ipc_tuning import default_cache_path as pcie_ipc_default_cache_path  # noqa: E402
from .pcie_ipc_policy import get_pcie_ipc_launch_config as get_pcie_ipc_launch_config  # noqa: E402
from .pcie_ipc_topology import probe_pcie_ipc_rank_topology as probe_pcie_ipc_rank_topology  # noqa: E402
from .pcie_ipc_topology import resolve_pcie_ipc_profile as resolve_pcie_ipc_profile  # noqa: E402
"""
    ci.write_text(s)
print("installed into", dst)
