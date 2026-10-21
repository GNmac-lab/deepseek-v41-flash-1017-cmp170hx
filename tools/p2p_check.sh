#!/usr/bin/env bash
# Check what TP4 needs from the PCIe fabric: link width/speed per card and GPU peer-to-peer read/write support.
set -euo pipefail
echo "== PCIe links (TP4 wants x16; CMP 170HX tops out at PCIe 2.0 = 5 GT/s)"
nvidia-smi --query-gpu=index,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,pcie.link.width.max --format=csv
echo "== P2P read support (OK on every pair = NCCL and the PCIe IPC all-reduce can use P2P)"
nvidia-smi topo -p2p r
echo "== P2P write support"
nvidia-smi topo -p2p w
echo "Optional exact all-reduce check over P2P (server stopped):"
echo "  nvcc -O3 -std=c++17 -arch=sm_80 bench/ring_ar.cu -o /tmp/ring_ar && /tmp/ring_ar   # every line must say 'exact'"
