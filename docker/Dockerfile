# syntax=docker/dockerfile:1
# DeepSeek-V4.1-Flash-1017 serving engine for 4x NVIDIA CMP 170HX (sm_80).
#
# The official vLLM DeepSeek-V4.1 image, plus the upstream projects below at fixed commits, plus this recipe's
# patches (../patches), applied in the order the engine was developed and measured. NOTICE.md lists every upstream,
# its licence and what is changed. Build with docker/build.sh.
ARG BASE=vllm/vllm-openai@sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d

# ---------------------------------------------------------------------------------------------------------------
# Upstream sources, each fetched at a full commit SHA.
FROM ${BASE} AS src
WORKDIR /src
RUN set -eu; \
    fetch() { mkdir -p "$1"; curl -fsSL "https://codeload.github.com/$2/tar.gz/$3" | tar xz -C "$1" --strip-components=1; }; \
    fetch dsv41reap-pp zebgop-ops/dsv41reap-pp            138e74881130a1510b8bcf22cc01456bd6e46219; \
    fetch kaka         kaka86mm/dsv41-flash-pp4-170hx     950f4c003c074be9434735a37ec67466559909ec; \
    fetch exllamav3    turboderp-org/exllamav3            d3739fd393337b1ff4d6c2a342b12f0c87a9592f; \
    fetch vllm-exl3    vcruz305/vllm-exl3                 223e246ff8da552d2ace976c4bcbed804a8ec04c; \
    fetch tc-moe       CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE 88eb6e13de0ea5491579fddb195116a4176f4bc7; \
    fetch flashinfer   flashinfer-ai/flashinfer           946200de1ae94fc93fdd0926f0a13afd1fa7f0f1; \
    # FlashInfer 0.7.0.post1's PCIe IPC all-reduce, laid out the way patches/install_fipcie.py reads it
    mkdir -p fipcie/data/csrc fipcie/data/include/flashinfer/comm fipcie/jit fipcie/trace/templates fipcie/comm; \
    cp flashinfer/flashinfer/comm/pcie_ipc_*.py fipcie/comm/; \
    cp flashinfer/flashinfer/jit/comm.py fipcie/jit/; \
    cp flashinfer/flashinfer/trace/templates/comm.py fipcie/trace/templates/; \
    cp flashinfer/csrc/pcie_ipc_all_reduce.cu fipcie/data/csrc/; \
    cp flashinfer/include/flashinfer/comm/pcie_ipc_all_reduce.cuh fipcie/data/include/flashinfer/comm/; \
    rm -rf flashinfer

# ---------------------------------------------------------------------------------------------------------------
# exllamav3 extension with the cooperative MoE decode kernel changes (branch-free tile, wide down tiles, parallel
# epilogue, decoded weights as the MMA A operand, per-expert widths, pointer-walk loads, row and lane clamps).
# Built separately, as during development, then dropped in.
FROM ${BASE} AS exl3ext
COPY --from=src /src/exllamav3 /tmp/exllamav3
COPY patches/patch_coop_branchfree.py patches/patch_coop_down_wide16.py patches/patch_coop_down_epilogue.py \
     patches/patch_coop_mma_swap.py patches/patch_coop_widths.py patches/patch_coop_ptr.py patches/patch_coop_rowclamp.py \
     patches/patch_coop_noclamp.py patches/patch_coop_laneclamp.py /tmp/
ENV TORCH_CUDA_ARCH_LIST=8.0
RUN set -eu; Q=/tmp/exllamav3/exllamav3/exllamav3_ext/quant; \
    cp -n /usr/local/lib/python3.12/dist-packages/nvidia/cu13/include/*.h /usr/local/cuda/include/; \
    python3 /tmp/patch_coop_branchfree.py $Q/exl3_moe_coop_kernel.cuh; \
    python3 /tmp/patch_coop_down_wide16.py $Q/exl3_moe_coop_kernel.cuh $Q/exl3_moe_coop.cu; \
    python3 /tmp/patch_coop_down_epilogue.py $Q/exl3_moe_coop_kernel.cuh $Q/exl3_moe_coop.cu; \
    python3 /tmp/patch_coop_mma_swap.py $Q/exl3_moe_coop_kernel.cuh; \
    python3 /tmp/patch_coop_widths.py /tmp/exllamav3/exllamav3/exllamav3_ext; \
    python3 /tmp/patch_coop_ptr.py $Q/exl3_moe_coop_kernel.cuh; \
    python3 /tmp/patch_coop_rowclamp.py $Q/exl3_moe_coop_kernel.cuh; \
    python3 /tmp/patch_coop_noclamp.py $Q/exl3_moe_coop_kernel.cuh; \
    python3 /tmp/patch_coop_laneclamp.py $Q/exl3_moe_coop_kernel.cuh; \
    cd /tmp/exllamav3 && MAX_JOBS=${MAX_JOBS:-16} python3 setup.py build_ext > /tmp/build_ext.log 2>&1 || { tail -40 /tmp/build_ext.log; exit 1; }; \
    find /tmp/exllamav3/build -name "exllamav3_ext*.so" -exec cp {} /tmp/exllamav3_ext.so \;; ls -la /tmp/exllamav3_ext.so

# ---------------------------------------------------------------------------------------------------------------
FROM ${BASE}
ARG SP=/usr/local/lib/python3.12/dist-packages
ENV TORCH_CUDA_ARCH_LIST=8.0
COPY patches/ /opt/ds41-patches/

# 1. zebgop-ops/dsv41reap-pp overlay: sm_80 sparse attention, PP shadow KV, DSpark V1, Engram SSD row store
RUN --mount=type=bind,from=src,source=/src,target=/src set -eu; O=/src/dsv41reap-pp/overlay; \
    cp -r $O/vllm/. ${SP}/vllm/; mkdir -p ${SP}/hybrid ${SP}/engram_ssd; cp -r $O/hybrid/. ${SP}/hybrid/; cp -r $O/engram_ssd/. ${SP}/engram_ssd/; \
    bash ${SP}/hybrid/build.sh; \
    g++ -O2 -Wall -std=c++17 -shared -fPIC -pthread ${SP}/engram_ssd/row_store.cpp -o ${SP}/engram_ssd/librow_store.so

# 2. kaka86mm/dsv41-flash-pp4-170hx patches
RUN --mount=type=bind,from=src,source=/src,target=/src set -eu; K=/src/kaka/patches; \
    python3 $K/pp_relay_img_ids.py ${SP}/vllm/models/deepseek_v4_1/nvidia/model.py; \
    python3 $K/engram_sm80_fp8.py ${SP}/vllm/models/deepseek_v4_1/common/engram.py; \
    python3 $K/moe_cand_key_fix.py ${SP}/vllm/v1/worker/gpu/model_runner.py; \
    python3 $K/patch_dspark_meta_cache.py ${SP}/hybrid/dspark_proposer.py

# 3. exllamav3 1.5.3 for sm_80, then the vcruz305/vllm-exl3 plugin built against its headers
RUN --mount=type=bind,from=src,source=/src,target=/src set -eu; cp -n ${SP}/nvidia/cu13/include/*.h /usr/local/cuda/include/; \
    cp -r /src/exllamav3 /tmp/exllamav3; cd /tmp/exllamav3; MAX_JOBS=${MAX_JOBS:-16} pip install --no-deps --no-build-isolation --no-cache-dir .; \
    python3 -c "import torch, exllamav3_ext; print('exllamav3_ext ok', hasattr(exllamav3_ext, 'exl3_moe'))"
RUN --mount=type=bind,from=src,source=/src,target=/src set -eu; cp -r /src/vllm-exl3 /tmp/vllm-exl3; \
    # exllamav3 1.5.3 added a 7th template parameter to exl3_gemv_kernel; match the forward declaration
    sed -i "s/^template <int, bool, int, int, int, bool>$/template <int, bool, int, int, int, bool, bool>/" /tmp/vllm-exl3/csrc/*.cu; \
    grep -q "int, int, int, bool, bool>" /tmp/vllm-exl3/csrc/p2b_batched.cu; \
    cd /tmp/vllm-exl3; EXL3_EXT_INCLUDE=/tmp/exllamav3/exllamav3/exllamav3_ext MAX_JOBS=${MAX_JOBS:-16} pip install --no-deps --no-build-isolation --no-cache-dir .; \
    python3 -c "import torch, vllm_exl3; print('vllm_exl3 ok')"; rm -rf /tmp/exllamav3 /tmp/vllm-exl3

# 4. TP4 stack: tensor-core EXL3 MoE (CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE), hybrid-checkpoint loading
RUN --mount=type=bind,from=src,source=/src,target=/src set -eu; P=/opt/ds41-patches; \
    cp -r /src/tc-moe /opt/dsv41-tc-moe-src; cd /opt/dsv41-tc-moe-src; \
    python3 vllm_exl3/tc_moe/build_standalone.py --out /opt/dsv41-tc-moe --build-dir /tmp/tcbuild; \
    cp exl3_tc_hook.py /opt/dsv41-tc-moe/; rm -rf /tmp/tcbuild; \
    python3 $P/vllm-exl3/patch_tc_hook.py ${SP}/vllm_exl3/exl3.py; \
    python3 $P/patch_skip_file_experts_only.py ${SP}/vllm/model_executor/model_loader/weight_utils.py; \
    sed -i 's/_os.environ.get("DSV41_MODEL_DIR") or get_current_vllm_config/_os.environ.get("DSV41_ENGRAM_DIR") or _os.environ.get("DSV41_MODEL_DIR") or get_current_vllm_config/' ${SP}/vllm/models/deepseek_v4_1/common/engram.py; \
    grep -q DSV41_ENGRAM_DIR ${SP}/vllm/models/deepseek_v4_1/common/engram.py; \
    python3 $P/vllm-exl3/patch_prescan_aligned.py ${SP}/vllm_exl3/mixed_k_guard.py; \
    python3 $P/vllm-exl3/patch_prescan_offset.py ${SP}/vllm_exl3/exl3.py

# 5. Engram row store, attention decode split-K, cooperative MoE kernel
COPY --from=exl3ext /tmp/exllamav3_ext.so ${SP}/exllamav3_ext.cpython-312-x86_64-linux-gnu.so
COPY --from=exl3ext /tmp/exllamav3/exllamav3/exllamav3_ext/quant/exl3_moe_coop_kernel.cuh /tmp/exllamav3/exllamav3/exllamav3_ext/quant/exl3_moe_coop.cu /tmp/exllamav3/exllamav3/exllamav3_ext/quant/exl3_moe_coop.cuh ${SP}/exllamav3/exllamav3_ext/quant/
RUN set -eu; P=/opt/ds41-patches; \
    python3 $P/patch_attn_splitk.py ${SP}/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py; \
    python3 $P/patch_row_store_overlap.py ${SP}/engram_ssd/row_store.cpp; \
    g++ -O2 -Wall -std=c++17 -shared -fPIC -pthread ${SP}/engram_ssd/row_store.cpp -o ${SP}/engram_ssd/librow_store.so; \
    python3 $P/patch_engram_async.py ${SP}/vllm/models/deepseek_v4_1/common/engram.py; \
    python3 -c "import torch, exllamav3_ext; print('exllamav3_ext ok', hasattr(exllamav3_ext, 'exl3_moe_coop'))"

# 6. Decode-step patches, in development order (each one measured; see docs/how-it-works.md)
RUN set -eu; P=/opt/ds41-patches; \
    python3 $P/vllm-exl3/patch_exl3_coop_fastpath.py ${SP}/vllm_exl3/exl3.py; \
    python3 $P/patch_dspark_shard.py ${SP}/hybrid/dspark_proposer.py; \
    cp $P/dsv41_router_mm.py ${SP}/vllm/model_executor/layers/fused_moe/router/; \
    python3 $P/patch_router_mm.py ${SP}/vllm/model_executor/layers/fused_moe/router/gate_linear.py; \
    python3 $P/patch_greedy_argmax.py ${SP}/vllm/v1/worker/gpu_model_runner.py; \
    python3 $P/patch_share_t2r.py ${SP}/vllm/v1/worker/gpu_model_runner.py ${SP}/hybrid/dspark_proposer.py; \
    python3 $P/vllm-exl3/patch_exl3_coop_prep.py ${SP}/vllm_exl3/exl3.py; \
    python3 $P/patch_shared_rebalance.py ${SP}; \
    python3 $P/patch_mhc_fused.py ${SP}

# 7. FlashInfer PCIe IPC all-reduce (0.7.0.post1, vendored into the image's 0.6.18), DeepInfer-SM80 attention
RUN --mount=type=bind,from=src,source=/src,target=/src set -eu; P=/opt/ds41-patches; \
    python3 $P/install_fipcie.py ${SP}/flashinfer /src/fipcie; \
    cd ${SP}; (patch -p1 -N < $P/deepinfer_sm80_attn.patch || true); \
    grep -q "_decode_e8m0_scales_bf16_triton" ${SP}/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py; \
    rm -f ${SP}/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py.rej ${SP}/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py.orig; \
    patch -p1 -N < $P/deepinfer_sm80_rocm.patch; rm -f ${SP}/vllm/models/deepseek_v4_1/amd/rocm.py.orig

# 8. FP8 wo_a, ring all-reduce, FP8-only wo_a weights, deployment-sized long-context workspaces
RUN set -eu; P=/opt/ds41-patches; \
    python3 $P/patch_v32_wo_a.py ${SP} $P/dsv41_wo_a.py; \
    python3 $P/vllm-exl3/patch_exl3_ids_cast.py ${SP}; \
    python3 $P/patch_true_ring.py ${SP}/flashinfer; \
    python3 $P/patch_wo_a_free.py ${SP}; \
    python3 $P/patch_long_ctx_mem.py ${SP}

# 9. Decode speedups (1.1.0), in development order: draft-only FP8 head and router split-K knobs, greedy verification
#    from the vocab-parallel argmax ids, the DSpark sampling loop as one CUDA graph, the fused Markov step
RUN set -eu; P=/opt/ds41-patches; \
    python3 $P/patch_v38.py ${SP} $P/dsv41_fp8_gemv.py; \
    python3 $P/patch_greedy_ids.py ${SP}/vllm; \
    python3 $P/patch_dspark_sample_graph.py ${SP}/hybrid/dspark_proposer.py; \
    cp $P/dsv41_markov.py ${SP}/hybrid/dsv41_markov.py; \
    python3 $P/patch_dspark_markov_fused.py ${SP}/hybrid/dspark_proposer.py; \
    python3 -m py_compile ${SP}/hybrid/dspark_proposer.py ${SP}/hybrid/dsv41_markov.py ${SP}/hybrid/dsv41_fp8_gemv.py

# 10. MXFP8 linears on one fragment-native weight copy (CUDA GEMV for M <= 8, smem-staged kernel for M <= 64,
#     exact dequant + cuBLAS above) and the decode wo_a as one CUDA kernel with the inverse RoPE fused
RUN set -eu; P=/opt/ds41-patches; \
    for e in dsv41_fp8w dsv41_woa; do \
      cp -r $P/$e /tmp/$e; cd /tmp/$e; MAX_JOBS=${MAX_JOBS:-16} python3 setup.py build_ext > /tmp/$e.log 2>&1 || { tail -40 /tmp/$e.log; exit 1; }; \
      find /tmp/$e/build -name "${e}_ext*.so" -exec cp {} ${SP}/ \;; cd /; rm -rf /tmp/$e; \
    done; \
    python3 $P/patch_fp8_gemv_cuda.py ${SP}; \
    python3 $P/patch_wo_a_cuda.py ${SP}; \
    python3 -c "import torch, dsv41_fp8w_ext, dsv41_woa_ext; print('dsv41_fp8w_ext, dsv41_woa_ext ok')"

# 11. Router GEMM before the shared experts, sm80 sparse-MLA decode launch (BLOCK_K 16, 8 warps), gemm_mid table
RUN set -eu; P=/opt/ds41-patches; \
    python3 $P/patch_gate_first.py ${SP}; \
    python3 $P/patch_attn_cfg.py ${SP}/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py; \
    python3 $P/patch_mid_table.py ${SP}; \
    python3 -m py_compile ${SP}/vllm/model_executor/layers/fused_moe/runner/moe_runner.py \
      ${SP}/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py ${SP}/vllm/model_executor/kernels/linear/mxfp8/dsv41_fp8w.py

# 12. Runtime variants applied before graph capture (DSV41_VARIANTS; empty = none): one-launch decode metadata,
#     repeated-H2D skips, and confidence-gated verification (DSV41_GATE_TAU, 0 = off)
RUN set -eu; P=/opt/ds41-patches; \
    python3 $P/dsv41_variants/patch_variants.py ${SP}/vllm $P/dsv41_variants; \
    python3 -m py_compile ${SP}/vllm/dsv41_variants.py ${SP}/vllm/v1/worker/gpu_worker.py ${SP}/vllm/dsv41_variants_d/*.py

# 13. Prefill and decode (1.2.0), in development order: prefill TP sums beside compute on a copy-engine owner sum
#     (dsv41_ce_ring, dsv41_split_reduce), Morrowmake's prefill and decode tensor-core MoE schedules built for the TP
#     expert shapes (dsv41_moe_try, dsv41_moe_v2try), Engram row prefetch, prefill tiles for the fused mHC kernel,
#     packed sparse-attention prefill gathers, prefill attention micro-batches (dsv41_attn_slices; the variant list
#     is in patches/dsv41_variants/dsv41_variants.py), and the per-step DSpark verify width (DSV41_DYNK)
RUN set -eu; P=/opt/ds41-patches; \
    python3 $P/tc_moe_try/build_try.py --out /opt/dsv41-tc-moe --build-dir /tmp/trybuild; \
    python3 $P/tc_moe_v2try/build_v2try.py --out /opt/dsv41-tc-moe --build-dir /tmp/v2trybuild; \
    rm -rf /tmp/trybuild /tmp/v2trybuild; \
    cp -r $P/tc_moe_try /opt/dsv41-tc-moe-src/moe_prefill_try; cp -r $P/tc_moe_v2try /opt/dsv41-tc-moe-src/moe_v2_try; \
    cp $P/dsv41_dynk.py ${SP}/hybrid/dsv41_dynk.py; \
    python3 $P/patch_dynk.py; \
    python3 -m py_compile ${SP}/hybrid/dsv41_dynk.py ${SP}/hybrid/dspark_proposer.py ${SP}/vllm/v1/core/sched/scheduler.py; \
    test -s /opt/dsv41-tc-moe/_dsv41_mm_try.abi3.so; test -s /opt/dsv41-tc-moe/_dsv41_mm_v2try.abi3.so
