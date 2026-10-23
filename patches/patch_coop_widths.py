#!/usr/bin/env python3
"""dsv41: per-expert intermediate widths in the cooperative MoE kernels (exllamav3).

exl3_moe_coop_set_widths(int32 device tensor [n_local] or None) sets the width table for the next exl3_moe_coop call
on this thread (consumed by that call, so a CUDA graph capture bakes each layer's own table into its launch). p.I
stays the scratch stride (the widest expert); kernel A skips the column groups beyond an expert's width and tiles
its gate/up GEMV with width / 16 tiles, kernel B runs the down GEMV over width / 16 k-slices. Every width must be a
multiple of 128 and <= p.I. With no table (or a uniform one) the kernels behave exactly as before.
usage: patch_coop_widths.py <exllamav3_ext dir>"""
import sys

ext = sys.argv[1]
q = f"{ext}/quant"


def patch(p, pairs, mark):
    s = open(p).read()
    if mark in s:
        print("already patched", p)
        return
    for old, new in pairs:
        assert s.count(old) == 1, (p, s.count(old), old[:80])
        s = s.replace(old, new)
    open(p, "w").write(s)
    print("patched", p)


M = "dsv41 coop widths"
patch(f"{q}/exl3_moe_coop.cuh", [
    ("""    int n_local;                // entries in the pointer tables
""", """    int n_local;                // entries in the pointer tables
    const int* inter_w;         // dsv41 coop widths: per local expert intermediate width (nullable = I)
"""),
    ("""// Tensor front end (prepare + run; the test entry point)
""", """// dsv41 coop widths: per-local-expert widths (int32 device tensor) for the next exl3_moe_coop call on this thread
void exl3_moe_coop_set_widths(const c10::optional<at::Tensor>& inter_w);

// Tensor front end (prepare + run; the test entry point)
"""),
], M)

patch(f"{q}/exl3_moe_coop.cu", [
    ("""void exl3_moe_coop
(
    const at::Tensor& x,""", """// dsv41 coop widths: one-shot per-thread width table for the next exl3_moe_coop call
static thread_local const int* t_dsv41_inter_w = nullptr;
static thread_local int64_t t_dsv41_inter_n = 0;

void exl3_moe_coop_set_widths(const c10::optional<at::Tensor>& inter_w)
{
    if (!inter_w) { t_dsv41_inter_w = nullptr; t_dsv41_inter_n = 0; return; }
    TORCH_CHECK_DTYPE(inter_w.value(), kInt);
    TORCH_CHECK(inter_w->is_cuda() && inter_w->is_contiguous(), "exl3_moe_coop_set_widths: contiguous int32 CUDA tensor");
    t_dsv41_inter_w = (const int*) inter_w->data_ptr();
    t_dsv41_inter_n = inter_w->numel();
}

void exl3_moe_coop
(
    const at::Tensor& x,"""),
    ("""    p.min_expert = min_expert;
    p.max_expert = max_expert;
    exl3_moe_coop_run(p, K_gu, K_d, cb, x, sel, rw, sh_out);""", """    p.min_expert = min_expert;
    p.max_expert = max_expert;
    p.inter_w = t_dsv41_inter_w;
    if (p.inter_w) TORCH_CHECK(t_dsv41_inter_n >= p.n_local, "exl3_moe_coop: width table shorter than the pointer tables");
    t_dsv41_inter_w = nullptr;
    t_dsv41_inter_n = 0;
    exl3_moe_coop_run(p, K_gu, K_d, cb, x, sel, rw, sh_out);"""),
], M)

patch(f"{q}/exl3_moe_coop_kernel.cuh", [
    # kernel A: per-expert column extent and tile count
    ("""    const int local = slot_info(p, sh_rows[0]).local;
    const int kslices_all = p.Hi / 16;
    const int k_begin = kslices_all * ks / p.ksplit_a;
    const int k_end = kslices_all * (ks + 1) / p.ksplit_a;
""", """    const int local = slot_info(p, sh_rows[0]).local;
    // dsv41 coop widths: this expert's intermediate width; groups past it have no columns
    const int I_e = p.inter_w ? p.inter_w[local] : p.I;
    if (group * TCOLS >= I_e) return;
    const int kslices_all = p.Hi / 16;
    const int k_begin = kslices_all * ks / p.ksplit_a;
    const int k_end = kslices_all * (ks + 1) / p.ksplit_a;
"""),
    ("""        gemv_tile<bits, cb, WIDE, HALF>(B32, A2, a_stride2, sh_rows, nrows, C, p.I, p.gu_f32, k_begin, k_end, p.I / 16, group, sh_red, sh_stage);""",
     """        gemv_tile<bits, cb, WIDE, HALF>(B32, A2, a_stride2, sh_rows, nrows, C, p.I, p.gu_f32, k_begin, k_end, I_e / 16, group, sh_red, sh_stage);"""),
    # kernel B: per-expert k extent
    ("""    const int local = slot_info(p, sh_rows[0]).local;
    const int kslices_all = p.I / 16;
""", """    const int local = slot_info(p, sh_rows[0]).local;
    const int kslices_all = (p.inter_w ? p.inter_w[local] : p.I) / 16;   // dsv41 coop widths
"""),
], M)

patch(f"{ext}/bindings.cpp", [
    ("""    m.def("exl3_moe_coop", &exl3_moe_coop, "exl3_moe_coop");
""", """    m.def("exl3_moe_coop", &exl3_moe_coop, "exl3_moe_coop");
    m.def("exl3_moe_coop_set_widths", &exl3_moe_coop_set_widths, "dsv41 coop widths: per-expert widths for the next coop call");
"""),
], M)
