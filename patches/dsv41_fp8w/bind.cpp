#include <torch/extension.h>
void gemv_small(torch::Tensor W, torch::Tensor SC, torch::Tensor x, torch::Tensor out, int64_t N, int64_t K,
                int64_t TW, int64_t WPT, int64_t TPB, int64_t DEPTH, int64_t HINT);
void gemm_mid(torch::Tensor W, torch::Tensor SC, torch::Tensor x, torch::Tensor out, torch::Tensor part, torch::Tensor cnt,
              int64_t N, int64_t K, int64_t TW, int64_t KS, int64_t PC, int64_t DEPTH, int64_t WARPS);
void dequant(torch::Tensor W, torch::Tensor SC, torch::Tensor out, int64_t N, int64_t K);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemv_small", &gemv_small);
    m.def("gemm_mid", &gemm_mid);
    m.def("dequant", &dequant);
}
