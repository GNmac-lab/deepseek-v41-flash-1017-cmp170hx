#include <torch/extension.h>
void wo_a_small(torch::Tensor W, torch::Tensor S, torch::Tensor x, torch::Tensor pos, torch::Tensor cs, torch::Tensor out,
                int64_t G, int64_t R, int64_t D, int64_t HEAD, int64_t NOPE, int64_t WPT, int64_t TPB, int64_t DEPTH);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("wo_a_small", &wo_a_small); }
