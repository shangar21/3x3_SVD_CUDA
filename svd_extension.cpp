#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "svd3x3/svd3x3/svd3_cuda.h"

// Wrapper for launching the kernel
std::vector<torch::Tensor> svd_cuda(torch::Tensor input) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
	int batch_size = input.size(0);
    auto u = torch::zeros({batch_size, 3, 3}, options);
    auto s = torch::zeros({batch_size, 3}, options);
    auto v = torch::zeros({batch_size, 3, 3}, options);

    float* input_ptr = input.data_ptr<float>();
    float* u_ptr = u.data_ptr<float>();
    float* s_ptr = s.data_ptr<float>();
    float* v_ptr = v.data_ptr<float>();

    launch_svd_kernel_batch(input_ptr, u_ptr, s_ptr, v_ptr, batch_size);

    return {u, s, v};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("svd_cuda", &svd_cuda, "SVD computation using CUDA");
}

