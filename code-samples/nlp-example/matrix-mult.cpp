#include <iostream>
#include <numeric>
#include <vector>
#include "oneapi/dnnl/dnnl.hpp"

int main() {
    // Initialize oneDNN engine
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);

    // Set matrix dimensions
    const int M = 4;
    const int N = 3;
    const int K = 5;

    // Create memory descriptors for matrices A, B, and C
    dnnl::memory::dims a_dims = {M, K};
    dnnl::memory::dims b_dims = {K, N};
    dnnl::memory::dims c_dims = {M, N};
    dnnl::memory::desc a_md(a_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    dnnl::memory::desc b_md(b_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    dnnl::memory::desc c_md(c_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);

    // Create memory objects for matrices A, B, and C
    std::vector<float> a_data(M * K);
    std::vector<float> b_data(K * N);
    std::vector<float> c_data(M * N);
    std::iota(a_data.begin(), a_data.end(), 1.0f);
    std::iota(b_data.begin(), b_data.end(), 1.0f);
    std::fill(c_data.begin(), c_data.end(), 0.0f);
    dnnl::memory a_mem(a_md, eng);
    dnnl::memory b_mem(b_md, eng);
    dnnl::memory c_mem(c_md, eng);

    // Create matrix multiplication primitive
    dnnl::inner_product_forward::desc ip_desc(dnnl::prop_kind::forward_training,
                                               a_md, b_md, c_md);
    dnnl::inner_product_forward::primitive_desc ip_pd(ip_desc, eng);

    // Bind memory objects to the primitive
    dnnl::primitive_attr ip_attr;
    dnnl::inner_product_forward(ip_pd).execute(dnnl::stream(eng),
                                                {{DNNL_ARG_SRC, a_mem},
                                                 {DNNL_ARG_WEIGHTS, b_mem},
                                                 {DNNL_ARG_BIAS, c_mem}});

    // Print the result matrix C
    auto c_ptr = c_mem.map_data<float>();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << c_ptr[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    c_mem.unmap_data(c_ptr);

    return 0;
}

