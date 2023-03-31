#include <iostream>
#include <CL/sycl.hpp>
#include "oneapi/dnnl.hpp"

int main() {
    constexpr auto batch_size = 32;
    constexpr auto max_seq_length = 128;
    constexpr auto embedding_dim = 300;
    constexpr auto num_filters = 100;
    constexpr auto filter_sizes = std::array<int, 3>{ 3, 4, 5 };
    constexpr auto num_classes = 10;

    // Create the memory descriptors for the input data and labels
    auto input_md = dnnl::memory::desc({ batch_size, max_seq_length, embedding_dim },
                                       dnnl::memory::data_type::f32,
                                       dnnl::memory::format_tag::tnc);
    auto label_md = dnnl::memory::desc({ batch_size, num_classes },
                                       dnnl::memory::data_type::f32,
                                       dnnl::memory::format_tag::nc);

    // Create the convolution descriptor and memory descriptors for weights and bias
    auto conv_desc = dnnl::convolution_forward::desc(dnnl::prop_kind::forward_inference,
                                                     dnnl::algorithm::convolution_direct,
                                                     input_md, dnnl::memory::desc({ num_filters, embedding_dim, filter_sizes[0] },
                                                     dnnl::memory::data_type::f32, dnnl::memory::format_tag::oihw),
                                                     dnnl::memory::desc({ num_filters }, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x),
                                                     dnnl::memory::desc({ num_filters, 1, 1 }, dnnl::memory::data_type::f32, dnnl::memory::format_tag::oihw),
                                                     dnnl::memory::desc({ 1 }, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x),
                                                     std::array<int, 2>{ 1, 1 },
                                                     std::array<int, 2>{ 0, 0 },
                                                     std::array<int, 2>{ 1, 1 });
    auto conv_weights_md = conv_desc.weights_desc();
    auto conv_bias_md = conv_desc.bias_desc();

    // Create the memory objects for the input data, labels, weights, and bias
    auto input_mem = dnnl::memory(input_md, queue);
    auto label_mem = dnnl::memory(label_md, queue);
    auto conv_weights_mem = dnnl::memory(conv_weights_md, queue);
    auto conv_bias_mem = dnnl::memory(conv_bias_md, queue);

    // Initialize the input data, labels, weights, and bias

    // Create the descriptor and primitive for the convolution layer
    auto conv_prim_desc = dnnl::convolution_forward::primitive_desc(conv_desc, queue.get_context());

    // Create the memory descriptor and object for the output of the convolution layer
    auto conv_output_md = conv_prim_desc.dst_desc();
    auto conv_output_mem = dnnl::memory(conv_output_md, queue);

    // Create the descriptor and primitive for the ReLU activation layer
    auto relu_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference,
                                                 dnnl::algorithm::eltwise_relu,
                                                 conv_output_md,
                                                 0.0f);
    auto relu_prim_desc = dnnl::eltwise_forward::
