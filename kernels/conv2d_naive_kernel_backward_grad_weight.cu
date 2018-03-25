#define CUDA_KERNEL_LOOP(i, n)                          \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n);                                       \
         i += blockDim.x * gridDim.x)

extern "C" __global__ void conv2d_naive_backward_grad_weight_kernel(
    const ${Dtype} *const top_diff, const ${Dtype} *const bottom_data, ${Dtype} *const buffer_data)
{
    CUDA_KERNEL_LOOP(index, ${nthreads})
    {
        // out_channels, in_channels, kernel_h, kernel_w, batch_size, output_h, output_w
        const int oW = index % ${top_width};
        const int oH = index / ${top_width} % ${top_height};
        const int kW = index / ${top_width} / ${top_height} / ${batch_size} % ${kernel_w};
        const int kH = index / ${top_width} / ${top_height} / ${batch_size} / ${kernel_w} % ${kernel_h};

        const int h_in = -${pad_h} + oH * ${stride_h} + kH * ${dilation_h};
        const int w_in = -${pad_w} + oW * ${stride_w} + kW * ${dilation_w};
        if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width}))
        {
            const int bS = index / ${top_width} / ${top_height} % ${batch_size};
            const int iC = index / ${top_width} / ${top_height} / ${batch_size} / ${kernel_w} / ${kernel_h} % ${in_channels};
            const int oC = index / ${top_width} / ${top_height} / ${batch_size} / ${kernel_w} / ${kernel_h} / ${in_channels};
            const int top_offset = ((bS * ${out_channels} + oC) * ${top_height} + oH) * ${top_width} + oW;
            const int bot_offset = ((bS * ${in_channels} + iC) * ${bottom_height} + h_in) * ${bottom_width} + w_in;

            buffer_data[index] = top_diff[top_offset] * bottom_data[bot_offset];
        }
        else
        {
            buffer_data[index] = 0;
        }
    }
}