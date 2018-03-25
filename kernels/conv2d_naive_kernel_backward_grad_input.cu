extern "C" __global__ void conv2d_naive_backward_grad_input_kernel(
    const ${Dtype} *const top_diff, const ${Dtype} *const weight_data, ${Dtype} *const bottom_diff)
{
    CUDA_KERNEL_LOOP(index, ${nthreads})
    {
        // 4D indices
        const int iN = index / ${bottom_width} / ${bottom_height} / ${in_channels};
        const int iC = index / ${bottom_width} / ${bottom_height} % ${in_channels};
        const int iH = index / ${bottom_width} % ${bottom_height};
        const int iW = index % ${bottom_width};

        // offset
        // kernel weight's 1st and 2nd axes are swapped: [in_channels, out_channels, kernel_h, kernel_w]
        const ${Dtype} *weight = weight_data + iC * ${out_channels} * ${kernel_h} * ${kernel_w};
        const int top_offset0 = iN * ${out_channels} * ${top_height} * ${top_width};

        ${Dtype} value = 0;
        for (int oocc = 0; oocc < ${out_channels}; oocc++)
        {
            const int top_offset1 = oocc * ${top_height} * ${top_width};
#pragma unroll
            for (int kkhh = 0; kkhh < ${kernel_h}; ++kkhh)
            {
#pragma unroll
                for (int kkww = 0; kkww < ${kernel_w}; ++kkww)
                {
                    const int h_out_s = iH + ${pad_h} - kkhh * ${dilation_h};
                    const int w_out_s = iW + ${pad_w} - kkww * ${dilation_w};
                    if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0))
                    {
                        const int h_out = h_out_s / ${stride_h};
                        const int w_out = w_out_s / ${stride_w};
                        if ((h_out >= 0) && (h_out < ${top_height}) && (w_out >= 0) && (w_out < ${top_width}))
                        {
                            const int offset = top_offset1 + h_out * ${top_width} + w_out;
                            value += (*weight) * top_diff[offset];
                        }
                    }
                    ++weight;
                }
            }
        }
        bottom_diff[index] = value;
    }
}