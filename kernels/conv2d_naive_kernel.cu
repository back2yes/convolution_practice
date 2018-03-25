#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define CUDA_NUM_THREADS 1024
inline int GET_BLOCKS(int n)
{
    return (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

extern "C" __global__ void conv2d_naive_forward_kernel(
    const ${Dtype} *const bottom_data, const ${Dtype} *const weight_data, ${Dtype} *const top_data)
{
    CUDA_KERNEL_LOOP(index, ${nthreads})
    {
        // 4D coordinates
        const int oN = index / ${top_width} / ${top_height} / ${out_channels};
        const int oC = index / ${top_width} / ${top_height} % ${out_channels};
        const int oH = index / ${top_width} % ${top_height};
        const int oW = index % ${top_width};

        // weight & image offset
        const ${Dtype} *weight = weight_data + oC * ${in_channels} * ${kernel_h} * ${kernel_w};
        const int image_offset0 = oN * ${in_channels} * ${bottom_height} * ${bottom_width};

        ${Dtype} value = 0;

        // main loop
        for (int iicc = 0; iicc < ${in_channels}; iicc++)
        {
            const int image_offset1 = image_offset0 + iicc * ${bottom_height} * ${bottom_width};
#pragma unroll
            for (int kkhh = 0; kkhh < ${kernel_h}; kkhh++)
            {
#pragma unroll
                for (int kkww = 0; kkww < ${kernel_w}; kkww++)
                {
                    const int h_in = -${pad_h} + oH * ${stride_h} + kkhh * ${dilation_h};
                    const int w_in = -${pad_w} + oW * ${stride_w} + kkww * ${dilation_w};
                    if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width}))
                    {
                        const int offset = image_offset1 + h_in * ${bottom_width} + w_in;
                        value += (*weight) * bottom_data[offset];
                    }

                    weight++;
                }
            }
        }

        top_data[index] = value;
    }
}
