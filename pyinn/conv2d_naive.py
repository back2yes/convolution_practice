from torch.autograd import Function
import torch
from torch.nn.modules.utils import _pair
from pyinn.utils import Dtype, Stream, load_kernel
import torch.nn.functional as F

CUDA_NUM_THREADS = 1024

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


_conv2d_naive_kernel = kernel_loop + '''
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
'''

_conv2d_naive_kernel_backward_grad_input = kernel_loop + '''
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
'''

_conv2d_naive_kernel_backward_grad_weight = kernel_loop + '''
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
'''


class Conv2dNaive(Function):

    def __init__(self, stride, padding, dilation):
        super(Conv2dNaive, self).__init__()
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

    def forward(self, input, weight):
        assert input.dim() == 4 and input.is_cuda and weight.is_cuda
        batch_size, in_channels, bottom_height, bottom_width = input.size()
        out_channels, _, kernel_h, kernel_w = weight.size()
        print(in_channels, out_channels, batch_size)
        output_h = int(
            (bottom_height + 2 * self.padding[0] - (self.dilation[0] * (kernel_h - 1) + 1)) / self.stride[0] + 1)
        output_w = int(
            (bottom_width + 2 * self.padding[1] - (self.dilation[1] * (kernel_w - 1) + 1)) / self.stride[1] + 1)

        output = input.new(batch_size, out_channels, output_h, output_w)
        n = output.numel()

        with torch.cuda.device_of(input):
            f = load_kernel('conv2d_naive_forward_kernel', _conv2d_naive_kernel, Dtype=Dtype(input), nthreads=n,
                            batch_size=batch_size,
                            in_channels=in_channels,
                            out_channels=out_channels,
                            bottom_height=bottom_height, bottom_width=bottom_width,
                            top_height=output_h, top_width=output_w,
                            kernel_h=kernel_h, kernel_w=kernel_w,
                            stride_h=self.stride[0], stride_w=self.stride[1],
                            dilation_h=self.dilation[0], dilation_w=self.dilation[1],
                            pad_h=self.padding[0], pad_w=self.padding[1])
            f(block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(n), 1, 1),
              args=[input.data_ptr(), weight.data_ptr(), output.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        self.save_for_backward(input, weight)
        return output

    def backward(self, grad_output):
        assert grad_output.is_cuda and grad_output.is_contiguous()
        input, weight = self.saved_tensors

        batch_size, in_channels, bottom_height, bottom_width = input.size()
        out_channels, _, kernel_h, kernel_w = weight.size()
        top_height, top_width = grad_output.size()[2:]

        grad_input, grad_weight = None, None

        opt = dict(Dtype=Dtype(grad_output),
                   batch_size=batch_size,
                   in_channels=in_channels,
                   out_channels=out_channels,
                   bottom_height=bottom_height, bottom_width=bottom_width,
                   top_height=top_height, top_width=top_width,
                   kernel_h=kernel_h, kernel_w=kernel_w,
                   stride_h=self.stride[0], stride_w=self.stride[1],
                   dilation_h=self.dilation[0], dilation_w=self.dilation[1],
                   pad_h=self.padding[0], pad_w=self.padding[1])

        with torch.cuda.device_of(input):
            if self.needs_input_grad[0]:
                grad_input = input.new(input.size())
                n = grad_input.numel()
                opt['nthreads'] = n
                weight_transposed = weight.permute(1, 0, 2, 3).contiguous()
                f = load_kernel('conv2d_naive_backward_grad_input_kernel',
                                _conv2d_naive_kernel_backward_grad_input, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), weight_transposed.data_ptr(), grad_input.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
            else:
                grad_input = None

            if self.needs_input_grad[1]:
                weight_buffer = weight.new(out_channels, in_channels, kernel_h, kernel_w, batch_size, top_height,
                                           top_width)

                n = weight_buffer.numel()
                opt['nthreads'] = n

                f = load_kernel('conv2d_naive_backward_grad_weight_kernel',
                                _conv2d_naive_kernel_backward_grad_weight, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), input.data_ptr(), weight_buffer.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
                grad_weight = weight_buffer.view(weight.size() + (-1,)).sum(-1)
            else:
                grad_weight = None

        return grad_input, grad_weight


def conv2d_naive(input, weight, bias=None, stride=1, padding=0, dilation=1):
    """Depthwise 2D convolution.

    Implements depthwise convolution as in https://arxiv.org/pdf/1704.04861v1.pdf
    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

    CUDA kernels from https://github.com/BVLC/caffe/pull/5665
    CPU side is done by F.conv2d

    Equivalent to:
        `F.conv2d(input, weight, groups=input.size(1))`
    """
    # assert input.size(1) == weight.size(0)
    if input.is_cuda:
        out = Conv2dNaive(stride, padding, dilation)(input, weight)
        if bias is not None:
            out += bias.view(1, -1, 1, 1)
    else:
        # groups = input.size(1)
        out = F.conv2d(input, weight, bias, stride, padding, dilation)
    return out
