# import unittest
from functools import partial
import torch
from torch.autograd import gradcheck, Variable
import pyinn as P
# from pyinn.modules import Conv2dDepthwise
import torch.nn.functional as F


def check_conv2d_naive_forward():
    in_channels = 5
    out_channels = 7
    x = Variable(torch.randn(1, in_channels, 4, 4).double().cuda(), requires_grad=True)
    w = Variable(torch.randn(out_channels, in_channels, 3, 3).double().cuda(), requires_grad=True)
    y_naive = P.conv2d_naive(x, w, padding=1)
    y_gt = F.conv2d(x, w, padding=1)
    print((y_naive - y_gt).data.abs().max())
    # print(y_naive - y_gt)
    # print(y_gt)


def check_conv2d_naive_backward_grad_input():
    in_channels = 5
    out_channels = 7
    x = Variable(torch.randn(1, in_channels, 4, 4).double().cuda(), requires_grad=True)
    w = Variable(torch.randn(out_channels, in_channels, 3, 3).double().cuda(), requires_grad=False)
    y_naive = P.conv2d_naive(x, w, padding=1)
    y_gt = F.conv2d(x, w, padding=1)
    # print((y_naive - y_gt).data.abs().max())
    # print(y_naive - y_gt)
    # print(y_gt)
    go = torch.randn(y_naive.size()).double().cuda()
    x.requires_grad = True
    # w.requires_grad = True
    y_naive.backward(go)
    gx_naive = x.grad.data.clone()
    # gw_naive = w.grad.data.clone()

    x.grad.data.zero_()
    y_gt.backward(go)
    gx_ref = x.grad.data.clone()
    # gw_ref = w.grad.data.clone()
    # print(gx_naive)
    # print(gx_ref)
    # print()
    print((gx_naive - gx_ref).abs().max() / gx_ref.abs().max())


def check_conv2d_naive_backward_grad_weight():
    in_channels = 3
    out_channels = 5
    x = Variable(torch.randn(3, in_channels, 7, 9).double().cuda(), requires_grad=False)
    w = Variable(torch.randn(out_channels, in_channels, 5, 5).double().cuda(), requires_grad=True)
    y_naive = P.conv2d_naive(x, w, padding=2)
    y_gt = F.conv2d(x, w, padding=2)
    # print((y_naive - y_gt).data.abs().max())
    # print(y_naive - y_gt)
    # print(y_gt)
    go = torch.randn(y_naive.size()).double().cuda()
    x.requires_grad = False
    w.requires_grad = True
    y_naive.backward(go)
    # gx_naive = x.grad.data.clone()
    gw_naive = w.grad.data.clone()

    # x.grad.data.zero_()
    w.grad.data.zero_()
    y_gt.backward(go)
    # gx_ref = x.grad.data.clone()
    gw_ref = w.grad.data.clone()
    # print(gx_naive)
    # print(gx_ref)
    # print(gw_naive)
    # print(gw_ref)
    print(gw_ref / gw_naive)
    # print()
    # print((gx_naive - gx_ref).abs().max() / gx_ref.abs().max())
    # print((gw_naive - gw_ref).abs().max() / gw_ref.abs().max())


def check_conv2d_naive():
    n = 6
    x = Variable(torch.randn(1, n, 5, 5).double().cuda(), requires_grad=True)
    w = Variable(torch.randn(n, 1, 3, 3).double().cuda(), requires_grad=True)
    y_fast = P.conv2d_depthwise(x, w, padding=1)
    y_ref = F.conv2d(x, w, padding=1, groups=n)
    go = torch.randn(y_fast.size()).double().cuda()

    # self.assertLess(, 1e-9)
    print((y_fast - y_ref).data.abs().max())

    x.requires_grad = True
    w.requires_grad = True
    y_fast.backward(go)
    gx_fast = x.grad.data.clone()
    gw_fast = w.grad.data.clone()

    x.grad.data.zero_()
    w.grad.data.zero_()
    y_ref.backward(go)
    gx_ref = x.grad.data.clone()
    gw_ref = w.grad.data.clone()

    gradcheck(partial(P.conv2d_depthwise, padding=1), (x, w,))


if __name__ == '__main__':
    # check_conv2d_naive_forward()
    # check_conv2d_naive_backward_grad_input()
    check_conv2d_naive_backward_grad_weight()
