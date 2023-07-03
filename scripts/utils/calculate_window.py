#!/usr/bin/env python

import argparse

def calculate_strides(kernel_sizes, strides):
    if len(strides) == 1:
        return strides[0]
    return calculate_strides(kernel_sizes[:-1], strides[:-1])*strides[-1]

def calculate_kernel_size(kernel_sizes, strides):
    if len(kernel_sizes) == 1:
        return kernel_sizes[0]
    return (kernel_sizes[-1]-1)*calculate_strides(kernel_sizes[:-1], strides[:-1]) + calculate_kernel_size(kernel_sizes[:-1], strides[:-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'calculate window size when kernel sizes and strides are given')
    parser.add_argument('--kernel_sizes', type=int, nargs='+', required=True, help='kernel sizes')
    parser.add_argument('--strides', type=int, nargs='+', required=True, help='strides')
    args = parser.parse_args()
    kernel_size = calculate_kernel_size(args.kernel_sizes, args.strides)
    stride = calculate_strides(args.kernel_sizes, args.strides)
    print(f'kernel size: {kernel_size}, stride: {stride}')