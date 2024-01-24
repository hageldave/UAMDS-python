from __future__ import division
from numba import cuda
import numpy
import math


def test():
    print(cuda.gpus)  # should print someting like: <Managed Device 0>

    # CUDA kernel
    @cuda.jit
    def cuda_multiply_by_2(array_in, array_out):
        pos = cuda.grid(1)
        if pos < array_in.size:
            array_out[pos] = array_in[pos] * 2  # do the computation



    # Host code
    data = numpy.ones(1024)
    threadsperblock = 64
    blockspergrid = math.ceil(data.shape[0] / threadsperblock)

    data_in = cuda.to_device(data)
    data_out = cuda.to_device(data)

    # compute x * (2^exp)
    exp = 10
    for i in range(exp):
        # swap arrays
        temp = data_in
        data_in = data_out
        data_out = temp
        # compute on gpu
        cuda_multiply_by_2[blockspergrid, threadsperblock](data_in, data_out)

    data = data_out.copy_to_host()
    print(data)


if __name__ == '__main__':
    test()
