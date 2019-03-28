from pycuda import elementwise
from pycuda import gpuarray
import numpy

from basic import numpify

def axpbyz():
    a = numpify(None)
    b = numpify(None)
    c = numpy.zeros_like(a)
    print('Before axpbyz')
    print('a = {0}'.format(a))
    print('b = {0}'.format(b))
    print('c = {0}'.format(c))

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_gpu = gpuarray.to_gpu(c)
    c_gpu = a_gpu._axpbyz(5.0, b_gpu, 3.0, c_gpu)
    print('After axpbyz')
    print('a_gpu = {0}'.format(a_gpu))
    print('b_gpu = {0}'.format(b_gpu))
    print('c_gpu = {0}'.format(c_gpu))

def axpbyz_kernel(dtype=float):
    a = numpify(None, dtype)
    b = numpify(None, dtype)
    c = numpy.zeros_like(a)
    print('Before axpbyz_kernel')
    print('a = {0}'.format(a))
    print('b = {0}'.format(b))
    print('c = {0}'.format(c))

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_gpu = gpuarray.to_gpu(c)

    f = elementwise.get_axpbyz_kernel(dtype, dtype, dtype)
    mem_size = a.size
    grid, block = gpuarray.splay(mem_size)
    print('Calling {0} kernel with:  grid={1}  block={2}  mem_size={3}'.format(
          'axpbyz', grid, block, mem_size))
    f.prepared_async_call(grid, block, None, 5.0, a_gpu.gpudata, 3.0,
                          b_gpu.gpudata, c_gpu.gpudata, mem_size)
    print('After axpbyz_kernel')
    print('a_gpu = {0}'.format(a_gpu))
    print('b_gpu = {0}'.format(b_gpu))
    print('c_gpu = {0}'.format(c_gpu))
