from math import sin, cos

import numpy
import numba
import numba.cuda

from _dot import dot

blocks = 32
threads = 256
n = 10000

# initialise data and calculate reference values on CPU
x = numpy.array([sin(i) * 2.3 for i in range(n)], float)
y = numpy.array([cos(i) * 1.1 for i in range(n)], float)
z_ref = sum(x * y)

# allocate + copy initial values
x_ = numba.cuda.to_device(x)
y_ = numba.cuda.to_device(y)
z_ = numba.cuda.device_array(1, float)

# calculate dot product on GPU (and reduce partial sums)
dot(blocks, threads, n, x_.device_ctypes_pointer.value,
    y_.device_ctypes_pointer.value, z_.device_ctypes_pointer.value)

# copy result back to host and print with reference
z = z_.copy_to_host()[0]
print(' reference: {0}'.format(z_ref))
print('    result: {0}'.format(z))
