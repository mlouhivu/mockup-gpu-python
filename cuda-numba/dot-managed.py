from math import sin, cos

import numpy
import numba
import numba.cuda

from _dot import dot, cpu_dot

blocks = 32
threads = 256
n = 10000

# initialise data and calculate reference values on CPU
x = numpy.array([sin(i) * 2.3 for i in range(n)], float)
y = numpy.array([cos(i) * 1.1 for i in range(n)], float)
z_ref = sum(x * y)

# allocate + copy initial values
x_ = numba.cuda.managed_array(x.shape, float)
y_ = numba.cuda.managed_array(y.shape, float)
z_ = numba.cuda.managed_array((1,), float)
x_[:] = x[:]
y_[:] = y[:]

# calculate dot product on GPU (and reduce partial sums)
dot(blocks, threads, n, x_.device_ctypes_pointer.value,
    y_.device_ctypes_pointer.value, z_.device_ctypes_pointer.value)
numba.cuda.synchronize()

# copy result back to host and print with reference
z = numpy.empty((1,), float)
z[:] = z_[:]
print('GPU only')
print(' reference: {0}'.format(z_ref))
print('    result: {0}'.format(z[0]))
print('')

# repeat using a CPU-only algorithm on GPU arrays
z_ = numba.cuda.managed_array((1,), float)
dot(blocks, threads, n, x_.device_ctypes_pointer.value,
    y_.device_ctypes_pointer.value, z_.device_ctypes_pointer.value)
numba.cuda.synchronize()

# copy result back to host and print with reference
z = numpy.empty((1,), float)
z[:] = z_[:]
print('GPU arrays + CPU funtion')
print(' reference: {0}'.format(z_ref))
print('    result: {0}'.format(z[0]))
