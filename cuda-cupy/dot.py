from math import sin, cos

from numpy import array
import cupy

from _dot import dot

blocks = 32
threads = 256
n = 10000

# initialise data and calculate reference values on CPU
x = array([sin(i) * 2.3 for i in range(n)], float)
y = array([cos(i) * 1.1 for i in range(n)], float)
z_ref = sum(x * y)

# allocate + copy initial values
x_ = cupy.asarray(x)
y_ = cupy.asarray(y)
z_ = cupy.empty(1, float)

# calculate dot product on GPU (and reduce partial sums)
dot(blocks, threads, n, x_.data.ptr, y_.data.ptr, z_.data.ptr)

# copy result back to host and print with reference
z = z_.get()[0]
print(' reference: {0}'.format(z_ref))
print('    result: {0}'.format(z))
