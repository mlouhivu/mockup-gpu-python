from math import sin, cos

from numpy import array

from _dot import dot
from _mem import Dallocate, deallocate, Dmemcpy_h2d, Dpeek

blocks = 32
threads = 256
n = 10000

# initialise data and calculate reference values on CPU
x = array([sin(i) * 2.3 for i in range(n)], float)
y = array([cos(i) * 1.1 for i in range(n)], float)
z_ref = sum(x * y)

# allocate + copy initial values
x_ = Dallocate(n)
y_ = Dallocate(n)
z_ = Dallocate(1)
Dmemcpy_h2d(x_, x, n)
Dmemcpy_h2d(y_, y, n)

# calculate dot product on GPU (and reduce partial sums)
dot(blocks, threads, n, x_, y_, z_)

# copy result back to host and print with reference
z = Dpeek(z_)
print(' reference: {0}'.format(z_ref))
print('    result: {0}'.format(z))
