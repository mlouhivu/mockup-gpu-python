from math import sin, cos
from array import array

from _dot import dot
from _mem import Dallocate, deallocate, Dmemcpy_h2d, Dpeek

blocks = 32
threads = 256
n = 10000

# initialise data and calculate reference values on CPU
x = array('d', [sin(i) * 2.3 for i in range(n)])
y = array('d', [cos(i) * 1.1 for i in range(n)])
z_ref = sum([i * j for (i,j) in zip(x,y)])

# memory addresses of buffers
xp = x.buffer_info()[0]
yp = y.buffer_info()[0]

# allocate + copy initial values
x_ = Dallocate(n)
y_ = Dallocate(n)
z_ = Dallocate(1)
Dmemcpy_h2d(x_, xp, n)
Dmemcpy_h2d(y_, yp, n)

# calculate dot product on GPU (and reduce partial sums)
dot(blocks, threads, n, x_, y_, z_)

# copy result back to host and print with reference
z = Dpeek(z_)
print(' reference: {0}'.format(z_ref))
print('    result: {0}'.format(z))
