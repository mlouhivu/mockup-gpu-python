import cupy
import numpy

def managed():
    cupy.cuda.set_allocator(cupy.cuda.malloc_managed)

    x = cupy.arange(10)
    y = cupy.arange(10) * 2
    z = x + y
    print('pure CuPy (managed): z = x + y')
    print('  x={0}'.format(x))
    print('  y={0}'.format(y))
    print('  z={0}'.format(z))
    print('')

    x = cupy.arange(10)
    y = numpy.arange(10) * 2
    z = y + numpy.array(x)
    #z = x + y
    print('CuPy (managed) + numpy: z = x + y')
    print('  x={0}'.format(x))
    print('  y={0}'.format(y))
    print('  z={0}'.format(z))
    print('')

if __name__ == '__main__':
    managed()
