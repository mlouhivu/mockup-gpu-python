#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <hip/hip_runtime.h>

__global__ void evolve_kernel(double *u, double *u_previous, double a, double dt, int nx, int ny,
                              double dx2, double dy2)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if (i > 0 && j > 0 && i < nx-1 && j < ny-1) {
    int ij = j + i*nx;
    int ip = j + (i+1)*nx;
    int im = j + (i-1)*nx;
    int jp = (j + 1) + i*nx;
    int jm = (j - 1) + i*nx;
    u[ij] = u_previous[ij] + a * dt * (
            (u_previous[ip] - 2*u_previous[ij] + u_previous[im]) / dx2 +
            (u_previous[jp] - 2*u_previous[ij] + u_previous[jm]) / dy2 );
  }
}

extern "C"
PyObject* evolve_buffer_gpu(PyObject *self, PyObject *args)
{
  PyObject* field;
  PyObject* field_previous;
  double a, dt, dx2, dy2;

  if (!PyArg_ParseTuple(args, "OOdddd", &field, &field_previous, &a, &dt, &dx2, &dy2))
    return NULL;

  Py_buffer view;
  Py_buffer view_previous;
  if (PyObject_GetBuffer(field, &view, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1)
    return NULL;
  if (PyObject_GetBuffer(field_previous, &view_previous, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1)
    return NULL;

  int nx = view.shape[0];
  int ny = view.shape[1];

  double *u = (double *) view.buf;
  double *u_previous = (double *) view_previous.buf;

  const int blocksize = 16;
  dim3 dimBlock(blocksize, blocksize);
  dim3 dimGrid((ny + blocksize - 1) / blocksize, 
                 (nx + blocksize - 1) / blocksize);

  hipLaunchKernelGGL(evolve_kernel, dimGrid, dimBlock, 0, 0, 
                     u, u_previous, nx, ny, a, dt, dx2, dy2);

  hipMemcpy(u_previous, u, nx*ny*sizeof(double), hipMemcpyDeviceToDevice);

  PyBuffer_Release(&view);
  PyBuffer_Release(&view_previous);

  Py_RETURN_NONE;
}


extern "C"
PyObject* evolve_pointer(PyObject *self, PyObject *args)
{
  double *u;
  double *u_previous;
  int nx, ny;
  double a, dt, dx2, dy2;

  if (!PyArg_ParseTuple(args, "nniidddd", &u, &u_previous, &nx, &ny, &a, &dt, &dx2, &dy2))
    return NULL;

  for (int i=1; i < nx-1; i++)
    for (int j=1; j < ny-1; j++) {
      // Linearisation for 2D array
      int ij = j + i*nx;
      int ip = j + (i+1)*nx;
      int im = j + (i-1)*nx;
      int jp = (j + 1) + i*nx;
      int jm = (j - 1) + i*nx;
      u[ij] = u_previous[ij] + a * dt * (
              (u_previous[ip] - 2*u_previous[ij] + u_previous[im]) / dx2 +
              (u_previous[jp] - 2*u_previous[ij] + u_previous[jm]) / dy2 );
      }

  memcpy(u_previous, u, nx*ny*sizeof(double));

  Py_RETURN_NONE;
}


extern "C"
PyObject* evolve_array(PyObject *self, PyObject *args)
{
  PyArrayObject* field;
  PyArrayObject* field_previous;
  double a, dt, dx2, dy2;

  if (!PyArg_ParseTuple(args, "OOdddd", &field, &field_previous, &a, &dt, &dx2, &dy2))
    return NULL;

  int nx = PyArray_DIMS(field)[0];
  int ny = PyArray_DIMS(field)[1];

  double *u = (double *) PyArray_DATA(field);
  double *u_previous = (double *) PyArray_DATA(field_previous);

  for (int i=1; i < nx-1; i++)
    for (int j=1; j < ny-1; j++) {
      // Linearisation for 2D array
      int ij = j + i*nx;
      int ip = j + (i+1)*nx;
      int im = j + (i-1)*nx;
      int jp = (j + 1) + i*nx;
      int jm = (j - 1) + i*nx;
      u[ij] = u_previous[ij] + a * dt * (
              (u_previous[ip] - 2*u_previous[ij] + u_previous[im]) / dx2 +
              (u_previous[jp] - 2*u_previous[ij] + u_previous[jm]) / dy2 );
      }

  memcpy(u_previous, u, nx*ny*sizeof(double));

  Py_RETURN_NONE;
}

extern "C"
PyObject* evolve_buffer(PyObject *self, PyObject *args)
{
  PyObject* field;
  PyObject* field_previous;
  double a, dt, dx2, dy2;

  if (!PyArg_ParseTuple(args, "OOdddd", &field, &field_previous, &a, &dt, &dx2, &dy2))
    return NULL;

 Py_buffer view;
 Py_buffer view_previous;
 if (PyObject_GetBuffer(field, &view, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1)
   return NULL;
 if (PyObject_GetBuffer(field_previous, &view_previous, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1)
   return NULL;

 int nx = view.shape[0];
 int ny = view.shape[1];

 double *u = (double *) view.buf;
 double *u_previous = (double *) view_previous.buf;

  for (int i=1; i < nx-1; i++)
    for (int j=1; j < ny-1; j++) {
      // Linearisation for 2D array
      int ij = j + i*nx;
      int ip = j + (i+1)*nx;
      int im = j + (i-1)*nx;
      int jp = (j + 1) + i*nx;
      int jm = (j - 1) + i*nx;
      u[ij] = u_previous[ij] + a * dt * (
              (u_previous[ip] - 2*u_previous[ij] + u_previous[im]) / dx2 +
              (u_previous[jp] - 2*u_previous[ij] + u_previous[jm]) / dy2 );
      }

  memcpy(u_previous, u, nx*ny*sizeof(double));

  PyBuffer_Release(&view);
  PyBuffer_Release(&view_previous);

  Py_RETURN_NONE;
}



static PyMethodDef functions[] = {
 {"evolve_pointer", evolve_pointer, METH_VARARGS, 0},
 {"evolve_array", evolve_array, METH_VARARGS, 0},
 {"evolve_buffer", evolve_buffer, METH_VARARGS, 0},
 {"evolve_buffer_gpu", evolve_buffer_gpu, METH_VARARGS, 0},
 {0, 0, 0, 0}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "heat",
    NULL,
    -1,
    functions
};

PyMODINIT_FUNC PyInit__heat(void)
{
    import_array(); // Required for NumPy API
    return PyModule_Create(&module);
}
