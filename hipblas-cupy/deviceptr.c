#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include <hipblas.h>


void daxpy(hipblasHandle_t handle, int n, double *a_, double *x_, double *y_)
{
    hipblasDaxpy(handle, n, a_, x_, 1, y_, 1);
}


static hipblasHandle_t _handle;

PyObject* daxpy_global(PyObject *self, PyObject *args)
{
    int n;
    double *a_;
    double *x_;
    double *y_;

    if (!PyArg_ParseTuple(args, "innn", &n, &a_, &x_, &y_))
        return NULL;

    daxpy(_handle, n, a_, x_, y_);

    Py_RETURN_NONE;
}

PyObject* create_global(PyObject *self, PyObject *args)
{
    hipblasCreate(&_handle);
    hipblasSetPointerMode(_handle, HIPBLAS_POINTER_MODE_DEVICE);

    Py_RETURN_NONE;
}

PyObject* destroy_global(PyObject *self, PyObject *args)
{
    hipDeviceSynchronize();
    hipblasDestroy(_handle);

    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"daxpy",          daxpy_global, METH_VARARGS, "Daxpy w/ global"},
    {"create_handle",  create_global, METH_VARARGS, "Create hipBLAS handle"},
    {"destroy_handle", destroy_global, METH_VARARGS, "Destroy hipBLAS handle"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "deviceptr",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC
PyInit__deviceptr(void)
{
    return PyModule_Create(&module);
}
