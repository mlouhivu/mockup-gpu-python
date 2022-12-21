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


/* Global variable version */

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


/* Capsuled pointer version */

PyObject* daxpy_capsule(PyObject *self, PyObject *args)
{
    PyObject *c;
    int n;
    double *a_;
    double *x_;
    double *y_;
    hipblasHandle_t *handle;

    if (!PyArg_ParseTuple(args, "Oinnn", &c, &n, &a_, &x_, &y_))
        return NULL;
    handle = PyCapsule_GetPointer(c, NULL);

    daxpy(*handle, n, a_, x_, y_);

    Py_RETURN_NONE;
}

PyObject* create_capsule(PyObject *self, PyObject *args)
{
    hipblasHandle_t *handle;

    handle = malloc(sizeof(hipblasHandle_t));
    hipblasCreate(handle);
    hipblasSetPointerMode(*handle, HIPBLAS_POINTER_MODE_DEVICE);

    return PyCapsule_New(handle, NULL, NULL);
}

PyObject* destroy_capsule(PyObject *self, PyObject *args)
{
    PyObject *c;
    hipblasHandle_t *handle;

    if (!PyArg_ParseTuple(args, "O", &c))
        return NULL;
    handle = PyCapsule_GetPointer(c, NULL);

    hipDeviceSynchronize();
    hipblasDestroy(*handle);
    free(handle);

    Py_RETURN_NONE;
}


static PyMethodDef methods[] = {
    {"daxpy_global",    daxpy_global, METH_VARARGS, "Daxpy w/ global"},
    {"create_global",   create_global, METH_VARARGS, "Create hipBLAS handle"},
    {"destroy_global",  destroy_global, METH_VARARGS, "Destroy hipBLAS handle"},
    {"daxpy_capsule",   daxpy_capsule, METH_VARARGS, "Daxpy w/ capsules"},
    {"create_capsule",  create_capsule, METH_VARARGS, "Create hipBLAS handle"},
    {"destroy_capsule", destroy_capsule, METH_VARARGS, "Destroy hipBLAS handle"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "async",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC
PyInit__async(void)
{
    return PyModule_Create(&module);
}
