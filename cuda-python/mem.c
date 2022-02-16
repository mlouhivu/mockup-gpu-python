#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime.h>

PyObject* Dallocate(PyObject *self, PyObject *args)
{
    int n;
    double *x_;

    if (!PyArg_ParseTuple(args, "i", &n))
        return NULL;

    cudaMalloc((void **) &x_, sizeof(double) * n);

    return PyCapsule_New(x_, NULL, NULL);
}

PyObject* deallocate(PyObject *self, PyObject *args)
{
    int n;
    PyObject *c_;
    void *x_;

    if (!PyArg_ParseTuple(args, "O", &c_))
        return NULL;
    x_ = PyCapsule_GetPointer(c_, NULL);

    cudaFree(x_);

    Py_RETURN_NONE;
}

PyObject* Dmemcpy_h2d(PyObject *self, PyObject *args)
{
    int n;
    PyObject *c_;
    double *x;
    double *x_;

    if (!PyArg_ParseTuple(args, "Oni", &c_, &x, &n))
        return NULL;
    x_ = (double *) PyCapsule_GetPointer(c_, NULL);

    cudaMemcpy(x_, x, sizeof(double) * n, cudaMemcpyHostToDevice);

    Py_RETURN_NONE;
}

PyObject* Dmemcpy_d2h(PyObject *self, PyObject *args)
{
    int n;
    PyObject *c_;
    double *x;
    double *x_;

    if (!PyArg_ParseTuple(args, "nOi", &x, &c_, &n))
        return NULL;
    x_ = (double *) PyCapsule_GetPointer(c_, NULL);

    cudaMemcpy(x, x_, sizeof(double) * n, cudaMemcpyDeviceToHost);

    Py_RETURN_NONE;
}

PyObject* Dpeek(PyObject *self, PyObject *args)
{
    PyObject *c_;
    double x;
    double *x_;

    if (!PyArg_ParseTuple(args, "O", &c_))
        return NULL;
    x_ = (double *) PyCapsule_GetPointer(c_, NULL);

    cudaMemcpy(&x, x_, sizeof(double), cudaMemcpyDeviceToHost);

    return Py_BuildValue("d", x);
}

static PyMethodDef methods[] = {
    {"Dallocate", Dallocate, METH_VARARGS,
        "cudaMalloc for doubles"},
    {"deallocate", deallocate, METH_VARARGS,
        "cudaFree"},
    {"Dmemcpy_h2d", Dmemcpy_h2d, METH_VARARGS,
        "cudaMemcpy host->device for doubles"},
    {"Dmemcpy_d2h", Dmemcpy_d2h, METH_VARARGS,
        "cudaMemcpy device->host for doubles"},
    {"Dpeek", Dpeek, METH_VARARGS,
        "copy a single double from device memory"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "mem",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC
PyInit__mem(void)
{
    return PyModule_Create(&module);
}
