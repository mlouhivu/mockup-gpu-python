LDFLAGS := -I$(CUDA_INSTALL_ROOT)/include -L$(CUDA_INSTALL_ROOT)/lib64
LIBS := -lcublas -lcudart -lm
CC := gcc

all: axpy axpy-async dot

axpy: axpy.c
	$(CC) $(LDFLAGS) $(LIBS) $< -o $@

axpy-async: axpy-async.c
	$(CC) $(LDFLAGS) $(LIBS) $< -o $@

dot: dot.c
	$(CC) $(LDFLAGS) $(LIBS) $< -o $@
