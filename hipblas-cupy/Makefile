CFLAGS := -shared -I$(PYTHONHOME)/include/python3.9 -fPIC
CFLAGS := $(CFLAGS) -I$(PYTHONHOME)/lib/python3.9/site-packages/numpy/core/include 
CFLAGS := $(CFLAGS) -I$(ROCM_PATH)/hip/include -I$(ROCM_PATH)/hipblas/include
CFLAGS := $(CFLAGS) -D__HIP_PLATFORM_AMD__
LDFLAGS := -L$(ROCM_PATH)/hipblas/lib
LIBS := -lhipblas
CC := gcc

exe := $(patsubst %.c, _%.so, $(wildcard *.c))

all: $(exe)

_%.so: %.c
	$(CC) $(CFLAGS) $(LDFLAGS) $(LIBS) $< -o $@

clean:
	rm $(exe)
