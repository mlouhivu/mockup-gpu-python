CFLAGS := -I$(ROCM_PATH)/hip/include -I$(ROCM_PATH)/hipblas/include
CFLAGS := $(CFLAGS) -D__HIP_PLATFORM_AMD__
LDFLAGS := -L$(ROCM_PATH)/hipblas/lib
LIBS := -lhipblas
CC := gcc

exe := $(patsubst %.c, %, $(wildcard *.c))

all: $(exe)

%: %.c
	$(CC) $(CFLAGS) $(LDFLAGS) $(LIBS) $< -o $@

clean:
	rm $(exe)
