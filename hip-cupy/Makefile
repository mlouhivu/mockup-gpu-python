CFLAGS := -shared -I$(PYTHONHOME)/include/python3.9 -fPIC
CFLAGS := $(CFLAGS) -I$(PYTHONHOME)/lib/python3.9/site-packages/numpy/core/include 
CC := hipcc

exe := $(patsubst %.cpp, _%.so, $(wildcard *.cpp))

all: $(exe)

_%.so: %.cpp
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm $(exe)
