CXX = g++
NVCC = nvcc

CXXFLAGS = -std=c++17 -Wall -I./xsimd/include
NVCCFLAGS = -Xcompiler -Wall -std=c++17 -I./xsimd/include --gpu-architecture=sm_86

DIR = src

CPU_SRC = HW2.cpp ssw_cpp.cpp ssw.c
CUDA_SRC = Hw3CUDA.cpp HW2_cuda.cu ssw_cpp.cpp ssw.c

.PHONY: all cpu cuda clean

all:  cuda cpu

cpu:
	cd $(DIR) && $(CXX) $(CXXFLAGS) $(CPU_SRC) -o mysw_cpu && ./mysw_cpu

cuda:
	cd $(DIR) && \
	$(NVCC) $(NVCCFLAGS) -c HW2_cuda.cu -o HW2_cuda.o && \
	$(NVCC) $(NVCCFLAGS) Hw3CUDA.cpp HW2_cuda.o ssw_cpp.cpp ssw.c -o mysw_cuda && \
	./mysw_cuda


clean:
	rm -f $(DIR)/*.o $(DIR)/mysw_cpu $(DIR)/mysw_cuda
