#include "child_kernel.cu"

extern "C" __global__ void test_kernel() {
	child_func();
}