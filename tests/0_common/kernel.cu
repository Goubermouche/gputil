#include <cstdio>

__device__ void test() {
	printf("Mangled kernel function call\n");
}

__global__ void test_kernel() {
	test();
}