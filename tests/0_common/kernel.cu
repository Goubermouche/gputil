#include "structure.h"

__device__ void test() {
	printf("Mangled kernel function call\n");
}

__global__ void test_kernel(my_struct s, int x) {
	printf("kernel call: %i\n", s.get_value() * x);
	test();
}