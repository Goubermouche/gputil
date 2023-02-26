#include <cstdio>

extern "C" __device__ void child_func() {
	printf("Hello, world!\n");
}
