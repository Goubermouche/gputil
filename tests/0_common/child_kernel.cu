#include <cstdio>

extern "C" __device__ void child_func() {
	printf("xd\n");
}
