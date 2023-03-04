#include "iostream"
#include "global/vector.h"

using namespace gputil::types;

inline DEVICE void test() {
	printf("Mangled kernel function call\n");
}

GLOBAL void test_kernel(gputil::vector<i32> vec, i32 x) {
	printf("%i\n", vec.value * x);
	test();
}