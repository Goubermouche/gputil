#include "structure.h"
#include "iostream"

inline DEVICE void test() {
	printf("Mangled kernel function call\n");
}

GLOBAL void test_kernel(my_struct s, gputil::i32 x) {
	printf("kernel call: %i\n", s.get_value() * x);
	test();
}