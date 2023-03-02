#include <iostream>
#include <gputil.h>

#include "structure.h"

int main()
{
	CUDA_ASSERT(cuInit(0));

	const gputil::device dev = gputil::device::create([](const gputil::device& device) {
		return device.theoretical_memory_bandwidth;
	});

	const gputil::context ctx = gputil::context::create(dev);

	const gputil::program program = gputil::program::create("kernel.cu");
	gputil::kernel kernel = program.get_kernel("test_kernel");
	std::cout << "\nrunning kernel:\n";
	kernel.start({}, my_struct{ 99 }, 10);

	CUDA_ASSERT(cuCtxSynchronize());

	ctx.destroy();
	return 0;
}