#include <iostream>
#include <gputil.h>

int main()
{
	CUDA_ASSERT(cuInit(0));

	const gputil::device dev = gputil::device::create([](const gputil::device& device) {
		return device.theoretical_memory_bandwidth;
	});

	const gputil::context ctx = gputil::context::create(dev);

	const gputil::program program = gputil::program::create("kernel.cu");
	gputil::kernel kernel = program.get_kernel("test_kernel");
	kernel.start(10);

	ctx.destroy();
	return 0;
}