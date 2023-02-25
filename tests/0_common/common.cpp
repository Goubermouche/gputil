#include <iostream>

#include <gputil.h>

/*
#include <cuda.h>
#include <nvrtc.h>

#define NUM_THREADS 10
#define NUM_BLOCKS 32

const char* saxpy = "                                         \n\
extern \"C\" __global__                                       \n\
void saxpy(float a, float *x, float *y, float *out, size_t n) \n\
{                                                             \n\
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;         \n\
  if (tid < n) {                                              \n\
    out[tid] = a * x[tid] + y[tid];                           \n\
  }                                                           \n\
}                                                           \n";*/  

int main()
{
	//nvrtcProgram program;

	//NVRTC_ASSERT(nvrtcCreateProgram(
	//	&program,   // program
	//	saxpy,      // buffer
	//	"saxpy.cu", // name
	//	0,          // numHeaders
	//	NULL,       // headers
	//	NULL        // includeNames
	//));

	//// Compile the program with fmad disabled.
	//// Note: Can specify GPU target architecture explicitly with '-arch' flag.
	//const char* opts[] = { "--fmad=false" };
	//const nvrtcResult compileResult = nvrtcCompileProgram(program, 1, opts);

	//// Obtain compilation log from the program.
	//size_t logSize;
	//NVRTC_ASSERT(nvrtcGetProgramLogSize(program, &logSize));
	//char* log = new char[logSize];
	//NVRTC_ASSERT(nvrtcGetProgramLog(program, log));
	//std::cout << log << '\n';
	//delete[] log;

	//if (compileResult != NVRTC_SUCCESS) {
	//	exit(1);
	//}

	//// Obtain PTX from the program.
	//size_t ptxSize;
	//NVRTC_ASSERT(nvrtcGetPTXSize(program, &ptxSize));
	//char* ptx = new char[ptxSize];
	//NVRTC_ASSERT(nvrtcGetPTX(program, ptx));
	//NVRTC_ASSERT(nvrtcDestroyProgram(&program)); // Destroy the program.

	//// Load the generated PTX and get a handle to the SAXPY kernel.
	//CUdevice cuDevice;
	//CUcontext context;
	//CUmodule module;
	//CUfunction kernel;

	//CUDA_ASSERT(cuInit(0));
	//CUDA_ASSERT(cuDeviceGet(&cuDevice, 0));
	//CUDA_ASSERT(cuCtxCreate(&context, 0, cuDevice));
	//CUDA_ASSERT(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
	//CUDA_ASSERT(cuModuleGetFunction(&kernel, module, "saxpy"));

	//// Generate input for execution, and create output buffers.
	//size_t n = NUM_THREADS * NUM_BLOCKS;
	//const size_t bufferSize = n * sizeof(float);
	//float a = 5.1f;
	//float* hX = new float[n], * hY = new float[n], * hOut = new float[n];

	//for (size_t i = 0; i < n; ++i) {
	//	hX[i] = static_cast<float>(i);
	//	hY[i] = static_cast<float>(i * 2);
	//}

	//CUdeviceptr dX, dY, dOut;
	//CUDA_ASSERT(cuMemAlloc(&dX, bufferSize));
	//CUDA_ASSERT(cuMemAlloc(&dY, bufferSize));
	//CUDA_ASSERT(cuMemAlloc(&dOut, bufferSize));
	//CUDA_ASSERT(cuMemcpyHtoD(dX, hX, bufferSize));
	//CUDA_ASSERT(cuMemcpyHtoD(dY, hY, bufferSize));

	//// Execute SAXPY
	//void* args[] = { &a, &dX, &dY, &dOut, &n };
	//CUDA_ASSERT(cuLaunchKernel(
	//	kernel,            // kernel
	//	NUM_BLOCKS, 1, 1,  // grid dim
	//	NUM_THREADS, 1, 1, // block dim
	//	0, NULL,           // shared mem and stream
	//	args, 0            // arguments
	//));

	//CUDA_ASSERT(cuCtxSynchronize());

	//// Retrieve and print output
	//CUDA_ASSERT(cuMemcpyDtoH(hOut, dOut, bufferSize));

	//for (size_t i = 0; i < n; ++i) {
	//	std::cout << a << " * " << hX[i] << " + " << hY[i] << " = " << hOut[i] << '\n';
	//}

	//// Release resources
	//CUDA_ASSERT(cuMemFree(dX));
	//CUDA_ASSERT(cuMemFree(dY));
	//CUDA_ASSERT(cuMemFree(dOut));
	//CUDA_ASSERT(cuModuleUnload(module));
	//CUDA_ASSERT(cuCtxDestroy(context));

	//delete[] hX;
	//delete[] hY;
	//delete[] hOut;
	//delete[] ptx;

	//return 0;

	CUDA_ASSERT(cuInit(0));

	const gputil::device dev = gputil::device::create([](const gputil::device& device) {
		return device.theoretical_memory_bandwidth;
	});

	std::cout << "Theoretical memory bandwidth: " << gputil::format_bytes(dev.theoretical_memory_bandwidth) << '\n';
	std::cout << "Core count: " << dev.core_count << '\n';
	std::cout << "Name: " << dev.name << '\n';
}