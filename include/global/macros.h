// placeholder macros for purposes of easier-to-read kernels
#ifndef __CUDACC__
#define __host__ 
#define __device__ 
#define __global__ 
#endif

/**
 * \brief Declares a function that is executed on, and only callable from the host.
 */
#define HOST __host__  

/**
 * \brief Declares a function that is executed on, and only callable from the device.
 */
#define DEVICE __device__

/**
 *\brief Declares a function as being a kernel.
 */
#define GLOBAL __global__