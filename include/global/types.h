#pragma once

#include <cstdint>

namespace gputil {
	namespace types {
		// Unsigned 
		using u8  = uint8_t;
		using u16 = uint16_t;
		using u32 = uint32_t;
		using u64 = uint64_t;

		// Signed
		using i8  = int8_t;
		using i16 = int16_t;
		using i32 = int32_t;
		using i64 = int64_t;

		// Floating point
		using f32  = float;
		using f64  = double;
		using f128 = long double;
		
		/**
		* \brief Supported memory types.
		*/
		enum class memory {
			local, // Local memory, encapsulates the default scope allocations
			device // Allocates device-side memory on the host
			// Unified?
		};
	}

	using namespace gputil::types;
}


#if !defined(__CUDACC__) && !defined(__host__)
#define __host__ 
#endif

#if !defined(__CUDACC__) && !defined(__device__)
#define __device__ 
#endif

#if !defined(__CUDACC__) && !defined(__global__)
#define __global__ 
#endif

#define HOST __host__
#define DEVICE __device__
#define GLOBAL __global__