#pragma once

#include "types.h"

namespace gputil {
	using namespace gputil::types;
}

#define MAX_PATH 260

/**
 * \brief CUDA assertion macro, asserts whenever \a cond does not evaluate to CUDA_SUCCESS and prints the relevant error message retrieved from CUDA.
 * \param cond Condition to evaluate
 */
#define CUDA_ASSERT(cond)                                                                        \
	do {                                                                                         \
		if((cond) == cudaError_enum::CUDA_SUCCESS) {                                             \
		} else {                                                                                 \
			const char* msg;                                                                     \
			cuGetErrorName(cond, &msg);                                                           \
			printf("CUDA ASSERTION FAILED (%s:%i): %s (%s)\n", __FILE__, __LINE__, #cond, msg); \
			__debugbreak();                                                                      \
		}                                                                                        \
	} while(false)

 /**
  * \brief NVRTC assertion macro, asserts whenever \a cond does not evaluate to NVRTC_SUCCESS and prints the relevant error message retrieved from CUDA.
  * \param cond Condition to evaluate
  */
#define NVRTC_ASSERT(cond)                                                                        \
    do {                                                                                          \
          if((cond) == NVRTC_SUCCESS) {                                                           \
          } else {                                                                                \
	  		printf("NVRTC ASSERTION FAILED (%s:%i): %s (%i)\n", __FILE__, __LINE__, #cond, cond); \
	  		__debugbreak();                                                                       \
          }                                                                                       \
    } while(false)

/**
 * \brief Basic assertion macro, asserts whenever \a cond evaluates to false and prints \a message.
 * \param cond Condition to evaluate
 * \param mesg Assertion notification message
 */
#define ASSERT(cond, mesg)                                                    \
    do {                                                                      \
	    if((cond) == true) {                                                  \
	    } else {                                                              \
			printf("ASSERTION FAILED (%s:%i): %s", __FILE__, __LINE__, mesg); \
			__debugbreak();                                                   \
	    }                                                                     \
    } while(false)