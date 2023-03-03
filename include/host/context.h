#pragma once

#include "device.h"

namespace gputil {
	// TODO: Implement context thread affinity. 
	/**
	 * \brief Context handle manager.
	 */
	struct context {
	public:
		context() = default;

		static inline context create(const device& device, const context_flag flags = context_flag::automatic_scheduling);

		/**
		 * \brief Destroys the context.
		 */
		void destroy() const {
			CUDA_ASSERT(cuCtxDestroy(m_context));
		}
	private:
		/**
		 * \brief Constructs the context using the supplied CUcontext.
		 * \param context Context to construct the context with
		 */
		context(CUcontext context) : m_context(context) {}
	private:
		CUcontext m_context;
	};

	/**
	 * \brief Initializes a new context using the specified \a device.
	 * \param device Device to initialize the context with
	 * \param flags Context creation flags
	 * \returns Newly created context
	 */
	inline context context::create(const device& device, const context_flag flags) {
		CUcontext cu_context;
		CUDA_ASSERT(cuCtxCreate(&cu_context, static_cast<u32>(flags), device));

		return context(cu_context);
	}
}
