#pragma once

#include "utility.h"

namespace gputil{
	/**
	 * \brief Base device structure, contains information about a specific device (GPU).
	 */
	struct device {
	public:
		device() = default;

		static device create(const std::function<u64(const device& device)>& selector);

		/**
		 * \brief Acquires the requested device property for the device.
		 * \tparam T Property type
		 * \param property Property to acquire
		 * \returns Value that maps to the requested value
		 */
		template<class T>
		inline T get_device_property(const device_property property) const {
			i32 result;
			CUDA_ASSERT(cuDeviceGetAttribute(&result, static_cast<CUdevice_attribute>(property), m_device));
			return static_cast<T>(result);
		}

		inline constexpr CUdevice get() const {
			return  m_device;
		}
	private:
		/**
		 * \brief Constructs the device using the supplied CUdevice.
		 * \param device Device to construct the device with
		 */
		device(CUdevice device) : m_device(device) {
			char device_name[100];
			CUDA_ASSERT(cuDeviceGetName(device_name, 100, m_device));
			name = std::string(device_name);

			compute_capability = {
				.major = get_device_property<u16>(device_property::compute_capability_major),
				.minor = get_device_property<u16>(device_property::compute_capability_minor)
			};

			constant_memory              = get_device_property<u32>(device_property::total_constant_memory);
			shared_memory_per_block      = get_device_property<u32>(device_property::max_shared_memory_per_block);
			max_threads_per_block        = get_device_property<u32>(device_property::max_threads_per_block);
			clock_rate                   = get_device_property<u32>(device_property::clock_rate);
			unified_addressing           = get_device_property<bool>(device_property::unified_addressing);
			memory_bus_width             = get_device_property<u32>(device_property::global_memory_bus_width);
			managed_memory               = get_device_property<bool>(device_property::managed_memory);
			multiprocessor_count         = get_device_property<u32>(device_property::multiprocessor_count);
			memory_clock_rate            = get_device_property<u32>(device_property::memory_clock_rate);
			theoretical_memory_bandwidth = static_cast<u64>(memory_clock_rate * 1e3 * (memory_bus_width / 8) * 2);
			core_count                   = calculate_cuda_core_count(multiprocessor_count, compute_capability);
		}
	public:
		std::string name = {};
		compute_capability compute_capability = {}; // CUDA compute capability of the device
		// u64 global_memory = {};                  // Global memory available on the device in bytes
		u32 constant_memory = {};                   // Constant memory available on the device in bytes
		u32 shared_memory_per_block = {};           // Shared memory available per block in bytes
		u32 max_threads_per_block = {};             // Maximum number of threads per block
		u32 clock_rate = {};                        // Clock frequency in kilohertz
		bool unified_addressing = {};               // Device shares a unified address space with the host
		u32 memory_bus_width = {};                  // Global memory bus width in bits
		bool managed_memory = {};                   // Device supports allocating managed memory on this system
		u32 multiprocessor_count = {};              // Number of multiprocessors on the device
		u32 core_count = {};                        // Number of cores on the device
		u32 memory_clock_rate = {};                 // Peak memory clock frequency in kilohertz
		u64 theoretical_memory_bandwidth = {};      // Theoretical memory bandwidth of onboard memory units in bytes
	private:
		CUdevice m_device = {};

		friend inline std::vector<device> get_available_devices();
	};

	/**
	 * \brief Returns all available devices on the current system.
	 * \return All available devices on the current system
	 */
	inline std::vector<device> get_available_devices() {
		i32 device_count;
		CUDA_ASSERT(cuDeviceGetCount(&device_count));
		std::vector<device> devices(device_count);

		for (i32 i = 0; i < device_count; ++i) {
			CUdevice cu_device;
			CUDA_ASSERT(cuDeviceGet(&cu_device, i));

			devices[i] = device(cu_device);
		}

		return devices;
	}

	/**
	 * \brief Chooses a device with the highest score generated by the specified \a selector.
	 * \param selector Device selector that generates score based on device parameters
	 * \return Chosen device
	 */
	device inline device::create(const std::function<u64(const device& device)>& selector) {
		const std::vector<device> devices = get_available_devices();
		ASSERT(!devices.empty(), "No CUDA-capable device was found!");

		const auto device_comparator = [&](const device& left, const device& right) {
			return selector(left) < selector(right);
		};

		return *std::ranges::max_element(devices.begin(), devices.end(), device_comparator);
	}
}