#pragma once

#include "utility.h"

namespace gputil{
	/**
	 * \brief Base device structure, contains information about a specific device (GPU).
	 */
	struct device {
	public:
		device() = default;

		static inline device create(const std::function<u64(const device& device)>& selector);

		constexpr operator CUdevice() const {
			return m_device;
		}

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

		constexpr const std::string& get_name() const {
			return m_name;
		}

		constexpr const compute_capability& get_compute_capability() const {
			return m_compute_capability;
		}

		constexpr const u32 get_constant_memory() const {
			return m_constant_memory;
		}

		constexpr const u32 get_shared_memory_per_block() const {
			return m_shared_memory_per_block;
		}

		constexpr const u32 get_max_threads_per_block() const {
			return m_max_threads_per_block;
		}

		constexpr const u32 get_clock_rate() const {
			return m_clock_rate;
		}

		constexpr const bool has_unified_addressing() const {
			return m_unified_addressing;
		}

		constexpr const bool has_managed_memory() const {
			return m_managed_memory;
		}

		constexpr const u32 get_memory_bus_width() const {
			return m_memory_bus_width;
		}

		constexpr const u32 get_multiprocessor_count() const {
			return m_multiprocessor_count;
		}

		constexpr const u32 get_core_count() const {
			return m_core_count;
		}

		constexpr const u32 get_memory_clock_rate() const {
			return m_memory_clock_rate;
		}

		constexpr const u64 get_theoretical_memory_bandwidth() const {
			return m_theoretical_memory_bandwidth;
		}
	private:
		/**
		 * \brief Constructs the device using the supplied CUdevice.
		 * \param device Device to construct the device with
		 */
		device(CUdevice device) : m_device(device) {
			char device_name[100];
			CUDA_ASSERT(cuDeviceGetName(device_name, 100, m_device));
			m_name = std::string(device_name);

			m_compute_capability = {
				.major = get_device_property<u16>(device_property::compute_capability_major),
				.minor = get_device_property<u16>(device_property::compute_capability_minor)
			};

			m_constant_memory              = get_device_property<u32>(device_property::total_constant_memory);
			m_shared_memory_per_block      = get_device_property<u32>(device_property::max_shared_memory_per_block);
			m_max_threads_per_block        = get_device_property<u32>(device_property::max_threads_per_block);
			m_clock_rate                   = get_device_property<u32>(device_property::clock_rate);
			m_memory_bus_width             = get_device_property<u32>(device_property::global_memory_bus_width);
			m_multiprocessor_count         = get_device_property<u32>(device_property::multiprocessor_count);
			m_memory_clock_rate            = get_device_property<u32>(device_property::memory_clock_rate);
			m_unified_addressing           = get_device_property<bool>(device_property::unified_addressing);
			m_managed_memory               = get_device_property<bool>(device_property::managed_memory);
			m_theoretical_memory_bandwidth = static_cast<u64>(m_memory_clock_rate * 1e3 * (m_memory_bus_width / 8) * 2);
			m_core_count                   = detail::calculate_cuda_core_count(m_multiprocessor_count, m_compute_capability);
		}
	private:
		std::string m_name = {};
		compute_capability m_compute_capability = {}; // CUDA compute capability of the device
		// u64 m_global_memory = {};                    // Global memory available on the device in bytes
		u32 m_constant_memory = {};                   // Constant memory available on the device in bytes
		u32 m_shared_memory_per_block = {};           // Shared memory available per block in bytes
		u32 m_max_threads_per_block = {};             // Maximum number of threads per block
		u32 m_clock_rate = {};                        // Clock frequency in kilohertz
		u32 m_memory_bus_width = {};                  // Global memory bus width in bits
		u32 m_multiprocessor_count = {};              // Number of multiprocessors on the device
		u32 m_core_count = {};                        // Number of cores on the device
		u32 m_memory_clock_rate = {};                 // Peak memory clock frequency in kilohertz
		u64 m_theoretical_memory_bandwidth = {};      // Theoretical memory bandwidth of onboard memory units in bytes
		bool m_unified_addressing = {};               // Device shares a unified address space with the host
		bool m_managed_memory = {};                   // Device supports allocating managed memory on this system
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
		ASSERT(!devices.empty(), "no CUDA-capable device was found!");

		const auto device_comparator = [&](const device& left, const device& right) {
			return selector(left) < selector(right);
		};

		return *std::ranges::max_element(devices.begin(), devices.end(), device_comparator);
	}
}
