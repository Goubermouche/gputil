#pragma once

#include <cstdint>
#include <functional>
#include <algorithm>
#include <format>

#include <cuda.h>
// #include <nvrtc.h>

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
		* \brief CUDA device compute capability
		*/
		struct compute_capability {
			u16 major = {}; // Major compute capability version
			u16 minor = {}; // Minor compute capability version
		};

		/**
		* \brief CUDA device properties.
		*/
		enum class cuda_device_properties {
			max_threads_per_block = 1,
			max_block_dim_x = 2,
			max_block_dim_y = 3,
			max_block_dim_z = 4,
			max_grid_dim_x = 5,
			max_grid_dim_y = 6,
			max_grid_dim_z = 7,
			max_shared_memory_per_block = 8,
			// shared_memory_per_block = 8,                     [DEPRECATED] Use max_shared_memory_per_block instead
			total_constant_memory = 9,					        
			warp_size = 10,								        
			max_pitch = 11,								        
			max_registers_per_block = 12,				        
			// registers_per_block = 12                         [DEPRECATED] Use max_registers_per_block instead
			clock_rate = 13,							        
			texture_alignment = 14,						        
			gpu_overloap = 15,							        
			multiprocessor_count = 16,					        
			kernel_exec_timeout = 17,					        
			integrated = 18,							        
			can_map_host_memory = 19,					        
			compute_mode = 20,							        
			max_texture_1D_width = 21,					        
			max_texture_2D_width = 22,					        
			max_texture_2D_height = 23,					        
			max_texture_3D_width = 24,					        
			max_texture_3D_height = 25,					        
			max_texture_3D_depth = 26,					        
			max_texture_2D_layered_width = 27,			        
			max_texture_2D_layered_height = 28,			        
			max_texture_2D_layered_layers = 29,			        
			// max_texture2D_array_width = 27,                  [DEPRECATED] Use max_texture_2D_layered_width instead
			// max_texture2D_array_height = 28,                 [DEPRECATED] Use max_texture_2D_layered_height instead
			// max_texture2D_array_slice_count = 29,            [DEPRECATED] Use max_texture_2D_layered_layers instead
			surface_alignment = 30,						        
			concurrent_kernels = 31,					        
			ecc_enabled = 32,							        
			pci_bus_id = 33,							        
			pci_device_id = 34,							        
			tcc_driver = 35,							        
			memory_clock_rate = 36,						        
			global_memory_bus_width = 37,				        
			l2_cache_size = 38,							        
			max_threads_per_multiprocessor = 39,		        
			async_engine_count = 40,					        
			unified_addressing = 41,					        
			max_texture_1D_layered_width = 42,			        
			max_texture_1D_layered_layers = 43,			        
			// can_tex_2D_gather = 44,                          [DEPRECATED] Do not use
			max_texture_2D_gather_width = 45,			        
			max_texture_2D_gather_height = 46,			        
			max_texture_3D_width_alternate = 47,		        
			max_texture_3D_height_alternate = 48,		        
			max_texture_3D_depth_alternate = 49,		        
			pci_domain_id = 50,							        
			texture_pitch_alignment = 51,				        
			max_texture_cubemap_width = 52,				        
			max_texture_cubemap_layered_width = 53,		        
			max_texture_cubemap_layered_layers = 54,	        
			max_surface_1D_width = 55,					        
			max_surface_2D_width = 56,					        
			max_surface_2D_height = 57,					        
			max_surface_3D_width = 58,					        
			max_surface_3D_height = 59,					        
			max_surface_3D_depth = 60,					        
			max_surface_1D_layered_width = 61,			        
			max_surface_1D_layered_layers = 62,			        
			max_surface_2D_layered_width = 63,			        
			max_surface_2D_layered_height = 64,			        
			max_surface_2D_layered_layers = 65,			        
			max_surface_cubemap_width = 66,				        
			max_surface_cubemap_layered_width = 67,		        
			max_surface_cubemap_layered_layers = 68,	        
			// max_texture_1D_linear_width = 69,                [DEPRECATED] Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead
			max_texture_2D_linear_width = 70,
			max_texture_2D_linear_height = 71,
			max_texture_2D_linear_pitch = 72,
			max_texture_2D_mipmapped_width = 73,
			max_texture_2D_mipmapped_height = 74,
			compute_capability_major = 75,
			compute_capability_minor = 76,
			max_texture_1D_mipmapped_width = 77,
			stream_priorities_supported = 78,
			global_l1_cache_supported = 79,
			local_l1_cache_supported = 80,
			max_shared_memory_per_multiprocessor = 81,
			max_registers_per_mutliprocessor = 82,
			managed_memory = 83,
			multi_gpu_board = 84,
			multi_gpu_board_group_id = 85,
			host_native_atomic_supported = 86,
			single_to_double_performance_ratio = 87,
			pageable_memory_access = 88,
			concurrent_managed_access = 89,
			compute_preemption_supported = 90,
			can_use_host_pointer_for_registered_memory = 91,
			// can_use_stream_memory_operations_v1 = 92,        [DEPRECATED] Use cuStreamBatchMemOp and related APIs instead
			// can_use_64_bit_stream_memory_operations_v1 = 93, [DEPRECATED] Use 64-bit operations are supported in cuStreamBatchMemOp and related APIs
			// can_use_stream_wait_value_nor_v1 = 94,           [DEPRECATED] Use CU_STREAM_WAIT_VALUE_NOR instead 
			cooperative_launch = 95,
			// cooperative_multi_device_launch = 96,            [DEPRECATED]
			max_shared_memory_per_block_optin = 97,
			can_flush_remote_writes = 98,
			host_register_supported = 99,
			pageable_memory_access_uses_host_page_tables = 100,
			direct_memory_access_from_host = 101,
			virtual_memory_management_supported = 102,
			// virtual_address_management_supported = 102,      [DEPRECATED] Use virtual_memory_management_supported instead
			handle_type_posix_file_descriptor_supported = 103,
			handle_type_win32_handle_supported = 104,
			handle_type_win32_kmt_handle_supported = 105,
			max_blocks_per_multiprocessor = 106,
			generic_compression_supported = 107,
			max_persisting_l2_cache_size = 108,
			max_access_policy_window_size = 109,
			gpu_direct_rdma_with_cuda_vmm_supported = 110,
			reserved_shared_memory_per_block = 111,
			sparse_cuda_array_supported = 112,
			read_only_host_registers_supported = 113,
			timeline_semaphore_interop_supported = 114,
			memory_pools_supported = 115,
			gpu_direct_rdma_supported = 116,
			gpu_direct_rdma_flush_writes_options = 117,
			gpu_direct_rdma_writes_ordering = 118,
			memory_pool_supported_handle_types = 119,
			cluster_launch = 120,
			deferred_mapping_cuda_array_supported = 121,
			can_use_64_bit_stream_memory_operations = 122,
			can_use_stream_wait_value_nor = 123,
			dma_buffer_supported = 124,
			ipc_event_supported = 125,
			memory_sync_domain_count = 126,
			tensor_map_access_supported = 127,
			unified_function_pointers = 129
		};

		/**
		* \brief Supported memory types.
		*/
		enum class memory {
			local, // Local memory, encapsulates the default scope allocations
			device // Allocates device-side memory on the host
			// Unified?
		};
	}
}