#include "../global/types.h"

#include <functional>
#include <algorithm>
#include <format>
#include <fstream>
#include <regex>
#include <filesystem>
#include <map>
#include <windows.h>

#include <cuda.h>
#include <nvrtc.h>

namespace gputil {
    namespace types {
        using nvrtc_result = nvrtcResult;
        
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
		enum class device_property {
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
		* \brief CUDA result type.
		*/
		enum class cuda_result {
			success = 0,
			error_invalid_value = 1,
			error_memory_allocation = 2,
			error_initialization_error = 3,
			error_cudart_unloading = 4,
			error_profiler_disabled = 5,
			error_profiler_not_initialized = 6,
			error_profiler_already_started = 7,
			error_profiler_already_stopped = 8,
			error_invalid_configuration = 9,
			error_invalid_pitch_value = 12,
			error_invalid_symbol = 13,
			// error_invalid_host_pointer = 16, // Deprecated
			error_invalid_device_pointer = 17, // Deprecated
			error_invalid_texture = 18,
			error_invalid_texture_binding = 19,
			error_invalid_channel_descriptor = 20,
			error_invalid_memcpy_direction = 21,
			// error_address_of_constant = 22, // Deprecated
			// error_texture_fetch_failed = 23, // Deprecated
			// error_texture_not_bound = 24, // Deprecated
			// error_synchronization_error = 25, // Deprecated
			error_invalid_filter_setting = 26,
			error_invalid_norm_setting = 27,
			// error_mixed_device_execution = 28, // Deprecated
			// error_not_yet_implemented = 31, // Deprecated
			// error_memory_value_too_large = 32, // Deprecated
			error_stub_library = 34,
			error_insufficient_driver = 35,
			error_call_requires_newer_driver = 36,
			error_invalid_surface = 37,
			error_duplicate_variable_name = 43,
			error_duplicate_texture_name = 44,
			error_duplicate_surface_name = 45,
			error_devices_unavailable = 46,
			error_incompatible_driver_context = 49,
			error_missing_configuration = 52,
			// error_prior_launch_failure = 53, // Deprecated
			error_launch_max_depth_exceeded = 65,
			error_launch_file_scoped_texture = 66,
			error_launch_file_scoped_surface = 67,
			error_synchronization_depth_exceeded = 68,
			error_launch_pending_count_exceeded = 69,
			error_invalid_device_function = 98,
			error_no_device = 100,
			error_invalid_device = 101,
			error_device_not_licensed = 102,
			error_software_validity_not_established = 103,
			error_startup_failure = 127,
			error_invalid_kernel_image = 200,
			error_device_uninitialized = 201,
			error_map_buffer_object_failed = 205,
			error_unmap_buffer_object_failed = 206,
			error_array_is_mapped = 207,
			error_already_mapped = 208,
			error_no_kernel_image_for_device = 209,
			error_already_acquired = 210,
			error_not_mapped = 211,
			error_not_mapped_as_array = 212,
			error_not_mapped_as_pointer = 213,
			error_ecc_uncorrectable = 214,
			error_unsupported_limit = 215,
			error_device_already_in_use = 216,
			error_peer_access_unsupported = 217,
			error_invalid_ptx = 218,
			error_invalid_graphics_context = 219,
			error_nvlink_uncorrectable = 220,
			error_jit_compilation_not_found = 221,
			error_unsupported_ptx_version = 222,
			error_jit_compilation_disabled = 223,
			error_unsupported_exec_affinity = 224,
			error_invalid_source = 300,
			error_file_not_found = 301,
			error_shared_object_symbol_not_found = 302,
			error_shared_object_initialization_failed = 303,
			error_operating_system = 304,
			error_invalid_resource_handle = 400,
			error_illegal_state = 401,
			error_symbol_not_found = 500,
			error_not_ready = 600,
			error_illegal_address = 700,
			error_launch_out_of_resources = 701,
			error_launch_timeout = 702,
			error_launch_incompatible_texturing = 703,
			error_peer_access_already_enabled = 704,
			error_peer_access_not_enabled = 705,
			error_set_on_active_process = 708,
			error_context_is_destroyed = 709,
			error_assert = 710,
			error_too_many_peers = 711,
			error_host_memory_already_registered = 712,
			error_host_memory_not_registered = 713,
			error_hardware_stack_error = 714,
			error_illegal_instruction = 715,
			error_misaligned_address = 716,
			error_invalid_address_space = 717,
			error_invalid_program_counter = 718,
			error_launch_failure = 719,
			error_cooperative_launch_too_large = 720,
			error_not_permitted = 800,
			error_not_supported = 801,
			error_system_not_ready = 802,
			error_system_driver_mismatch = 803,
			error_compatibility_not_supported_on_device = 804,
			error_mps_connection_failed = 805,
			error_mps_rps_failure = 806,
			error_mps_server_not_ready = 807,
			error_mps_max_clients_reached = 808,
			error_mps_max_connections_reached = 809,
			error_mps_client_terminated = 810,
			error_cdp_not_supported = 811,
			error_cdp_version_mismatch = 812,
			error_stream_capture_unsupported = 900,
			error_stream_capture_invalidated = 901,
			error_stream_capture_merge = 902,
			error_stream_capture_unmatched = 903,
			error_stream_capture_unjoined = 904,
			error_stream_capture_isolation = 905,
			error_stream_capture_implicit = 906,
			error_captured_event = 907,
			error_stream_capture_wrong_thread = 908,
			error_timeout = 909,
			error_graph_execution_update_failure = 910,
			error_external_device = 911,
			error_invalid_cluster_size = 912,
			error_unknown = 999,
			error_api_failure_base = 100000
		};

		/**
		* \brief CUDA context flags.
		*/
		enum class context_flag {
			automatic_scheduling = 0x00,
			spin_scheduling = 0x01,
			yield_scheduling = 0x02,
			blocking_scheduling = 0x04,
			keep_local_memory_allocations_after_launch = 0x10
		}; 
    }
}
