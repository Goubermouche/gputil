#pragma once

#include "macros.h"

namespace gputil::detail {
	/**
	 * \brief Formats the given /a byte_count into a string containing the appropriate file size extension.
	 * \param byte_count Count of bytes to format
	 * \returns String containing the format byte count
	 */
	inline std::string format_bytes(const u64 byte_count) {
		const char* formats[] = { "B", "K", "MB", "GB", "TB", "PB" };

		f64 value = static_cast<f64>(byte_count);
		u8 index = 0;

		// Note: we should probably use 1024.0 to be more accurate, but 1000.0 appears to provide
		//       results that are more in line with producer specs.
		while((value / 1000.0) >= 1.0) {
			value = value / 1000.0;
			index++;
		}

		return std::format("{:.2f} {}", value, formats[index]);
	}

	/**
	 * \brief Calculates the number of CUDA cores available on the specified architecture.
	 * \param multiprocessor_count Multiprocessor count
	 * \param major Major compute capability version
	 * \param minor Minor compute capability version
	 * \returns Count of CUDA cores on the specified architecture
	 */
	constexpr inline u32 calculate_cuda_core_count(const u32 multiprocessor_count, const compute_capability& compute_capability) {
		u32 core_count = 0;

		switch (compute_capability.major) {
		case 2: // fermi
			if (compute_capability.minor) {
				core_count = multiprocessor_count * 48;
			}
			else {
				core_count = multiprocessor_count * 32;
			}
			break;
		case 3: // kepler
			core_count = multiprocessor_count * 192;
			break;
		case 5: // maxwell
			core_count = multiprocessor_count * 128;
			break;
		case 6: // pascal
			if (compute_capability.minor == 1 || compute_capability.minor == 2) {
				core_count = multiprocessor_count * 128;
			}
			else if (compute_capability.minor == 0) {
				core_count = multiprocessor_count * 64;
			}
			else {
				ASSERT(false, "unknown device type");
			}
			break;
		case 7: // volta, turing
			if (compute_capability.minor == 0 || compute_capability.minor == 5) {
				core_count = multiprocessor_count * 64;
			}
			else {
				ASSERT(false, "unknown device type");
			}
			break;
		case 8: // ampere
			if (compute_capability.minor == 0) {
				core_count = multiprocessor_count * 64;
			}
			else if (compute_capability.minor == 6 || compute_capability.minor == 9) {
				core_count = multiprocessor_count * 128; // ada lovelace
			}
			else {
				ASSERT(false, "unknown device type");
			}
			break;
		case 9: // hopper
			if (compute_capability.minor == 0) {
				core_count = multiprocessor_count * 128;
			}
			else {
				ASSERT(false, "unknown device type");
			}
			break;
		default:
			ASSERT(false, "unknown device type");
			break;
		}

		return core_count;
	}

	/**
	 * \brief Looks for all instances of text that matches the given \a regex and extracts it.
	 * \param text Text to search
	 * \param regex Regex to use for searching
	 * \returns Vector of strings that match the given \a regex
	 */
	inline std::vector<std::string> extract_regex_from_string(const std::string& text, const std::regex& regex) {
		std::vector<std::string> result;

		std::smatch match;
		std::string::const_iterator search_start(text.cbegin());

		while (std::regex_search(search_start, text.cend(), match, regex)) {
			result.emplace_back(match[1]);
			search_start = match.suffix().first;
		}

		return result;
	}

	/**
	 * \brief Acquires the current executable location.
	 * \returns Current executable location
	 */
	inline const char* get_current_executable_path() {
		static char buffer[MAX_PATH] = {};

#ifdef __linux__
		if (!::realpath("/proc/self/exe", buffer)) {
			return nullptr;
		}
#elif defined(_WIN32) || defined(_WIN64)
		if (!GetModuleFileNameA(nullptr, buffer, MAX_PATH)) {
			return nullptr;
		}
#endif
		return buffer;
	}

	/**
	 * \brief Checks if \a str ends with \a suffix.
	 * \param str Str to check
	 * \param suffix Suffix to look for
	 * \returns True if \a str ends with \a suffic, otherwise returns False
	 */
	constexpr inline bool ends_with(const std::string& str, const std::string& suffix) {
		return str.size() >= suffix.size() && str.substr(str.size() - suffix.size()) == suffix;
	}

	/**
	 * \brief Infers the JIT input type from the filename suffix. If no known suffix is present, the filename is assumed to refer to a library, and the associated suffix (and possibly prefix) is automatically added to the filename.
	 * \param filename Filename to check
	 * \returns Inferred CUjitInputType
	 */
	constexpr inline CUjitInputType get_cuda_jit_input_type(std::string* filename) {
		if (ends_with(*filename, ".ptx")) {
			return CU_JIT_INPUT_PTX;
		}
		else if (ends_with(*filename, ".cubin")) {
			return CU_JIT_INPUT_CUBIN;
		}
		else if (ends_with(*filename, ".fatbin")) {
			return CU_JIT_INPUT_FATBINARY;
		}
		else if (ends_with(*filename,
#if defined _WIN32 || defined _WIN64
			".obj"
#else  // linux
			".o"
#endif
		)) {
			return CU_JIT_INPUT_OBJECT;
		}
		else {  // assume library
#if defined _WIN32 || defined _WIN64
			if (!ends_with(*filename, ".lib")) {
				*filename += ".lib";
			}
#else  // linux
			if (!ends_with(*filename, ".a")) {
				*filename = "lib" + *filename + ".a";
			}
#endif
			return CU_JIT_INPUT_LIBRARY;
		}
	}

	/**
	 * \brief Joins \a p1 and \a p2 into a single path.
	 * \returns Path consisting of the two specified paths
	 */
	constexpr inline std::string path_join(std::string p1, std::string p2) {
#ifdef _WIN32
		char sep = '\\';
#else
		char sep = '/';
#endif
		if (p1.size() && p2.size() && p2[0] == sep) {
			throw std::invalid_argument("cannot join to absolute path");
		}

		if (p1.size() && p1[p1.size() - 1] != sep) {
			p1 += sep;
		}

		return p1 + p2;
	}

	/**
	 * \brief Reads the text file at the given \a filepath.
	 * \returns String containing the file contents
	 */
	inline std::string read_file(const std::string& filepath) {
		std::ifstream file(filepath);

		if(file) {
			return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		}
		else {
			throw std::runtime_error("cannot read file '" + filepath + "'");
		}
	}
}