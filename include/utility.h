#pragma once

#include "macros.h"

namespace gputil {
	/**
	* \brief Formats the given /a byte_count into a string containing the appropriate file size extension.
	* \param byte_count Count of bytes to format
	* \returns String containing the format byte count
	*/
	inline std::string format_bytes(const u64 byte_count) {
		const char* formats[] = { "B", "K", "MB", "GB", "TB", "PB" };

		f64 value = static_cast<f64>(byte_count);
		u8 index = 0;

		// Note: We should probably use 1024.0 to be more accurate, but 1000.0 appears to provide
		//       results that are more accurate to producer specs.
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
		case 2: // Fermi
			if (compute_capability.minor) {
				core_count = multiprocessor_count * 48;
			}
			else {
				core_count = multiprocessor_count * 32;
			}
			break;
		case 3: // Kepler
			core_count = multiprocessor_count * 192;
			break;
		case 5: // Maxwell
			core_count = multiprocessor_count * 128;
			break;
		case 6: // Pascal
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
		case 7: // Volta, Turing
			if (compute_capability.minor == 0 || compute_capability.minor == 5) {
				core_count = multiprocessor_count * 64;
			}
			else {
				ASSERT(false, "unknown device type");
			}
			break;
		case 8: // Ampere
			if (compute_capability.minor == 0) {
				core_count = multiprocessor_count * 64;
			}
			else if (compute_capability.minor == 6 || compute_capability.minor == 9) {
				core_count = multiprocessor_count * 128; // Ada Lovelace
			}
			else {
				ASSERT(false, "unknown device type");
			}
			break;
		case 9: // Hopper
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
}