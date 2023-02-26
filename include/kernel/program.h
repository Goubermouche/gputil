#pragma once

#include "headers.h"

namespace gputil {

    struct program {
    public:
        program() = default;

        static inline program create(const std::string& file_path);
    private:
        program(nvrtcProgram program) : m_program(program) {
	        
        }

    private:
        nvrtcProgram m_program;
    };

    bool file_inside_of_included_directory(const std::string file_path) {
        return false;
    }

    //void read_program(const std::string& file_path) {
    //    // Check if the file exists 
	   // if(std::filesystem::exists(file_path)) {
    //        std::cout << file_path << "   [regular file]\n";
    //    // Check if the file is inside of an included directory
	   // } else if(detail::s_jitsafe_headers.contains(file_path)) {
    //        std::cout << file_path << "   [jitsafe header]\n";
	   // } else {
    //        std::cout << file_path << "   [does not exist]\n";
    //    }
    //}

    //inline program program::create(const std::string& file_path) {
    //    //std::ifstream file(file_path);
    //    //// Note: We have to keep the parentheses in the first argument, otherwise we can't compile.
    //    //std::string program_string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()); 

    //    //// Extract #includes from the given file
    //    //const std::regex include_regex("#include\\s+[\"<]([^\">]+)[\">]");
    //    //std::vector<std::string> program_includes = extract_regex_from_string(program_string, include_regex);

    //    //for(const std::string include : program_includes) {
    //    //    // std::cout << include << '\n';
    //    //    read_program(include);
    //    //}

    //    //nvrtcProgram nvrtc_program = {};

    //    //NVRTC_ASSERT(nvrtcCreateProgram(
    //    //    &nvrtc_program,
    //    //    program_string.c_str(),
    //    //    source_file.c_str(),
    //    //    0,
    //    //    0,
    //    //    0
    //    //));

    //    //const char* opts[] = { "--fmad=false" };
    //    //const nvrtc_result compilation_status = nvrtcCompileProgram(
    //    //    nvrtc_program,
    //    //    1,
    //    //    opts
    //    //);

    //    //u64 log_size;
    //    //NVRTC_ASSERT(nvrtcGetProgramLogSize(nvrtc_program, &log_size));
    //    //char* log = new char[log_size];
    //    //NVRTC_ASSERT(nvrtcGetProgramLog(nvrtc_program, log));
    //    //std::cout << log << '\n';
    //    //delete[] log;

    //    return program(nvrtc_program);
    //}

    static const std::regex s_include_regex("#include\\s+[\"<]([^\">]+)[\">]");
    static std::unordered_map<std::string, std::string> s_programs;
    static std::unordered_map<std::string, std::string> s_jitsafe_headers = {
        { "float.h", jitsafe_header_float_h         },
        { "limits.h", jitsafe_header_limits_h       },
        { "stdint.h", jitsafe_header_stdint_h       },
        { "stddef.h", jitsafe_header_stddef_h       },
        { "stdio.h", jitsafe_header_stdio_h         },
        { "iterator", jitsafe_header_iterator       },
        { "limits", jitsafe_header_limits           },
        { "type_traits", jitsafe_header_type_traits },
        { "utility", jitsafe_header_utility         },
        { "math.h", jitsafe_header_math_h           },
        { "complex", jitsafe_header_complex         },
        { "algorithm", jitsafe_header_algorithm     },
        { "stdlib.h", jitsafe_header_stdlib_h       },
        { "assert.h", jitsafe_header_assert_h       },
        { "iostream", jitsafe_header_iostream       },

        { "cfloat", jitsafe_header_float_h          },
        { "cassert", jitsafe_header_assert_h        },
        { "cstdlib", jitsafe_header_stdlib_h        },
        { "cmath", jitsafe_header_math_h            },
        { "cstdio", jitsafe_header_stdio_h          },
        { "cstddef", jitsafe_header_stddef_h        },
        { "cstdint", jitsafe_header_stdint_h        },
        { "climits", jitsafe_header_limits_h        },
    };

    void walk_program(const std::string& location, const u64 debug_depth, std::map<std::string, std::string>& include_files) {
        std::ifstream file(location);
        std::string program_string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        std::vector<std::string> program_includes = extract_regex_from_string(program_string, s_include_regex);

        std::cout << std::string(debug_depth * 2, ' ') << location << "-------------\n";
        for(const std::string& include : program_includes) {
            if(s_jitsafe_headers.contains(include)) {
                include_files[include] = s_jitsafe_headers[include];
                //std::cout << std::string(debug_depth * 2, ' ') << "[JITSAFE]   " << include << '\n';
            } else if(std::filesystem::exists(include)) {
                //std::cout << std::string(debug_depth * 2, ' ') << "[INCLUDE]   " << include << '\n';

                // Look for #includes in the child include
                walk_program(include, debug_depth + 1, include_files);
            } else {
                //std::cout << std::string(debug_depth * 2, ' ') << "[DOES NOT EXIST]   " << include << '\n';
            }
        }
    }

    bool compile_program(
        const std::string& kernel_source,
        const std::string& kernel_file,
        const std::unordered_map<std::string, std::string>& headers,
        std::string& log
    ) {
        std::vector<const char*> headers_names;
        std::vector<const char*> headers_content;

        size_t num_headers = headers.size() + s_jitsafe_headers.size();
        headers_names.reserve(num_headers);
        headers_content.reserve(num_headers);

        for (const auto& p : headers) {
            headers_names.push_back(p.first.c_str());
            headers_content.push_back(p.second.c_str());
        }

        for (const auto& p : s_jitsafe_headers) {
            headers_names.push_back(p.first.c_str());
            headers_content.push_back(p.second.c_str());
        }

        nvrtcProgram nvrtc_program;

        for(const auto h : headers_names) {
           std::cout << h << '\n';
        }

        NVRTC_ASSERT(nvrtcCreateProgram(
            &nvrtc_program,
            kernel_source.c_str(),
            kernel_file.c_str(),
            static_cast<int>(num_headers),
            headers_content.data(),
            headers_names.data())
        );

        const char* opts[] = { "-default-device" };
        nvrtcResult result = nvrtcCompileProgram(
            nvrtc_program,
            1,
            opts);

        if (result != NVRTC_SUCCESS && result != NVRTC_ERROR_COMPILATION) {
            NVRTC_ASSERT(result);
        }

        size_t log_size = 0;
        NVRTC_ASSERT(nvrtcGetProgramLogSize(nvrtc_program, &log_size));
        log.resize(log_size + 1);
        NVRTC_ASSERT(nvrtcGetProgramLog(nvrtc_program, &log[0]));

        if (result != NVRTC_SUCCESS) {
            return false;
        }

        return true;
    }

    static bool extract_unknown_header_from_log(
        const std::string& log,
        std::string& filename_out) {
        static constexpr auto patterns = {
            "could not open source file \"",
            "cannot open source file \"" };

        for (const char* pattern : patterns) {
            size_t begin = log.find(pattern);

            if (begin == std::string::npos) {
                continue;
            }

            begin += strlen(pattern);
            size_t end = log.find('\"', begin + 1);

            if (end == std::string::npos || end <= begin) {
                continue;
            }

            filename_out = log.substr(begin, end - begin);
            return true;
        }

        return false;
    }


    void create_program(const std::string& location) {
        std::vector<std::string> dirs = {};

        std::ifstream file(location);
        std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        std::unordered_map<std::string, std::string> headers;
        std::string log;

        for (int i = 0; i < 10; ++i) {
            bool success = compile_program(
                source,
                location,
                headers,
                log
            );

            std::cout << "NVRTC compilation of " << location << ":" << log << std::endl;

            if (success) {
                std::cout << "done compiling\n";
                return;
            }

            std::string header_name;
            // See if compilation failed due to missing header file
            if (!extract_unknown_header_from_log(log, header_name)) {
                break;
            }

            // Header already loaded. Something is wrong?
            if (headers.count(header_name) > 0) {
                break;
            }

            // Load missing header file
            std::string header_content;
            try {
                std::ifstream header_file(header_name);
                header_content = std::string((std::istreambuf_iterator<char>(header_file)), std::istreambuf_iterator<char>());
            }
            catch (const std::exception& e) {
                std::cout << "retrying compilation after error: " << e.what() << std::endl;
            }

            headers.emplace(std::move(header_name), std::move(header_content));
        }
    }
}