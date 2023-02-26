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

    bool compile_program(
        const std::string& kernel_source,
        const std::string& kernel_file,
        const std::unordered_map<std::string, std::string>& headers,
        std::string& log
    ) {
        std::vector<const char*> headers_names;
        std::vector<const char*> headers_content;

        headers_names.reserve(headers.size());
        headers_content.reserve(headers.size());

        for (const auto& p : headers) {
            headers_names.push_back(p.first.c_str());
            headers_content.push_back(p.second.c_str());
        }

        nvrtcProgram nvrtc_program;

        NVRTC_ASSERT(nvrtcCreateProgram(
            &nvrtc_program,
            kernel_source.c_str(),
            kernel_file.c_str(),
            static_cast<i32>(headers.size()),
            headers_content.data(),
            headers_names.data())
        );

        const char* opts[] = { "-default-device" };
        nvrtcResult result = nvrtcCompileProgram(
            nvrtc_program,
            1,
            opts
        );

        if (result != NVRTC_SUCCESS && result != NVRTC_ERROR_COMPILATION) {
            NVRTC_ASSERT(result);
        }

        u64 log_size = 0;
        NVRTC_ASSERT(nvrtcGetProgramLogSize(nvrtc_program, &log_size));
        log.resize(log_size + 1);
        NVRTC_ASSERT(nvrtcGetProgramLog(nvrtc_program, &log[0]));

        if (result != NVRTC_SUCCESS) {
            return false;
        }

        // TEMP: Kernel test
        u64 ptxSize;
        NVRTC_ASSERT(nvrtcGetPTXSize(nvrtc_program, &ptxSize));
        char* ptx = new char[ptxSize];
        NVRTC_ASSERT(nvrtcGetPTX(nvrtc_program, ptx));
        NVRTC_ASSERT(nvrtcDestroyProgram(&nvrtc_program)); // Destroy the program.

        // Load the generated PTX and get a handle to the SAXPY kernel.
        CUmodule module;
        CUfunction kernel;

        CUDA_ASSERT(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
        CUDA_ASSERT(cuModuleGetFunction(&kernel, module, "test_kernel"));

        CUDA_ASSERT(cuLaunchKernel(
            kernel,  // kernel
            1, 1, 1, // grid dim
            1, 1, 1, // block dim
            0, NULL, // shared mem and stream
            0, 0     // arguments
        ));

        CUDA_ASSERT(cuCtxSynchronize());

        return true;
    }

    inline bool extract_include_info_from_compile_error(std::string log,
        std::string& name,
        std::string& parent,
        int& line_num
    ) {
    	static const std::vector<std::string> pattern = {
            "could not open source file \"",
        	"cannot open source file \""
        };

        for (const std::string& p : pattern) {
            u64 beg = log.find(p);

            if (beg != std::string::npos) {
                beg += p.size();
                u64 end = log.find("\"", beg);
                name = log.substr(beg, end - beg);
                u64 line_beg = log.rfind("\n", beg);

                if (line_beg == std::string::npos) {
                    line_beg = 0;
                }
                else {
                    line_beg += 1;
                }

                u64 split = log.find("(", line_beg);
                parent = log.substr(line_beg, split - line_beg);
                line_num = atoi(log.substr(split + 1, log.find(")", split + 1) - (split + 1)).c_str());

                return true;
            }
        }

        return false;
    }

    void create_program(const std::string& location) {
        std::ifstream file(location);
        std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        std::unordered_map<std::string, std::string> headers;
        std::string log;
        bool success;

        while ((success = compile_program(source, location, headers, log)) == false) {
            std::cout << "NVRTC compilation of " << location << ":\n";

            if (success) {
                std::cout << "done compiling\n";
                return;
            }

            std::cout << "retrying compilation after error: \n\n" << log << '\n';

            std::string header_name;
            std::string header_parent;
            int header_line_num;

            if(!extract_include_info_from_compile_error(log, header_name, header_parent, header_line_num)) {
                break; // Not a header-related error
            }

            // Header already loaded. Something is wrong
            if (headers.count(header_name) > 0) {
                break;
            }

            // Load missing header file
            std::string header_content;

            // Check if the file is jitsafe
            if(s_jitsafe_headers.contains(header_name)) {
                header_content = s_jitsafe_headers[header_name];
            // If its not, check if its a regular header file
            } else {
                try {
                    std::ifstream header_file(header_name);
                    header_content = std::string((std::istreambuf_iterator<char>(header_file)), std::istreambuf_iterator<char>());
                } catch (const std::exception& e) {
                    std::cout << "retrying compilation after error: " << e.what() << std::endl;
                }
            }

            headers.emplace(std::move(header_name), std::move(header_content));
            std::cout << "---------------------------------------------------------------------------\n";
        }
    }
}