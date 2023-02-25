#pragma once

#include "headers.h"

namespace gputil {
	namespace detail {
        static const std::unordered_map<std::string, std::string> s_jitsafe_headers = {
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

             { "cfloat", jitsafe_header_float_h          },
             { "cassert", jitsafe_header_assert_h        },
             { "cstdlib", jitsafe_header_stdlib_h        },
             { "cmath", jitsafe_header_math_h            },
             { "cstdio", jitsafe_header_stdio_h          },
             { "cstddef", jitsafe_header_stddef_h        },
             { "cstdint", jitsafe_header_stdint_h        },
             { "climits", jitsafe_header_limits_h        },
        };
	}
   

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

    void read_program(const std::string& file_path) {
        // Check if the file exists 
	    if(std::filesystem::exists(file_path)) {
            std::cout << file_path << " exists\n";
        // Check if the file is inside of an included directory
	    } else if(detail::s_jitsafe_headers.contains(file_path)) {
            std::cout << file_path << " is a jitsafe header\n";
	    } else {
            std::cout << file_path << " does not exist\n";
        }
    }

    inline program program::create(const std::string& file_path) {
        std::ifstream file(file_path);
        // Note: We have to keep the parentheses in the first argument, otherwise we can't compile.
        std::string program_string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()); 

        // Extract #includes from the given file
        const std::regex include_regex("#include\\s+[\"<]([^\">]+)[\">]");
        std::vector<std::string> program_includes = extract_regex_from_string(program_string, include_regex);

        for(const std::string include : program_includes) {
            // std::cout << include << '\n';
            read_program(include);
        }

        nvrtcProgram nvrtc_program = {};
        //NVRTC_ASSERT(nvrtcCreateProgram(
        //    &nvrtc_program,
        //    program_string.c_str(),
        //    source_file.c_str(),
        //    0,
        //    0,
        //    0
        //));

        //const char* opts[] = { "--fmad=false" };
        //const nvrtc_result compilation_status = nvrtcCompileProgram(
        //    nvrtc_program,
        //    1,
        //    opts
        //);

        //u64 log_size;
        //NVRTC_ASSERT(nvrtcGetProgramLogSize(nvrtc_program, &log_size));
        //char* log = new char[log_size];
        //NVRTC_ASSERT(nvrtcGetProgramLog(nvrtc_program, log));
        //std::cout << log << '\n';
        //delete[] log;

        return program(nvrtc_program);
    }
}