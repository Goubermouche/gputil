#pragma once

#include "headers.h"

#define MAX_PATH 260

namespace gputil {
	namespace detail {
        static const std::unordered_map<std::string, std::string> s_jitsafe_headers = {
            { "float.h",     jitsafe_header_float_h     },
            { "limits.h",    jitsafe_header_limits_h    },
            { "stdint.h",    jitsafe_header_stdint_h    },
            { "stddef.h",    jitsafe_header_stddef_h    },
            { "stdio.h",     jitsafe_header_stdio_h     },
            { "iterator",    jitsafe_header_iterator    },
            { "limits",      jitsafe_header_limits      },
            { "type_traits", jitsafe_header_type_traits },
            { "utility",     jitsafe_header_utility     },
            { "math.h",      jitsafe_header_math_h      },
            { "complex",     jitsafe_header_complex     },
            { "algorithm",   jitsafe_header_algorithm   },
            { "stdlib.h",    jitsafe_header_stdlib_h    },
            { "assert.h",    jitsafe_header_assert_h    },
            { "iostream",    jitsafe_header_iostream    },
            { "cfloat",      jitsafe_header_float_h     },
            { "cassert",     jitsafe_header_assert_h    },
            { "cstdlib",     jitsafe_header_stdlib_h    },
            { "cmath",       jitsafe_header_math_h      },
            { "cstdio",      jitsafe_header_stdio_h     },
            { "cstddef",     jitsafe_header_stddef_h    },
            { "cstdint",     jitsafe_header_stdint_h    },
            { "climits",     jitsafe_header_limits_h    }
        };
	}
   
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

    inline void extract_linker_data_from_compiler_options(
        const std::vector<std::string>& compiler_options,
        std::vector<std::string>& linker_files,
        std::vector<std::string>& linker_paths
    ) {
        for (const std::string& option : compiler_options) {
            std::string flag = option.substr(0, 2);
            std::string value = option.substr(2);

            if (flag == "-l") {
                linker_files.push_back(value);
            }
            else if (flag == "-L") {
                linker_paths.push_back(value);
            }
        }
    }

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
#else  // Linux
            ".o"
#endif
        )) {
            return CU_JIT_INPUT_OBJECT;
        }
        else {  // Assume library
#if defined _WIN32 || defined _WIN64
            if (!ends_with(*filename, ".lib")) {
                *filename += ".lib";
            }
#else  // Linux
            if (!ends_with(*filename, ".a")) {
                *filename = "lib" + *filename + ".a";
            }
#endif
            return CU_JIT_INPUT_LIBRARY;
        }
    }

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

    inline bool extract_include_info_from_compile_error(std::string log, std::string& name) {
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

                return true;
            }
        }

        return false;
    }

    inline std::string read_file(const std::string& file_location) {
        std::ifstream file(file_location);
        return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    }

    struct kernel {
    private:
        kernel() = default;

        inline bool compile(
            const std::string& program_name,
            const std::unordered_map<std::string, std::string>& sources,
            const std::vector<std::string>& compiler_options,
            const std::string& instantiation,
            std::string& log, 
            std::string& ptx,
            std::string& mangled_instantiation
        ) const {
            std::string program_source = sources.at(program_name);
            i32 header_count = static_cast<i32>(sources.size() - 1);

            std::vector<const char*> header_names_char;
            std::vector<const char*> header_sources_char;

            header_names_char.reserve(header_count);
            header_sources_char.reserve(header_count);

            for (std::unordered_map<std::string, std::string>::const_iterator iter = sources.begin(); iter != sources.end(); ++iter) {
                const std::string& name = iter->first;
                const std::string& code = iter->second;

                if (name == program_name) {
                    continue;
                }

                header_names_char.push_back(name.c_str());
                header_sources_char.push_back(code.c_str());
            }

            std::vector<const char*> compiler_options_char(compiler_options.size());
            for (u64 i = 0; i < compiler_options.size(); i++) {
                compiler_options_char[i] = compiler_options[i].c_str();
            }

            nvrtcProgram nvrtc_program;

            NVRTC_ASSERT(nvrtcCreateProgram(
                &nvrtc_program,
                program_source.c_str(),
                program_name.c_str(),
                header_count,
                header_sources_char.data(),
                header_names_char.data())
            );

            if (!instantiation.empty()) {
                NVRTC_ASSERT(nvrtcAddNameExpression(nvrtc_program, instantiation.c_str()));
            }

            nvrtcResult result = nvrtcCompileProgram(
                nvrtc_program,
                static_cast<i32>(compiler_options_char.size()),
                compiler_options_char.data()
            );

            u64 log_size = 0;
            NVRTC_ASSERT(nvrtcGetProgramLogSize(nvrtc_program, &log_size));
            log.resize(log_size + 1);
            NVRTC_ASSERT(nvrtcGetProgramLog(nvrtc_program, log.data()));

            if (result != NVRTC_SUCCESS) {
                return false;
            }

            u64 ptx_size;
            NVRTC_ASSERT(nvrtcGetPTXSize(nvrtc_program, &ptx_size));
            ptx.resize(ptx_size + 1);
            NVRTC_ASSERT(nvrtcGetPTX(nvrtc_program, ptx.data()));

            if (instantiation.empty() == false) {
                const char* mangled_instantiation_cstr;

                NVRTC_ASSERT(nvrtcGetLoweredName(
                    nvrtc_program,
                    instantiation.c_str(),
                    &mangled_instantiation_cstr)
                );

                mangled_instantiation = mangled_instantiation_cstr;
            }

            return true;
        }

        kernel(
            const std::string& symbol,
            const std::unordered_map<std::string, std::string>& headers,
            const std::vector<std::string>& compiler_options,
            const std::string& source_file
        ) {
            std::vector<std::string> linker_files;
            std::vector<std::string> linker_paths;
            const std::string template_instantiation = "";
            const std::string instantiation = symbol + template_instantiation;
            std::string log;
            std::string ptx;
            std::string mangled_instantiation;

            extract_linker_data_from_compiler_options(
                compiler_options,
                linker_files,
                linker_paths
            );

            if(compile(
                source_file,
                headers,
                compiler_options,
                instantiation,
                log,
                ptx,
                mangled_instantiation
            )) {
                // TODO: check out the documentation for CUjit_option
                std::vector<CUjit_option> jit_options;
                std::vector<void*> jit_option_values;

                if (linker_files.empty()) {
                    CUDA_ASSERT(cuModuleLoadDataEx(
                        &m_module,
                        ptx.c_str(),
                        static_cast<u32>(jit_options.size()),
                        jit_options.data(),
                        jit_option_values.data()
                    ));
                }
            	else
                {
                    std::cout << "running linker...\n";
                    CUDA_ASSERT(cuLinkCreate(
                        static_cast<u32>(jit_options.size()),
                        jit_options.data(),
                        jit_option_values.data(),
                        &m_link_state)
                    );

                    CUDA_ASSERT(cuLinkAddData(
                        m_link_state,
                        CU_JIT_INPUT_PTX,
                        (void*)ptx.c_str(),
                        ptx.size(),
                        "jitified_source.ptx",
                        0,
                        0,
                        0
                    ));

                    CUresult cu_result;

                    for (std::string link_file : linker_files) {
                        CUjitInputType jit_input_type;
                        if (link_file == ".") {
                            // Special case for linking to current executable.
                            link_file = get_current_executable_path();
                            jit_input_type = CU_JIT_INPUT_OBJECT;
                        }
                    	else
                        {
                            // Infer based on filename.
                            jit_input_type = get_cuda_jit_input_type(&link_file);
                        }

                        cu_result = cuLinkAddFile(m_link_state, jit_input_type, link_file.c_str(), 0, 0, 0);
                        u64 path_count = 0;

                        while (cu_result == CUDA_ERROR_FILE_NOT_FOUND && path_count < (int)linker_paths.size()) {
                            std::string filename = path_join(linker_paths[path_count++], link_file);
                            cu_result = cuLinkAddFile(m_link_state, jit_input_type, filename.c_str(), 0, 0, 0);
                        }

                        CUDA_ASSERT(cu_result);
                    }

                    size_t cubin_size;
                    void* cubin;
                    CUDA_ASSERT(cuLinkComplete(m_link_state, &cubin, &cubin_size));
                    CUDA_ASSERT(cuModuleLoadData(&m_module, cubin));
                }

                CUDA_ASSERT(cuModuleGetFunction(&m_kernel, m_module, mangled_instantiation.c_str()));

                std::cout << "kernel compiled successfully\n";
                std::cout << "instantiation:         " << instantiation << '\n';
                std::cout << "mangled instantiation: " << mangled_instantiation << '\n';
            }
        	else 
            {
                throw std::runtime_error("encountered an unknown error during kernel compilation: \n" + log);
            }
        }

        inline void start_kernel(
            void** args
        ) {
            CUDA_ASSERT(cuLaunchKernel(
                m_kernel, // kernel
                1, 1, 1, // grid dim
                1, 1, 1, // block dim
                0, NULL,
                args, 0
            ));

            CUDA_ASSERT(cuCtxSynchronize());
        }
	public:
        template<class... Arguments>
        constexpr inline void start(Arguments&&... args) {
            if constexpr(sizeof...(Arguments) > 0) {
                void* pointers[sizeof...(Arguments)] = { &args... };
                start_kernel(pointers);
            }
            else {
                start_kernel(nullptr);
            }
        }
    private:
        CUfunction m_kernel;
        CUmodule m_module;
        CUlinkState m_link_state;

        friend struct program;
    };

    struct program {
    private:
        program() = default;
        program(
            const std::unordered_map<std::string, std::string>& headers,
            const std::vector<std::string>& compiler_options,
            const std::string& source_file
        ) : m_headers(headers), m_compiler_options(compiler_options), m_source_file(source_file) {}

        static inline bool compile(
            const std::string& source,
            const std::string& source_file,
            const std::vector<std::string>& compiler_options,
            std::unordered_map<std::string, std::string>& headers,
            std::string& log,
            std::string& ptx
        ) {
            std::vector<const char*> compiler_options_char(compiler_options.size());
            for (u64 i = 0; i < compiler_options.size(); i++) {
                compiler_options_char[i] = compiler_options[i].c_str();
            }

            std::vector<const char*> header_names_char;
            std::vector<const char*> header_content_char;

            header_names_char.reserve(headers.size());
            header_content_char.reserve(headers.size());

            for (const auto& header : headers) {
                header_names_char.push_back(header.first.c_str());
                header_content_char.push_back(header.second.c_str());
            }

            nvrtcProgram nvrtc_program;

            NVRTC_ASSERT(nvrtcCreateProgram(
                &nvrtc_program,
                source.c_str(),
                source_file.c_str(),
                static_cast<i32>(headers.size()),
                header_content_char.data(),
                header_names_char.data())
            );

            nvrtcResult result = nvrtcCompileProgram(
                nvrtc_program,
                static_cast<i32>(compiler_options_char.size()),
                compiler_options_char.data()
            );

            if (result != NVRTC_SUCCESS && result != NVRTC_ERROR_COMPILATION) {
                NVRTC_ASSERT(result);
            }

            u64 log_size = 0;
            NVRTC_ASSERT(nvrtcGetProgramLogSize(nvrtc_program, &log_size));
            log.resize(log_size + 1);
            NVRTC_ASSERT(nvrtcGetProgramLog(nvrtc_program, log.data()));

            if (result != NVRTC_SUCCESS) {
                return false;
            }

            u64 ptx_size;
            NVRTC_ASSERT(nvrtcGetPTXSize(nvrtc_program, &ptx_size));
            ptx.resize(ptx_size + 1);
            NVRTC_ASSERT(nvrtcGetPTX(nvrtc_program, ptx.data()));
            return true;
        }
    public:
        static inline program create(
            const std::string& source_file,
            std::vector<std::string> compiler_options = {}
        ) {
            std::unordered_map<std::string, std::string> headers;
            std::string source = read_file(source_file);
            std::string log;
            std::string ptx;
            bool result;

            compiler_options.push_back("--device-as-default-execution-space");
            compiler_options.push_back("--std=c++20"); 
            compiler_options.push_back("--dopt=on");

            std::cout << "compiler arguments:\n";
            for(const std::string& option : compiler_options) {
                std::cout << option << '\n';
            }

            while((result = compile(source, source_file, compiler_options, headers, log, ptx)) == false) {
                std::string header_name;

                if (!extract_include_info_from_compile_error(log, header_name)) {
                    break; // Not a header-related error
                }

                // Header already loaded. Something is wrong
                if (headers.count(header_name) > 0) {
                    break;
                }

                // Load missing header file
                std::string header_content;

                // Check if the file is jitsafe
                if (detail::s_jitsafe_headers.contains(header_name)) {
                    header_content = detail::s_jitsafe_headers.at(header_name);
                }
            	else {
                    // If its not, check if its a regular header file
                    try {
                        std::ifstream header_file(header_name);
                        header_content = std::string((std::istreambuf_iterator<char>(header_file)), std::istreambuf_iterator<char>());
                    }
                    catch (const std::exception& e) {
                        std::cout << "retrying compilation after error: " << e.what() << std::endl;
                    }
                }

                std::cout << "adding header: " << header_name << '\n';
                headers.emplace(std::move(header_name), std::move(header_content));
            }

            if(result) {
                std::cout << "program compiled successfully\n";
                headers.emplace(source_file, std::move(source));
                return { headers, compiler_options, source_file };
            }
            else {
                throw std::runtime_error("encountered an unknown error during program compilation: \n" + log);
            }
        }

        inline kernel get_kernel(const std::string& symbol) const {
            return { symbol, m_headers, m_compiler_options, m_source_file };
        }
    private:
        std::unordered_map<std::string, std::string> m_headers;
        std::vector<std::string> m_compiler_options;
        std::string m_source_file;
    };
}