#pragma once

#include "headers.h"
#include "stream.h"

namespace gputil {
    /**
     * \brief Extracts a set of linker files and paths from the given list of compiler options.
     * \param compiler_options Compiler options to located the linker options in
     * \param linker_files Output linker files
     * \param linker_paths Output linker paths
     */
    inline void extract_linker_data_from_compiler_options(
        const std::vector<std::string>& compiler_options,
        std::vector<std::string>& linker_files,
        std::vector<std::string>& linker_paths
    ) {
        for (const std::string& option : compiler_options) {
            std::string flag = option.substr(0, 2);
            std::string value = option.substr(2);

            if (flag == "-l") {
                linker_files.emplace_back(value);
            }
            else if (flag == "-L") {
                linker_paths.emplace_back(value);
            }
        }
    }

    // inline void extract_include_directories_from_compiler_options(
    //     const std::vector<std::string>& compiler_options,
    //     std::vector<std::string>& include_directories
    // ) {
    //     static const std::vector<std::string> pattern = {
    //         "--include-path="
    //     };
       
    //     for (const std::string& option : compiler_options) {
    //         std::cout << option << "\n";
       
    //         for(const std::string& p : pattern) {
    //             u64 pos = option.rfind(p, 0);
       
    //             if(pos == std::string::npos) {
    //                 continue;
    //             }
       
    //             const u64 p_size = p.size();
    //             include_directories.push_back(option.substr(p_size));
    //         }
    //     }
    // }

    /**
	 * \brief Extracts the header name from a header-related compiler error.
	 * \param log Log to find the header name in
	 * \param name Output header name
	 * \returns True, if the operation was successful, otherwise false
	 */
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

    /**
     * \brief 3-component dimension structure, used for storing kernel dimensions.
     */
    struct kernel_dim {
        kernel_dim() = default;
        kernel_dim(const u32 x) : m_dim{ x, 1, 1 } {}
        kernel_dim(const u32 x, const u32 y) : m_dim{ x, y, 1 } {}
        kernel_dim(const u32 x, const u32 y, const u32 z) : m_dim{ x, y, z } {}

        u32& operator[](i32 index) { return m_dim[index]; }
        const u32& operator[](i32 index) const { return m_dim[index]; }
	private:
        u32 m_dim[3] = { 1, 1, 1 };
	};

    /**
     * \brief Kernel options structure, supplied to the \a kernel structure when running it.
     */
    struct kernel_options {
        kernel_dim thread_count = { 1, 1, 1 }; // Number of threads to start in each dimension (default {1, 1, 1})
        kernel_dim block_count = { 1, 1, 1 };  // Number of blocks to start in each dimension (default {1, 1, 1})
        u32 shared_memory_size = {};           // Dynamic shared-memory size per thread block in bytes
        stream stream = {};                    // Execution stream
	};

    /**
     * \brief Kernel structure; contains information about, and wraps around the CUDA driver API's kernel.
     */
    struct kernel {
    private:
        kernel() = default;

        /**
	     * \brief Compiles the kernel.
         * \param program_name Name of the source program (program filepath)
         * \param sources Headers to include when compiling, includes the source files of the kernel itself
         * \param instantiation Unmangled kernel symbol
         * \param log Output compilation log
         * \param ptx Output PTX
         * \param mangled_instantiation Output mangled version of the specified instantiation
         * \returns True if the compilation was successful, otherwise False
	     */
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

            // TODO: store compiler options as a vector<const char*> (?)
			// convert compiler options to const char*
            std::vector<const char*> compiler_options_char(compiler_options.size());
            for (u64 i = 0; i < compiler_options.size(); i++) {
                compiler_options_char[i] = compiler_options[i].c_str();
            }

            // convert headers to const char*
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

                header_names_char.emplace_back(name.c_str());
                header_sources_char.emplace_back(code.c_str());
            }

            // create the NVRTC program
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

            // compile the program
            nvrtcResult result = nvrtcCompileProgram(
                nvrtc_program,
                static_cast<i32>(compiler_options_char.size()),
                compiler_options_char.data()
            );

            // extract the compilation log
            u64 log_size = 0;
            NVRTC_ASSERT(nvrtcGetProgramLogSize(nvrtc_program, &log_size));
            log.resize(log_size + 1);
            NVRTC_ASSERT(nvrtcGetProgramLog(nvrtc_program, log.data()));

            if (result != NVRTC_SUCCESS) {
                return false;
            }

            // extract the PTX string
            u64 ptx_size;
            NVRTC_ASSERT(nvrtcGetPTXSize(nvrtc_program, &ptx_size));
            ptx.resize(ptx_size + 1);
            NVRTC_ASSERT(nvrtcGetPTX(nvrtc_program, ptx.data()));

            // unmangle the symbol name
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

        /**
	     * \brief Constructs a kernel located under \a symbol.
         * \param symbol Kernel symbol
         * \param headers Headers to include during compilation
         * \param compiler_options Compiler options to use when compiling
         * \param source_file_filepath Filepath of the kernel source file 
	     */
        kernel(
            const std::string& symbol,
            const std::unordered_map<std::string, std::string>& headers,
            const std::vector<std::string>& compiler_options,
            const std::string& source_file_filepath
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

            // compile the kernel
            if(compile(
                source_file_filepath,
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

                // link files? 
                if (linker_files.empty()) {
                    // no linking required, just load the PTX
                    std::cout << "loading program data\n";
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
                    // link the PTX with the linker files
                    std::cout << "running linker\n";
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
                        static_cast<u32>(jit_options.size()),
                        jit_options.data(),
                        jit_option_values.data()
                    ));

                    CUresult cu_result;

                    for (std::string link_file : linker_files) {
                        CUjitInputType jit_input_type;
                        if (link_file == ".") {
                            // special case for linking to current executable
                            link_file = detail::get_current_executable_path();
                            jit_input_type = CU_JIT_INPUT_OBJECT;
                        }
                    	else
                        {
                            // infer from filename 
                            jit_input_type = detail::get_cuda_jit_input_type(&link_file);
                        }

                        cu_result = cuLinkAddFile(m_link_state, jit_input_type, link_file.c_str(), 0, 0, 0);
                        u64 path_count = 0;

                        while (cu_result == CUDA_ERROR_FILE_NOT_FOUND && path_count < (int)linker_paths.size()) {
                            std::string filename = detail::path_join(linker_paths[path_count++], link_file);
                            cu_result = cuLinkAddFile(m_link_state, jit_input_type, filename.c_str(), 0, 0, 0);
                        }

                        CUDA_ASSERT(cu_result);
                    }

                    size_t cubin_size;
                    void* cubin;
                    CUDA_ASSERT(cuLinkComplete(m_link_state, &cubin, &cubin_size));
                    CUDA_ASSERT(cuModuleLoadData(&m_module, cubin));
                }

                // extract the kernel 
                CUDA_ASSERT(cuModuleGetFunction(&m_kernel, m_module, mangled_instantiation.c_str()));
                std::cout << "kernel compiled successfully\n";
            }
        	else {
                throw std::runtime_error("encountered an unknown error during kernel compilation: \n" + log);
            }
        }

        /**
         * \brief Starts the kernel using the specified \a options and \a arguments.
         * \param options Kernel options struct
         * \param args Kernel arguments
         */
        inline void start_kernel(const kernel_options& options, void** args) {
            CUDA_ASSERT(cuLaunchKernel(
                m_kernel,
                options.thread_count[0],
                options.thread_count[1],
                options.thread_count[2],
                options.block_count[0],
                options.block_count[1],
                options.block_count[2],
                options.shared_memory_size,
                options.stream,
                args, 0
            ));
        }
	public:
        /**
		 * \brief Starts the kernel using the specified \a options and \a arguments.
		 * \tparam Arguments kernel arguments
		 * \param options Kernel options struct to use for kernel initialization
		 * \param args Kernel arguments
		 */
        template<class... Arguments>
        constexpr inline void start(const kernel_options& options, Arguments&&... args) {
            if constexpr(sizeof...(Arguments) > 0) {
                void* pointers[sizeof...(Arguments)] = { &args... };
                start_kernel(options, pointers);
            }
            else {
                start_kernel(options, nullptr);
            }
        }
    private:
        CUfunction m_kernel;
        CUmodule m_module;
        CUlinkState m_link_state;

        friend struct program;
    };

    /**
     * \brief Program structure; contains information about, and wraps around the CUDA driver API's module.
     */
    struct program {
    private:
        program() = default;

        /**
	     * \brief Constructs the program.
	     * \param headers Included headers
	     * \param compiler_options Used compiler options
	     * \param source_file Program source file
	     */
        program(
            const std::unordered_map<std::string, std::string>& headers,
            const std::vector<std::string>& compiler_options,
            const std::string& source_file
        ) : m_headers(headers), m_compiler_options(compiler_options), m_source_file(source_file) {}

        /**
         * \brief Compiles the CUDA module
         * \param source Program source
         * \param source_file Program filepath
         * \param compiler_options Compiler options to supply to the compiler
         * \param log Output compilation log
         * \param ptx Output PTX
         */
        static inline bool compile(
            const std::string& source,
            const std::string& source_file,
            const std::vector<std::string>& compiler_options,
            std::unordered_map<std::string, std::string>& headers,
            std::string& log,
            std::string& ptx
        ) {
            // TODO: store compiler options as a vector<const char*> (?)
            // convert compiler options to const char* 
            std::vector<const char*> compiler_options_char(compiler_options.size());
            for (u64 i = 0; i < compiler_options.size(); i++) {
                compiler_options_char[i] = compiler_options[i].c_str();
            }

            // convert headers to const char* 
            std::vector<const char*> header_names_char;
            std::vector<const char*> header_content_char;

            header_names_char.reserve(headers.size());
            header_content_char.reserve(headers.size());

            for (const auto& header : headers) {
                header_names_char.emplace_back(header.first.c_str());
                header_content_char.emplace_back(header.second.c_str());
            }

            // create an NVRTC program
            nvrtcProgram nvrtc_program;
            NVRTC_ASSERT(nvrtcCreateProgram(
                &nvrtc_program,
                source.c_str(),
                source_file.c_str(),
                static_cast<i32>(headers.size()),
                header_content_char.data(),
                header_names_char.data())
            );

            // compile the program
            nvrtcResult result = nvrtcCompileProgram(
                nvrtc_program,
                static_cast<i32>(compiler_options_char.size()),
                compiler_options_char.data()
            );

            if (result != NVRTC_SUCCESS && result != NVRTC_ERROR_COMPILATION) {
                NVRTC_ASSERT(result);
            }

            // extract the compilation log
            u64 log_size = 0;
            NVRTC_ASSERT(nvrtcGetProgramLogSize(nvrtc_program, &log_size));
            log.resize(log_size + 1);
            NVRTC_ASSERT(nvrtcGetProgramLog(nvrtc_program, log.data()));

            if (result != NVRTC_SUCCESS) {
                return false;
            }

            // extract the PTX string
            u64 ptx_size;
            NVRTC_ASSERT(nvrtcGetPTXSize(nvrtc_program, &ptx_size));
            ptx.resize(ptx_size + 1);
            NVRTC_ASSERT(nvrtcGetPTX(nvrtc_program, ptx.data()));
            return true;
        }
    public:
        /**
         * \brief Creates a new program using the provided  \a source_file.
         * \param source_file Program filepath
         * \param compiler_options Compiler options to give to the compiler (note that the following options are already used by default: --device-as-default-execution-space, --std=c++20, --dopt=on)
         */
        static inline program create(
            const std::string& source_file,
            std::vector<std::string> compiler_options = {}
        ) {
            std::unordered_map<std::string, std::string> headers;
            std::string source = detail::read_file(source_file);
            std::string log;
            std::string ptx;
            bool result;

            static const std::vector<std::string> default_compiler_options = {
                "--device-as-default-execution-space",
                "--std=c++20",
                "--dopt=on",
                "--include-path=C:\\dev\\projects\\gputil\\gputil\\include" // TODO: extract the include path from the visual studio solution file (?)
            };

            // insert default compiler arguments
            compiler_options.insert(compiler_options.end(), default_compiler_options.begin(), default_compiler_options.end());

            // std::vector<std::string> include_directories;
            // extract_include_directories_from_compiler_options(compiler_options, include_directories);

            // extract the icluded headers
            std::cout << "\nadding headers:\n";
            while((result = compile(source, source_file, compiler_options, headers, log, ptx)) == false) {
                std::string header_location;

                if (!extract_include_info_from_compile_error(log, header_location)) {
                    break; // not a header-related error
                }

                // header already loaded. Something is wrong
                if (headers.count(header_location) > 0) {
                    break;
                }

                // load missing header file
                std::string header_content;

                // check if the file is jitsafe
                if (detail::s_jitsafe_headers.contains(header_location)) {
                    header_content = detail::s_jitsafe_headers.at(header_location);
                }
            	else {
                    // check the immediate fs
                    if(std::filesystem::exists(header_location)) {
                        header_content = detail::read_file(header_location);
                    }
                    // check inside of include directories
                    else {
						// std::cout << "checking include directories for file: " + header_location + "\n";
                           
	                    // for(const std::string& include_dir : include_directories) {
						//    std::string included_header_location = detail::path_join(include_dir, header_location);
                           
	                    //    std::cout << included_header_location << '\n';
                           
	                    //    if(std::filesystem::exists(header_location)) {
	                    //        header_content = detail::read_file(header_location);
	                    //        header_location = included_header_location;
	                    //    }
	                    // }
                    }
                }

                // we've found the desired header, add it to the list of headers
                if(header_content.empty() == false) {
                    std::cout << header_location << '\n';
                    headers.emplace(std::move(header_location), std::move(header_content));
                }
                else {
                    ASSERT(false, std::string("cannot locate header file '" + header_location + "'").c_str());
                    return {};
                }
            }

            if(result) {
                std::cout << "\nprogram compiled successfully\n";
                headers.emplace(source_file, std::move(source));
                return { headers, compiler_options, source_file };
            }
            else {
                ASSERT(false, log.c_str());
                return {};
            }
        }

        /**
         * \brief Creates a new kernel structure using the compiled program.
         * \param symbol Kernel symbol.
         * \returns Kernel located under \a symbol
         */
        inline kernel get_kernel(const std::string& symbol) const {
            return { symbol, m_headers, m_compiler_options, m_source_file };
        }
    private:
        std::unordered_map<std::string, std::string> m_headers;
        std::vector<std::string> m_compiler_options;
        std::string m_source_file;
    };
}