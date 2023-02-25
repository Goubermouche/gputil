#pragma once

#include "utility.h"

namespace gputil {
    struct program {
    public:
        program() = default;

        static inline program create(const std::string& source_file);
    private:
        program(nvrtcProgram program) : m_program(program) {
	        
        }

    private:
        nvrtcProgram m_program;
    };

    inline program program::create(const std::string& source_file) {
        std::ifstream file(source_file);
        // Note: We have to keep the parentheses in the first argument, otherwise we can't compile.
        std::string program_string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()); 

        // Extract #includes from the given file
        //const std::regex include_regex("#include\\s+[\"<]([^\">]+)[\">]");
        //std::vector<std::string> program_includes = extract_regex_from_string(program_string, include_regex);

        //for(const std::string include : program_includes) {
        //    std::cout << include << '\n';
        //}

        nvrtcProgram nvrtc_program = {};
        NVRTC_ASSERT(nvrtcCreateProgram(
            &nvrtc_program,
            program_string.c_str(),
            source_file.c_str(),
            0,
            0,
            0
        ));

        const char* opts[] = { "--fmad=false" };
        const nvrtc_result compilation_status = nvrtcCompileProgram(
            nvrtc_program,
            1,
            opts
        );

        u64 log_size;
        NVRTC_ASSERT(nvrtcGetProgramLogSize(nvrtc_program, &log_size));
        char* log = new char[log_size];
        NVRTC_ASSERT(nvrtcGetProgramLog(nvrtc_program, log));
        std::cout << log << '\n';
        delete[] log;

        return program(nvrtc_program);
    }
}