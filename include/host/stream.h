#pragma once

namespace gputil {
    struct stream {
        stream() = default;

        constexpr inline operator CUstream() const {
            return m_stream;
        }
    private:
        CUstream m_stream = NULL;

        friend struct kernel;
    };
}
