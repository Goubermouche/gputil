#pragma once

#include "global_base.h"

namespace gputil {
	template<class T>
	struct vector : public gputil_base {
		T value;
	};
}
