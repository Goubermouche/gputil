#pragma once
#include "global/types.h"

struct my_struct {
	gputil::i32 value;

	gputil::i32 get_value() const {
		return value;
	}
};