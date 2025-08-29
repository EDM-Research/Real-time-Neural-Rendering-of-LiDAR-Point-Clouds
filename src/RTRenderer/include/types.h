#pragma once

#include "glm/detail/qualifier.hpp"
#include "glm/glm.hpp"

namespace glm {
typedef vec<3, unsigned char, defaultp> ucvec3;
typedef vec<4, unsigned char, defaultp> ucvec4;

}  // namespace glm

struct PaddedDVec3 {
  glm::dvec3 v;
  double pad_val;
};

struct PaddedUCVec3 {
  glm::ucvec3 v;
  unsigned char pad_val;
};

struct PCData {
  glm::dvec3 point;
  glm::ucvec3 color;
};