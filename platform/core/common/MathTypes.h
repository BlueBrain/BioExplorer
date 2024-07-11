/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

#define GLM_FORCE_CTOR_INIT
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext.hpp>
#include <glm/gtx/io.hpp>
#include <vector>

#include <platform/core/common/BaseObject.h>
#include <platform/core/common/Macros.h>

namespace core
{
template <typename T>
class Box;
}
namespace staticjson
{
class ObjectHandler;
template <typename U>
void init(core::Box<U>*, ObjectHandler*);
} // namespace staticjson

namespace core
{
template <class T>
class Box
{
public:
    using vec = glm::vec<3, T>;

    Box() = default;

    Box(const vec& pMin, const vec& pMax)
        : _min(glm::min(pMin, pMax))
        , _max(glm::max(pMin, pMax))
    {
    }
    inline bool operator==(const Box<T>& other) const { return _min == other._min && _max == other._max; }

    inline void merge(const Box<T>& aabb)
    {
        _min = glm::min(_min, aabb.getMin());
        _max = glm::max(_max, aabb.getMax());
    }

    inline void merge(const vec& point)
    {
        _min = glm::min(_min, point);
        _max = glm::max(_max, point);
    }

    inline void intersect(const Box<T>& aabb)
    {
        _min = glm::max(_min, aabb.getMin());
        _max = glm::min(_max, aabb.getMax());
    }

    inline void reset()
    {
        _min = vec(std::numeric_limits<T>::max());
        _max = vec(-std::numeric_limits<T>::max());
    }

    inline bool isEmpty() const { return _min.x >= _max.x || _min.y >= _max.y || _min.z >= _max.x; }

    inline vec getCenter() const { return (_min + _max) * .5; }
    inline vec getSize() const { return _max - _min; }
    inline const vec& getMin() const { return _min; }
    inline const vec& getMax() const { return _max; }

#ifdef __INTEL_COMPILER // Workaround for ICC. Make members public
public:
    vec _min{std::numeric_limits<T>::max()};
    vec _max{-std::numeric_limits<T>::max()};
#else
private:
    vec _min{std::numeric_limits<T>::max()};
    vec _max{-std::numeric_limits<T>::max()};

    SERIALIZATION_FRIEND(Box<double>)
    SERIALIZATION_FRIEND(Box<float>)
#endif
};

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const Box<T>& aabb)
{
    return os << aabb.getMin() << " - " << aabb.getMax();
}

/**
 * AABB definitions
 */
using Boxf = Box<float>;
using Boxd = Box<double>;

/**
 * Matrix definitions
 */
using Matrix4d = glm::mat<4, 4, double>;
using Matrix4f = glm::mat4;

/**
 * Vector definitions
 */
using Vector2i = glm::vec<2, int32_t>;
using Vector3i = glm::vec<3, int32_t>;

using Vector2ui = glm::vec<2, uint32_t>;
using Vector3ui = glm::vec<3, uint32_t>;

using Vector2f = glm::vec2;
using Vector3f = glm::vec3;
using Vector4f = glm::vec4;
typedef std::vector<Vector3f> Vector3fs;
typedef std::vector<Vector4f> Vector4fs;

using Vector2d = glm::vec<2, double>;
using Vector3d = glm::vec<3, double>;
using Vector4d = glm::vec<4, double>;
typedef std::vector<Vector2d> Vector2ds;

// Consts
const Vector3d UP_VECTOR = {0.0, 1.0, 0.0};

/**
 * Quaternion definitions
 */
using Quaterniond = glm::tquat<double, glm::highp>; //!< Double quaternion.

inline Quaterniond safeQuatlookAt(const Vector3d& v)
{
    const Vector3d vector = glm::normalize(v);
    Vector3d upVector = UP_VECTOR;
    if (glm::abs(glm::dot(vector, upVector)) > 0.999)
        // Gimble lock
        upVector = Vector3d(0.0, 0.0, 1.0);
    return quatLookAtRH(vector, upVector);
}
} // namespace core
