/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2021 Blue BrainProject / EPFL
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "RNAShape.h"

#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>

namespace bioexplorer
{
namespace common
{
using namespace brayns;
using namespace details;

RNAShape::RNAShape(const Vector4fs& clippingPlanes,
                   const RNAShapeType& shapeType, const uint64_t nbElements,
                   const Vector2f& shapeParams, const Vector2f& valuesRange,
                   const Vector3f& curveParams)
    : Shape(clippingPlanes)
    , _shapeType(shapeType)
    , _shapeParams(shapeParams)
    , _valuesRange(valuesRange)
    , _curveParams(curveParams)
{
    _bounds.merge(Vector3f(-curveParams.x / 2.f, -curveParams.y / 2.f,
                           -curveParams.z / 2.f));
    _bounds.merge(Vector3f(curveParams.x / 2.f, curveParams.y / 2.f,
                           curveParams.z / 2.f));

    _U = Vector3f(valuesRange.x, valuesRange.y, nbElements);
    _V = Vector3f(valuesRange.x, valuesRange.y, nbElements);

    switch (_shapeType)
    {
    case RNAShapeType::moebius:
        _U = {2.f * M_PI, 4.f * M_PI, nbElements};
        _V = {-0.4f, 0.4f, 1.f};
        break;
    case RNAShapeType::heart:
        _U = {0.f, 2.f * M_PI, nbElements};
        _V = {0.f, 1.f, 1.f};
        break;
    default:
        break;
    }

    _step = (_U.y - _U.x) / _U.z;
}

Transformation RNAShape::getTransformation(
    const uint64_t occurence, const uint64_t nbOccurences,
    const AnimationDetails& animationDetails, const float offset) const
{
    const float u = _valuesRange.x + _step * occurence;
    const float v = _valuesRange.x + _step * (occurence + 1);

    Vector3f src, dst;
    _getSegment(u, v, src, dst);

    const Vector3f direction = normalize(dst - src);
    const Vector3f normal = cross(UP_VECTOR, direction);
    float upOffset = 0.f;
    if (animationDetails.positionSeed != 0)
        upOffset = animationDetails.positionStrength *
                   rnd3(animationDetails.positionSeed + occurence);

    Vector3f pos = src + normal * (offset + upOffset);

    if (isClipped(pos, _clippingPlanes))
        throw std::runtime_error("Instance is clipped");

    Quaterniond rot = glm::quatLookAt(normal, UP_VECTOR);
    if (animationDetails.rotationSeed != 0)
        rot = weightedRandomRotation(rot, animationDetails.rotationSeed,
                                     occurence,
                                     animationDetails.rotationStrength);

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(rot);
    return transformation;
}

bool RNAShape::isInside(const Vector3f& point) const
{
    PLUGIN_THROW("isInside is not implemented for Parametric shapes");
}

void RNAShape::_getSegment(const float u, const float v, Vector3f& src,
                           Vector3f& dst) const
{
    switch (_shapeType)
    {
    case RNAShapeType::moebius:
    {
        src = _moebius(u, v);
        dst = _moebius(u + _step, v);
        break;
    }
    case RNAShapeType::torus:
    {
        src = _torus(u);
        dst = _torus(u + _step);
        break;
    }
    case RNAShapeType::star:
    {
        src = _star(u);
        dst = _star(u + _step);
        break;
    }
    case RNAShapeType::spring:
    {
        src = _spring(u);
        dst = _spring(u + _step);
        break;
    }
    case RNAShapeType::trefoilKnot:
    {
        src = _trefoilKnot(u);
        dst = _trefoilKnot(u + _step);
        break;
    }
    case RNAShapeType::heart:
    {
        src = _heart(u);
        dst = _heart(u + _step);
        break;
    }
    case RNAShapeType::thing:
    {
        src = _thing(u);
        dst = _thing(u + _step);
        break;
    }
    default:
        PLUGIN_THROW("Undefined shape");
        break;
    }
}

Vector3f RNAShape::_trefoilKnot(const float t) const
{
    return {_shapeParams.x * ((sin(t) + 2.f * sin(_curveParams.x * t))) / 3.f,
            _shapeParams.x * ((cos(t) - 2.f * cos(_curveParams.y * t))) / 3.f,
            _shapeParams.x * (-sin(_curveParams.z * t))};
}

Vector3f RNAShape::_torus(const float t) const
{
    return {_shapeParams.x *
                (cos(t) + _curveParams.x * cos(_curveParams.y * t) * cos(t)),
            _shapeParams.x *
                (sin(t) + _curveParams.x * cos(_curveParams.y * t) * sin(t)),
            _shapeParams.x * _curveParams.x * sin(_curveParams.y * t)};
}

Vector3f RNAShape::_star(const float t) const
{
    return {_shapeParams.x * (2.f * sin(3.f * t) * cos(t)),
            _shapeParams.x * (2.f * sin(3.f * t) * sin(t)),
            _shapeParams.x * sin(3.f * t)};
}

Vector3f RNAShape::_spring(const float t) const
{
    return {_shapeParams.x * cos(t) +
                (+_curveParams.x * cos(_curveParams.y * t)) * cos(t),
            _shapeParams.x * sin(t) +
                (+_curveParams.x * cos(_curveParams.y * t)) * sin(t),
            _shapeParams.x * t + _curveParams.x * sin(_curveParams.y * t)};
}

Vector3f RNAShape::_heart(const float u) const
{
    return {_shapeParams.x * 4.f * pow(sin(u), 3.f),
            _shapeParams.x * 0.25f *
                (13.f * cos(u) - 5.f * cos(2.f * u) - 2.f * cos(3.f * u) -
                 cos(4.f * u)),
            0.f};
}

Vector3f RNAShape::_thing(const float t) const
{
    return {_shapeParams.x *
                (sin(t) + _curveParams.x * sin(_curveParams.y * t)),
            _shapeParams.x *
                (cos(t) - _curveParams.x * cos(_curveParams.y * t)),
            _shapeParams.x * (-sin(_curveParams.z * t))};
}

Vector3f RNAShape::_moebius(const float u, const float v) const
{
    return {4.f * _shapeParams.x * (cos(u) + v * cos(u / 2.f) * cos(u)),
            4.f * _shapeParams.x * (sin(u) + v * cos(u / 2.f) * sin(u)),
            8.f * _shapeParams.x * (v * sin(u / 2.f))};
}

} // namespace common
} // namespace bioexplorer
