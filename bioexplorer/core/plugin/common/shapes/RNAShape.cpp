/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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

RNAShape::RNAShape(const Vector4ds& clippingPlanes,
                   const RNAShapeType& shapeType, const uint64_t nbElements,
                   const Vector2f& shapeParams, const Vector2f& valuesRange,
                   const Vector3d& curveParams)
    : Shape(clippingPlanes)
    , _shapeType(shapeType)
    , _shapeParams(shapeParams)
    , _valuesRange(valuesRange)
    , _curveParams(curveParams)
{
    _bounds.merge(Vector3d(-curveParams.x / 2.0, -curveParams.y / 2.0,
                           -curveParams.z / 2.0));
    _bounds.merge(Vector3d(curveParams.x / 2.0, curveParams.y / 2.0,
                           curveParams.z / 2.0));

    _U = Vector3d(valuesRange.x, valuesRange.y, nbElements);
    _V = Vector3d(valuesRange.x, valuesRange.y, nbElements);

    switch (_shapeType)
    {
    case RNAShapeType::moebius:
        _U = {2.0 * M_PI, 4.0 * M_PI, nbElements};
        _V = {-0.4, 0.4, 1.0};
        break;
    case RNAShapeType::heart:
        _U = {0.0, 2.0 * M_PI, nbElements};
        _V = {0.0, 1.0, 1.0};
        break;
    default:
        break;
    }

    _step = (_U.y - _U.x) / _U.z;
}

Transformation RNAShape::getTransformation(
    const uint64_t occurrence, const uint64_t nbOccurrences,
    const MolecularSystemAnimationDetails& MolecularSystemAnimationDetails,
    const double offset) const
{
    const double u = _valuesRange.x + _step * occurrence;
    const double v = _valuesRange.x + _step * (occurrence + 1);

    Vector3d src, dst;
    _getSegment(u, v, src, dst);

    const Vector3d direction = normalize(dst - src);
    const Vector3d normal = cross(UP_VECTOR, direction);
    double upOffset = 0.0;
    if (MolecularSystemAnimationDetails.positionSeed != 0)
        upOffset =
            MolecularSystemAnimationDetails.positionStrength *
            rnd3(MolecularSystemAnimationDetails.positionSeed + occurrence);

    Vector3d pos = src + normal * (offset + upOffset);

    if (isClipped(pos, _clippingPlanes))
        throw std::runtime_error("Instance is clipped");

    Quaterniond rot = safeQuatlookAt(normal);
    if (MolecularSystemAnimationDetails.rotationSeed != 0)
        rot = weightedRandomRotation(
            rot, MolecularSystemAnimationDetails.rotationSeed, occurrence,
            MolecularSystemAnimationDetails.rotationStrength);

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(rot);
    return transformation;
}

bool RNAShape::isInside(const Vector3d& point) const
{
    PLUGIN_THROW("isInside is not implemented for Parametric shapes");
}

void RNAShape::_getSegment(const double u, const double v, Vector3d& src,
                           Vector3d& dst) const
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

Vector3d RNAShape::_trefoilKnot(const double t) const
{
    return {_shapeParams.x * ((sin(t) + 2.0 * sin(_curveParams.x * t))) / 3.0,
            _shapeParams.x * ((cos(t) - 2.0 * cos(_curveParams.y * t))) / 3.0,
            _shapeParams.x * (-sin(_curveParams.z * t))};
}

Vector3d RNAShape::_torus(const double t) const
{
    return {_shapeParams.x *
                (cos(t) + _curveParams.x * cos(_curveParams.y * t) * cos(t)),
            _shapeParams.x *
                (sin(t) + _curveParams.x * cos(_curveParams.y * t) * sin(t)),
            _shapeParams.x * _curveParams.x * sin(_curveParams.y * t)};
}

Vector3d RNAShape::_star(const double t) const
{
    return {_shapeParams.x * (2.0 * sin(3.0 * t) * cos(t)),
            _shapeParams.x * (2.0 * sin(3.0 * t) * sin(t)),
            _shapeParams.x * sin(3.0 * t)};
}

Vector3d RNAShape::_spring(const double t) const
{
    return {_shapeParams.x * cos(t) +
                (+_curveParams.x * cos(_curveParams.y * t)) * cos(t),
            _shapeParams.x * sin(t) +
                (+_curveParams.x * cos(_curveParams.y * t)) * sin(t),
            _shapeParams.x * t + _curveParams.x * sin(_curveParams.y * t)};
}

Vector3d RNAShape::_heart(const double u) const
{
    return {_shapeParams.x * 4.0 * pow(sin(u), 3.0),
            _shapeParams.x * 0.25 *
                (13.0 * cos(u) - 5.0 * cos(2.0 * u) - 2.0 * cos(3.0 * u) -
                 cos(4.0 * u)),
            0.0};
}

Vector3d RNAShape::_thing(const double t) const
{
    return {_shapeParams.x *
                (sin(t) + _curveParams.x * sin(_curveParams.y * t)),
            _shapeParams.x *
                (cos(t) - _curveParams.x * cos(_curveParams.y * t)),
            _shapeParams.x * (-sin(_curveParams.z * t))};
}

Vector3d RNAShape::_moebius(const double u, const double v) const
{
    return {4.0 * _shapeParams.x * (cos(u) + v * cos(u / 2.0) * cos(u)),
            4.0 * _shapeParams.x * (sin(u) + v * cos(u / 2.0) * sin(u)),
            8.0 * _shapeParams.x * (v * sin(u / 2.0))};
}

} // namespace common
} // namespace bioexplorer
