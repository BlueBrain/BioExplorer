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

#include "Light.h"

#include <platform/core/common/utils/Utils.h>

namespace core
{
Light::Light(LightType type, const Vector3d& color, double intensity, bool isVisible)
    : _type(type)
    , _color(color)
    , _intensity(intensity)
    , _isVisible(isVisible)
{
}

DirectionalLight::DirectionalLight(const Vector3d& direction, double angularDiameter, const Vector3d& color,
                                   double intensity, bool isVisible)
    : Light(LightType::DIRECTIONAL, color, intensity, isVisible)
    , _direction(direction)
    , _angularDiameter(angularDiameter)
{
}

SphereLight::SphereLight(const Vector3d& position, double radius, const Vector3d& color, double intensity,
                         bool isVisible)
    : Light(LightType::SPHERE, color, intensity, isVisible)
    , _position(position)
    , _radius(radius)
{
}

QuadLight::QuadLight(const Vector3d& position, const Vector3d& edge1, const Vector3d& edge2, const Vector3d& color,
                     double intensity, bool isVisible)
    : Light(LightType::QUAD, color, intensity, isVisible)
    , _position(position)
    , _edge1(edge1)
    , _edge2(edge2)
{
}

SpotLight::SpotLight(const Vector3d& position, const Vector3d& direction, const double openingAngle,
                     const double penumbraAngle, const double radius, const Vector3d& color, double intensity,
                     bool isVisible)
    : Light(LightType::SPOTLIGHT, color, intensity, isVisible)
    , _position(position)
    , _direction(direction)
    , _openingAngle(openingAngle)
    , _penumbraAngle(penumbraAngle)
    , _radius(radius)
{
}

AmbientLight::AmbientLight(const Vector3d& color, double intensity, bool isVisible)
    : Light(LightType::AMBIENT, color, intensity, isVisible)
{
}

} // namespace core
