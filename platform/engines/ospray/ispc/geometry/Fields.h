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

#include <ospray/SDK/geometry/Geometry.h>

namespace core
{
namespace engine
{
namespace ospray
{
struct Fields : public ::ospray::Geometry
{
public:
    Fields();

    std::string toString() const final { return ("field"); }
    void finalize(::ospray::Model* model) final;
    void commit() final;

protected:
    ::ospray::Ref<::ospray::Data> _indices;
    ::ospray::Ref<::ospray::Data> _values;
    int _dataType;
    ::ospray::vec3i _dimensions;
    ::ospray::vec3f _spacing;
    ::ospray::vec3f _offset;
    float _distance;
    float _cutoff;
    float _gradientOffset;
    bool _gradientShadingEnabled;
    bool _useOctree;
    float _samplingRate;
    float _epsilon;
    int _accumulationSteps;
    int _accumulationCount;
};
} // namespace ospray
} // namespace engine
} // namespace core
