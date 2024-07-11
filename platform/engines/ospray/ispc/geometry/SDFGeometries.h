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

#include "ospray/SDK/geometry/Geometry.h"

namespace core
{
namespace engine
{
namespace ospray
{
struct SDFGeometries : public ::ospray::Geometry
{
    std::string toString() const final { return "SDFGeometries"; }
    void finalize(::ospray::Model* model) final;

    ::ospray::Ref<::ospray::Data> data;
    ::ospray::Ref<::ospray::Data> neighbours;
    ::ospray::Ref<::ospray::Data> geometries;
    float epsilon;
    ::ospray::uint64 nbMarchIterations;
    float blendFactor;
    float blendLerpFactor;
    float omega;
    float distance;

    SDFGeometries();

private:
    std::vector<void*> ispcMaterials_;
};
} // namespace ospray
} // namespace engine
} // namespace core