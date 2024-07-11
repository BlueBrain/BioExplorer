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

#include "Cones.h"
#include "Cones_ispc.h"

#include <platform/core/common/geometry/Cone.h>
#include <platform/engines/ospray/OSPRayProperties.h>

#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/common/Model.h>

#include <climits>

using namespace core;

namespace core
{
namespace engine
{
namespace ospray
{
Cones::Cones()
{
    this->ispcEquivalent = ::ispc::Cones_create(this);
}

void Cones::finalize(::ospray::Model* model)
{
    data = getParamData(OSPRAY_GEOMETRY_PROPERTY_CONES, nullptr);
    constexpr size_t bytesPerCone = sizeof(Cone);

    if (data.ptr == nullptr || bytesPerCone == 0)
        throw std::runtime_error("#ospray:geometry/cones: no 'cones' data specified");

    const size_t numCones = data->numBytes / bytesPerCone;
    ::ispc::ConesGeometry_set(getIE(), model->getIE(), data->data, numCones);
}

OSP_REGISTER_GEOMETRY(Cones, cones);
} // namespace ospray
} // namespace engine
} // namespace core
