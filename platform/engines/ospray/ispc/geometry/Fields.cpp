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

#include "Fields.h"
#include "Fields_ispc.h"

#include <platform/core/common/Properties.h>
#include <platform/engines/ospray/OSPRayProperties.h>

#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/common/Model.h>
#include <ospray/SDK/transferFunction/TransferFunction.h>

namespace core
{
namespace engine
{
namespace ospray
{
Fields::Fields()
{
    this->ispcEquivalent = ispc::Field_create(this);
}

void Fields::finalize(::ospray::Model *model)
{
    const size_t numFields = 1;
    _indices = getParamData(OSPRAY_GEOMETRY_PROPERTY_FIELD_INDICES, nullptr);
    _values = getParamData(OSPRAY_GEOMETRY_PROPERTY_FIELD_VALUES, nullptr);
    _dataType = getParam1i(OSPRAY_GEOMETRY_PROPERTY_FIELD_DATATYPE, 0);

    _dimensions = getParam3i(OSPRAY_GEOMETRY_PROPERTY_FIELD_DIMENSIONS, ::ospray::vec3i());
    _spacing = getParam3f(OSPRAY_GEOMETRY_PROPERTY_FIELD_SPACING, ::ospray::vec3f());
    _offset = getParam3f(OSPRAY_GEOMETRY_PROPERTY_FIELD_OFFSET, ::ospray::vec3f());
    _accumulationSteps = getParam1i(OSPRAY_GEOMETRY_PROPERTY_FIELD_ACCUMULATION_STEPS, 0);

    ::ispc::Field_set(getIE(), model->getIE(), (ispc::vec3i &)_dimensions, (ispc::vec3f &)_spacing,
                      (ispc::vec3f &)_offset, _indices->data, _values->data, _dataType, numFields);

    // Transfer function
    ::ospray::TransferFunction *transferFunction =
        (::ospray::TransferFunction *)getParamObject(DEFAULT_COMMON_TRANSFER_FUNCTION, nullptr);
    if (transferFunction)
        ::ispc::Field_setTransferFunction(getIE(), transferFunction->getIE());
    commit();
}

void Fields::commit()
{
    _distance = getParamf(OSPRAY_FIELD_PROPERTY_DISTANCE, 1.f);
    _cutoff = getParamf(OSPRAY_FIELD_PROPERTY_CUTOFF, 1500.f);
    _gradientOffset = getParamf(OSPRAY_FIELD_PROPERTY_GRADIENT_OFFSET, 1e-6f);
    _gradientShadingEnabled = getParam1i(OSPRAY_FIELD_PROPERTY_GRADIENT_SHADING_ENABLED, false);
    _useOctree = getParam1i(OSPRAY_FIELD_PROPERTY_USE_OCTREE, true);
    _samplingRate = getParamf(OSPRAY_FIELD_PROPERTY_SAMPLING_RATE, 1.f);
    _epsilon = getParamf(OSPRAY_FIELD_PROPERTY_EPSILON, 1e-6f);
    _accumulationSteps = getParam1i(OSPRAY_GEOMETRY_PROPERTY_FIELD_ACCUMULATION_STEPS, 0);
    _accumulationCount = getParam1i(OSPRAY_GEOMETRY_PROPERTY_FIELD_ACCUMULATION_COUNT, 0);

    ::ispc::Field_commit(getIE(), _distance, _cutoff, _gradientOffset, _gradientShadingEnabled, _useOctree,
                         _samplingRate, _epsilon, _accumulationSteps, _accumulationCount);
}

OSP_REGISTER_GEOMETRY(Fields, fields);
} // namespace ospray
} // namespace engine
} // namespace core
