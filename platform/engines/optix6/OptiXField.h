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

#include <platform/core/common/CommonTypes.h>
#include <platform/core/common/transferFunction/TransferFunction.h>
#include <platform/core/engineapi/Field.h>

#include "OptiXModel.h"
#include "OptiXTypes.h"

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

namespace core
{
namespace engine
{
namespace optix
{
class OptiXField : public Field
{
public:
    /** @copydoc Volume::Volume */
    OptiXField(const FieldParameters& parameters, const Vector3ui& dimensions, const Vector3f& spacing,
               const Vector3f& offset, const uint32_ts& indices, const floats& values, const OctreeDataType dataType)
        : Field(parameters, dimensions, spacing, offset, indices, values, dataType)
    {
    }
};
} // namespace optix
} // namespace engine
} // namespace core