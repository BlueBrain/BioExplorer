/*
    Copyright 2019 - 2024 Blue Brain Project / EPFL

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

#include <optixu/optixpp_namespace.h>

#include "OptiXUtils.h"

namespace core
{
namespace engine
{
namespace optix
{
class OptiXCamera;

/**
 * @brief The OptiXCameraProgram class is an abstract class that provides the
 * required programs for launching rays from a camera
 */
class OptiXCameraProgram
{
public:
    virtual ~OptiXCameraProgram()
    {
        RT_DESTROY(_rayGenerationProgram);
        RT_DESTROY(_missProgram);
        RT_DESTROY(_exceptionProgram);
    }

    ::optix::Program getRayGenerationProgram() { return _rayGenerationProgram; }
    ::optix::Program getMissProgram() { return _missProgram; }
    ::optix::Program getExceptionProgram() { return _exceptionProgram; }
    /**
     * @brief commit Virtual method for committing camera specific variables to the context
     * @param camera The main core camera
     * @param context The OptiX context
     */
    virtual void commit(const OptiXCamera& camera, ::optix::Context context) = 0;

protected:
    ::optix::Program _rayGenerationProgram{nullptr};
    ::optix::Program _missProgram{nullptr};
    ::optix::Program _exceptionProgram{nullptr};
};
} // namespace optix
} // namespace engine
} // namespace core