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

#include <memory>

#include <optixu/optixpp_namespace.h>

#include "OptiXCameraProgram.h"

namespace core
{
namespace engine
{
namespace optix
{
class OptiXOrthographicCamera : public OptiXCameraProgram
{
public:
    OptiXOrthographicCamera();
    ~OptiXOrthographicCamera() final = default;

    void commit(const OptiXCamera& camera, ::optix::Context context) final;
};
} // namespace optix
} // namespace engine
} // namespace core