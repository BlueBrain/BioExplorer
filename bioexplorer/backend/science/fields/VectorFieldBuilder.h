/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

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

#include "FieldBuilder.h"

#include <platform/core/common/Types.h>
#include <platform/core/engineapi/Scene.h>

namespace bioexplorer
{
namespace fields
{
/**
 * @brief The VectorFieldBuilder class handles electro-magnetic fields data
 * structures
 */
class VectorFieldBuilder : public FieldBuilder
{
public:
    /**
     * @brief Default constructor
     */
    VectorFieldBuilder();

    void buildOctree(core::Engine& engine, core::Model& model, const double voxelSize, const double density,
                     const uint32_ts& modelIds) final;
};
} // namespace fields
} // namespace bioexplorer
