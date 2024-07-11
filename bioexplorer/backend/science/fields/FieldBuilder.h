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

#include <platform/core/common/simulation/AbstractAnimationHandler.h>

#include <platform/core/common/Types.h>

namespace bioexplorer
{
namespace fields
{
/**
 * @brief The FieldBuilder class handles electro-magnetic fields data
 * structures
 */
class FieldBuilder
{
public:
    /**
     * @brief Default constructor
     */
    FieldBuilder() {}

    virtual void buildOctree(core::Engine& engine, core::Model& model, const double voxelSize, const double density,
                             const uint32_ts& modelIds) = 0;
};
} // namespace fields
} // namespace bioexplorer
