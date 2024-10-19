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

#include <science/common/Types.h>

namespace bioexplorer
{
namespace io
{
namespace filesystem
{
/**
 * Load molecular systems from an optimized binary representation of the 3D
 * scene
 */
class USDExporter
{
public:
    /**
     * @brief Construct a new object
     */
    USDExporter(const core::Scene& scene)
        : _scene(scene)
    {
    }

    /**
     * @brief Export scene to USD file
     *
     * @param filename Full path of the morphology file
     */
    void exportToFile(const std::string& filename) const;

private:
    const core::Scene& _scene;
};
} // namespace filesystem
} // namespace io
} // namespace bioexplorer
