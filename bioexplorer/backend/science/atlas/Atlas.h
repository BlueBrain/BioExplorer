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

#pragma once

#include <science/common/SDFGeometries.h>

namespace bioexplorer
{
namespace atlas
{
/**
 * Load brain atlas from database
 */
class Atlas : public common::SDFGeometries
{
public:
    /**
     * @brief Construct a new Vasculature object
     *
     * @param scene 3D scene into which the vasculature should be loaded
     * @param details Set of attributes defining how the vasculature should be
     * loaded
     */
    Atlas(core::Scene& scene, const details::AtlasDetails& details, const core::Vector3d& position,
          const core::Quaterniond& rotation, const core::LoaderProgress& callback = core::LoaderProgress());

private:
    double _getDisplacementValue(const DisplacementElement& element) final { return 0; }

    void _buildModel(const core::LoaderProgress& callback);

    const details::AtlasDetails _details;
    core::Scene& _scene;
};
} // namespace atlas
} // namespace bioexplorer
