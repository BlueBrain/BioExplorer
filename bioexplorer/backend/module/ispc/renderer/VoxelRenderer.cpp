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
#include "VoxelRenderer.h"

#include <science/common/Properties.h>

#include <ospray/SDK/common/Data.h>

#include "VoxelRenderer_ispc.h"

namespace bioexplorer
{
namespace rendering
{
void VoxelRenderer::commit()
{
    SimulationRenderer::commit();

    _simulationThreshold = getParam1f(BIOEXPLORER_RENDERER_PROPERTY_VOXEL_SIMULATION_THRESHOLD.name.c_str(),
                                      BIOEXPLORER_DEFAULT_RENDERER_VOXEL_SIMULATION_THRESHOLD);

    ::ispc::VoxelRenderer_set(getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr), spp,
                              (_userData ? (float*)_userData->data : nullptr), _simulationDataSize, _alphaCorrection,
                              _simulationThreshold, _exposure, _anaglyphEnabled, (ispc::vec3f&)_anaglyphIpdOffset);
}

VoxelRenderer::VoxelRenderer()
{
    ispcEquivalent = ::ispc::VoxelRenderer_create(this);
}

OSP_REGISTER_RENDERER(VoxelRenderer, bio_explorer_voxel);
OSP_REGISTER_MATERIAL(bio_explorer_voxel, core::engine::ospray::AdvancedMaterial, default);

} // namespace rendering
} // namespace bioexplorer
