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

#include "BlackHoleRenderer.h"

#include <plugin/common/Properties.h>

#include <platform/core/common/Properties.h>

#include "BlackHoleRenderer_ispc.h"

using namespace ospray;
using namespace core;

namespace spaceexplorer
{
namespace blackhole
{
void BlackHoleRenderer::commit()
{
    AbstractRenderer::commit();

    _grid = getParam(BLACK_HOLE_RENDERER_PROPERTY_DISPLAY_GRID.name.c_str(), BLACK_HOLE_DEFAULT_RENDERER_DISPLAY_GRID);
    _nbDisks = getParam1i(BLACK_HOLE_RENDERER_PROPERTY_NB_DISKS.name.c_str(), BLACK_HOLE_DEFAULT_RENDERER_NB_DISKS);
    _diskRotationSpeed = getParam1f(BLACK_HOLE_RENDERER_PROPERTY_DISK_ROTATION_SPEED.name.c_str(),
                                    BLACK_HOLE_DEFAULT_RENDERER_DISK_ROTATION_SPEED);
    _diskTextureLayers = getParam1i(BLACK_HOLE_RENDERER_PROPERTY_DISK_TEXTURE_LAYERS.name.c_str(),
                                    BLACK_HOLE_DEFAULT_RENDERER_TEXTURE_LAYERS);
    _blackHoleSize = getParam1f(BLACK_HOLE_RENDERER_PROPERTY_SIZE.name.c_str(), BLACK_HOLE_DEFAULT_RENDERER_SIZE);

    ::ispc::BlackHoleRenderer_set(getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr), _timestamp, spp, _exposure,
                                  _nbDisks, _grid, _diskRotationSpeed, _diskTextureLayers, _blackHoleSize,
                                  _anaglyphEnabled, (ispc::vec3f&)_anaglyphIpdOffset);
}

BlackHoleRenderer::BlackHoleRenderer()
{
    ispcEquivalent = ::ispc::BlackHoleRenderer_create(this);
}

OSP_REGISTER_RENDERER(BlackHoleRenderer, blackhole);
OSP_REGISTER_MATERIAL(blackhole, core::engine::ospray::AdvancedMaterial, default);

} // namespace blackhole
} // namespace spaceexplorer