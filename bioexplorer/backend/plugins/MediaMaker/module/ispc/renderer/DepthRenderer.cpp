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

#include "DepthRenderer.h"

#include <plugin/common/Properties.h>

#include <platform/engines/ospray/ispc/render/utils/AdvancedMaterial.h>

#include <ospray/SDK/lights/Light.h>

#include "DepthRenderer_ispc.h"

namespace bioexplorer
{
namespace mediamaker
{
namespace rendering
{
void DepthRenderer::commit()
{
    Renderer::commit();

    _infinity = getParam1f(MEDIA_MAKER_RENDERER_PROPERTY_DEPTH_INFINITY.name.c_str(),
                           DEFAULT_MEDIA_MAKER_RENDERER_DEPTH_INFINITY);
    ::ispc::DepthRenderer_set(getIE(), spp, _infinity);
}

DepthRenderer::DepthRenderer()
{
    ispcEquivalent = ::ispc::DepthRenderer_create(this);
}

OSP_REGISTER_RENDERER(DepthRenderer, depth);
OSP_REGISTER_MATERIAL(depth, core::engine::ospray::AdvancedMaterial, default);
} // namespace rendering
} // namespace mediamaker
} // namespace bioexplorer