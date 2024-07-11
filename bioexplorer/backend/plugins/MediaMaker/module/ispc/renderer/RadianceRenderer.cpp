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

#include "RadianceRenderer.h"

#include <platform/engines/ospray/ispc/render/utils/AdvancedMaterial.h>

// ospray
#include <ospray/SDK/lights/Light.h>

// ::ispc exports
#include "RadianceRenderer_ispc.h"

using namespace core;
using namespace ospray;

namespace bioexplorer
{
namespace mediamaker
{
namespace rendering
{
void RadianceRenderer::commit()
{
    Renderer::commit();

    ::ispc::RadianceRenderer_set(getIE(), spp);
}

RadianceRenderer::RadianceRenderer()
{
    ispcEquivalent = ::ispc::RadianceRenderer_create(this);
}

OSP_REGISTER_RENDERER(RadianceRenderer, radiance);
OSP_REGISTER_MATERIAL(radiance, core::engine::ospray::AdvancedMaterial, default);
} // namespace rendering
} // namespace mediamaker
} // namespace bioexplorer