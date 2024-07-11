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

#include "GolgiStyleRenderer.h"

#include <science/common/Properties.h>

#include <platform/engines/ospray/ispc/render/utils/AdvancedMaterial.h>

#include "GolgiStyleRenderer_ispc.h"

namespace bioexplorer
{
namespace rendering
{
void GolgiStyleRenderer::commit()
{
    AbstractRenderer::commit();
    _exponent = getParam1f(BIOEXPLORER_RENDERER_PROPERTY_GOLGI_EXPONENT.name.c_str(),
                           BIOEXPLORER_DEFAULT_RENDERER_GOLGI_EXPONENT);
    _inverse =
        getParam(BIOEXPLORER_RENDERER_PROPERTY_GOLGI_INVERSE.name.c_str(), BIOEXPLORER_DEFAULT_RENDERER_GOLGI_INVERSE);
    ::ispc::GolgiStyleRenderer_set(getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr), spp, _exponent, _inverse,
                                   _anaglyphEnabled, (ispc::vec3f&)_anaglyphIpdOffset);
}

GolgiStyleRenderer::GolgiStyleRenderer()
{
    ispcEquivalent = ::ispc::GolgiStyleRenderer_create(this);
}

OSP_REGISTER_RENDERER(GolgiStyleRenderer, bio_explorer_golgi_style);
OSP_REGISTER_MATERIAL(bio_explorer_golgi_style, core::engine::ospray::AdvancedMaterial, default);
} // namespace rendering
} // namespace bioexplorer
