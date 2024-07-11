/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

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

#include "BasicRenderer.h"

#include "BasicRenderer_ispc.h"

namespace core
{
namespace engine
{
namespace ospray
{
void BasicRenderer::commit()
{
    AbstractRenderer::commit();

    ::ispc::BasicRenderer_set(getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr), _timestamp, spp, _lightPtr,
                              _lightArray.size(), _anaglyphEnabled, (ispc::vec3f&)_anaglyphIpdOffset);
}

BasicRenderer::BasicRenderer()
{
    ispcEquivalent = ::ispc::BasicRenderer_create(this);
}

OSP_REGISTER_RENDERER(BasicRenderer, basic);
} // namespace ospray
} // namespace engine
} // namespace core