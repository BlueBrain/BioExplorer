/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2021 Blue BrainProject / EPFL
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "AdvancedRenderer.h"

// ospray
#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/common/Model.h>
#include <ospray/SDK/lights/Light.h>

// ispc exports
#include "AdvancedRenderer_ispc.h"

using namespace ospray;

namespace bioexplorer
{
void AdvancedRenderer::commit()
{
    Renderer::commit();

    _lightData = (ospray::Data*)getParamData("lights");
    _lightArray.clear();

    if (_lightData)
        for (size_t i = 0; i < _lightData->size(); ++i)
            _lightArray.push_back(
                ((ospray::Light**)_lightData->data)[i]->getIE());

    _lightPtr = _lightArray.empty() ? nullptr : &_lightArray[0];

    _timestamp = getParam1f("timestamp", 0.f);
    _bgMaterial = (AdvancedMaterial*)getParamObject("bgMaterial", nullptr);
    _exposure = getParam1f("exposure", 1.f);

    _useHardwareRandomizer = getParam("useHardwareRandomizer", 0);
    _showBackground = getParam("showBackground", 0);

    _fogThickness = getParam1f("fogThickness", 1e6f);
    _fogStart = getParam1f("fogStart", 0.f);

    _shadows = getParam1f("shadows", 0.f);
    _softShadows = getParam1f("softShadows", 0.f);
    _softShadowsSamples = getParam1i("softShadowsSamples", 1);

    _giStrength = getParam1f("giWeight", 0.f);
    _giDistance = getParam1f("giDistance", 1e20f);
    _giSamples = getParam1i("giSamples", 1);

    _maxBounces = getParam1i("maxBounces", 3);
    _randomNumber = getParam1i("randomNumber", 0);

    ispc::AdvancedRenderer_set(getIE(),
                               (_bgMaterial ? _bgMaterial->getIE() : nullptr),
                               _shadows, _softShadows, _softShadowsSamples,
                               _giStrength, _giDistance, _giSamples,
                               _randomNumber, _timestamp, spp, _lightPtr,
                               _lightArray.size(), _exposure, _fogThickness,
                               _fogStart, _useHardwareRandomizer, _maxBounces,
                               _showBackground);
}

AdvancedRenderer::AdvancedRenderer()
{
    ispcEquivalent = ispc::AdvancedRenderer_create(this);
}

OSP_REGISTER_RENDERER(AdvancedRenderer, bio_explorer);
} // namespace bioexplorer
