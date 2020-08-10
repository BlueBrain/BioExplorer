/* Copyright (c) 2020, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "BioExplorerRenderer.h"

// ospray
#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/common/Model.h>
#include <ospray/SDK/lights/Light.h>

// ispc exports
#include "BioExplorerRenderer_ispc.h"

using namespace ospray;

namespace BioExplorer
{
void BioExplorerRenderer::commit()
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
    _bgMaterial =
        (brayns::obj::BioExplorerMaterial*)getParamObject("bgMaterial",
                                                          nullptr);
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

    ispc::BioExplorerRenderer_set(
        getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr), _shadows,
        _softShadows, _softShadowsSamples, _giStrength, _giDistance, _giSamples,
        _randomNumber, _timestamp, spp, _lightPtr, _lightArray.size(),
        _exposure, _fogThickness, _fogStart, _useHardwareRandomizer,
        _maxBounces, _showBackground);
}

BioExplorerRenderer::BioExplorerRenderer()
{
    ispcEquivalent = ispc::BioExplorerRenderer_create(this);
}

OSP_REGISTER_RENDERER(BioExplorerRenderer, bio_explorer);
} // namespace BioExplorer
