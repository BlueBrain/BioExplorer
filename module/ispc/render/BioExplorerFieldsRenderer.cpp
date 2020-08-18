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

#include "BioExplorerFieldsRenderer.h"

// ospray
#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/lights/Light.h>
#include <ospray/SDK/transferFunction/TransferFunction.h>

// ispc exports
#include "BioExplorerFieldsRenderer_ispc.h"

using namespace ospray;

namespace brayns
{
void BioExplorerFieldsRenderer::commit()
{
    Renderer::commit();

    _lightData = (ospray::Data*)getParamData("lights");
    _lightArray.clear();

    if (_lightData)
        for (size_t i = 0; i < _lightData->size(); ++i)
            _lightArray.push_back(
                ((ospray::Light**)_lightData->data)[i]->getIE());

    _lightPtr = _lightArray.empty() ? nullptr : &_lightArray[0];

    _bgMaterial =
        (brayns::obj::BioExplorerMaterial*)getParamObject("bgMaterial",
                                                          nullptr);

    _useHardwareRandomizer = getParam("useHardwareRandomizer", 0);

    _shadows = getParam1f("shadows", 0.f);
    _softShadows = getParam1f("softShadows", 0.f);

    _shadingEnabled = bool(getParam1i("shadingEnabled", 1));
    _softnessEnabled = bool(getParam1i("softnessEnabled", 0));

    _exposure = getParam1f("exposure", 1.f);
    _randomNumber = getParam1i("randomNumber", 0);
    _timestamp = getParam1f("timestamp", 0.f);

    // Sampling
    _step = getParam1f("step", 0.1f);
    _maxSteps = getParam1i("maxSteps", 32);

    // Extra
    _cutoff = getParam1f("cutoff", 1.f);

    // Octree
    _userData = getParamData("simulationData");
    _userDataSize = _userData ? _userData->size() : 0;

    // Transfer function
    ospray::TransferFunction* transferFunction =
        (ospray::TransferFunction*)getParamObject("transferFunction", nullptr);
    if (transferFunction)
        ispc::BioExplorerFieldsRenderer_setTransferFunction(
            getIE(), transferFunction->getIE());

    // Renderer
    ispc::BioExplorerFieldsRenderer_set(
        getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr),
        (_userData ? (float*)_userData->data : nullptr), _userDataSize,
        _shadows, _softShadows, _shadingEnabled, _randomNumber, _timestamp, spp,
        _softnessEnabled, _lightPtr, _lightArray.size(), _step, _maxSteps,
        _exposure, _useHardwareRandomizer, _cutoff);
}

BioExplorerFieldsRenderer::BioExplorerFieldsRenderer()
{
    ispcEquivalent = ispc::BioExplorerFieldsRenderer_create(this);
}

OSP_REGISTER_RENDERER(BioExplorerFieldsRenderer, bio_explorer_fields);
} // namespace brayns
