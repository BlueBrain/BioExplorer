/* Copyright (c) 2015-2022, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 *
 * This file is part of Brayns <https://github.com/BlueBrain/Brayns>
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

#include "FieldRenderer.h"

// ospray
#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/lights/Light.h>
#include <ospray/SDK/transferFunction/TransferFunction.h>

// ispc exports
#include "FieldRenderer_ispc.h"

using namespace ospray;

namespace brayns
{
void FieldRenderer::commit()
{
    Renderer::commit();

    _lightData = (ospray::Data*)getParamData("lights");
    _lightArray.clear();

    if (_lightData)
        for (size_t i = 0; i < _lightData->size(); ++i)
            _lightArray.push_back(
                ((ospray::Light**)_lightData->data)[i]->getIE());

    _lightPtr = _lightArray.empty() ? nullptr : &_lightArray[0];

    _exposure = getParam1f("exposure", 1.f);
    _randomNumber = getParam1i("randomNumber", 0);
    _timestamp = getParam1f("timestamp", 0.f);
    _cutoff = getParam1f("cutoff", 100.f);
    _spaceDistortion = getParam1f("spaceDistortion", 0.f);
    _renderDirections = getParam("renderDirections", false);
    _processGeometry = getParam("processGeometry", false);
    _normalized = getParam("normalized", false);

    // Sampling
    _nbRaySteps = getParam1i("nbRaySteps", 64);
    _nbRefinementSteps = getParam1i("nbRefinementSteps", 8);
    _alphaCorrection = getParam1f("alphaCorrection", 1.0f);

    // Transfer function
    ospray::TransferFunction* transferFunction =
        (ospray::TransferFunction*)getParamObject("transferFunction", nullptr);
    if (transferFunction)
        ispc::FieldRenderer_setTransferFunction(getIE(),
                                                transferFunction->getIE());

    // Data
    _userData = getParamData("simulationData");
    _userDataSize = _userData ? _userData->size() : 0;

    // Renderer
    ispc::FieldRenderer_set(getIE(),
                            (_userData ? (float*)_userData->data : nullptr),
                            _userDataSize, _randomNumber, _timestamp, spp,
                            _lightPtr, _lightArray.size(), _nbRaySteps,
                            _nbRefinementSteps, _exposure, _cutoff,
                            _alphaCorrection, _renderDirections,
                            _spaceDistortion, _processGeometry, _normalized);
}

FieldRenderer::FieldRenderer()
{
    ispcEquivalent = ispc::FieldRenderer_create(this);
}

OSP_REGISTER_RENDERER(FieldRenderer, field_renderer);
} // namespace brayns
