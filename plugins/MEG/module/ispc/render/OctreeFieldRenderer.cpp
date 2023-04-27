/* Copyright (c) 2015-2023, EPFL/Blue Brain Project
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

#include "OctreeFieldRenderer.h"

// ospray
#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/lights/Light.h>
#include <ospray/SDK/transferFunction/TransferFunction.h>

// ispc exports
#include "OctreeFieldRenderer_ispc.h"

using namespace ospray;

namespace brayns
{
void OctreeFieldRenderer::commit()
{
    Renderer::commit();

    _lightData = (ospray::Data*)getParamData("lights");
    _lightArray.clear();

    if (_lightData)
        for (size_t i = 0; i < _lightData->size(); ++i)
            _lightArray.push_back(((ospray::Light**)_lightData->data)[i]->getIE());

    _lightPtr = _lightArray.empty() ? nullptr : &_lightArray[0];

    _exposure = getParam1f("exposure", 1.f);

    // Sampling
    _minRayStep = getParam1f("minRayStep", 0.1f);
    _nbRaySteps = getParam1i("nbRaySteps", 8);
    _nbRayRefinementSteps = getParam1i("nbRayRefinementSteps", 8);
    _alphaCorrection = getParam1f("alphaCorrection", 1.0f);

    // Extra
    _cutoff = getParam1f("cutoff", 1.f);

    // Octree
    _userData = getParamData("simulationData");
    _userDataSize = _userData ? _userData->size() : 0;

    // Transfer function
    ospray::TransferFunction* transferFunction = (ospray::TransferFunction*)getParamObject("transferFunction", nullptr);
    if (transferFunction)
        ispc::OctreeFieldRenderer_setTransferFunction(getIE(), transferFunction->getIE());

    // Renderer
    ispc::OctreeFieldRenderer_set(getIE(), (_userData ? (float*)_userData->data : nullptr), _userDataSize, spp,
                                  _lightPtr, _lightArray.size(), _minRayStep, _nbRaySteps, _nbRayRefinementSteps,
                                  _exposure, _cutoff, _alphaCorrection);
}

OctreeFieldRenderer::OctreeFieldRenderer()
{
    ispcEquivalent = ispc::OctreeFieldRenderer_create(this);
}

OSP_REGISTER_RENDERER(OctreeFieldRenderer, octree_field_renderer);
} // namespace brayns