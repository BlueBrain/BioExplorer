/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
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

#include "VectorFieldsRenderer.h"

#include <science/common/Properties.h>

#include <platform/core/common/Properties.h>
#include <platform/engines/ospray/OSPRayProperties.h>

// ospray
#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/lights/Light.h>
#include <ospray/SDK/transferFunction/TransferFunction.h>

// ::ispc exports
#include "VectorFieldsRenderer_ispc.h"

using namespace core;

namespace bioexplorer
{
namespace rendering
{
void VectorFieldsRenderer::commit()
{
    AbstractRenderer::commit();

    _exposure = getParam1f(COMMON_PROPERTY_EXPOSURE.name.c_str(), DEFAULT_COMMON_EXPOSURE);
    _minRayStep = getParam1f(BIOEXPLORER_RENDERER_PROPERTY_FIELDS_MIN_RAY_STEP.name.c_str(),
                             BIOEXPLORER_DEFAULT_RENDERER_FIELDS_MIN_RAY_STEP);
    _nbRaySteps = getParam1i(BIOEXPLORER_RENDERER_PROPERTY_FIELDS_NB_RAY_STEPS.name.c_str(),
                             BIOEXPLORER_DEFAULT_RENDERER_FIELDS_NB_RAY_STEPS);
    _nbRayRefinementSteps = getParam1i(BIOEXPLORER_RENDERER_PROPERTY_FIELDS_NB_RAY_REFINEMENT_STEPS.name.c_str(),
                                       BIOEXPLORER_DEFAULT_RENDERER_FIELDS_NB_RAY_REFINEMENT_STEPS);
    _alphaCorrection = getParam1f(RENDERER_PROPERTY_ALPHA_CORRECTION.name.c_str(), DEFAULT_RENDERER_ALPHA_CORRECTION);
    _cutoff = getParam1f(BIOEXPLORER_RENDERER_PROPERTY_FIELDS_CUTOFF_DISTANCE.name.c_str(),
                         BIOEXPLORER_DEFAULT_RENDERER_FIELDS_CUTOFF_DISTANCE);
    _showVectorDirections = getParam(BIOEXPLORER_RENDERER_PROPERTY_FIELDS_SHOW_VECTOR_DIRECTIONS.name.c_str(),
                                     BIOEXPLORER_DEFAULT_RENDERER_FIELDS_SHOW_VECTOR_DIRECTIONS);

    // Octree
    _userData = getParamData(RENDERER_PROPERTY_USER_DATA);
    _userDataSize = _userData ? _userData->size() : 0;

    // Transfer function
    ::ospray::TransferFunction* transferFunction =
        (::ospray::TransferFunction*)getParamObject(RENDERER_PROPERTY_TRANSFER_FUNCTION, nullptr);
    if (transferFunction)
        ::ispc::VectorFieldsRenderer_setTransferFunction(getIE(), transferFunction->getIE());

    ::ispc::VectorFieldsRenderer_set(getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr),
                                     (_userData ? (float*)_userData->data : nullptr), _userDataSize, _randomNumber,
                                     _timestamp, spp, _lightPtr, _lightArray.size(), _minRayStep, _nbRaySteps,
                                     _nbRayRefinementSteps, _exposure, _useHardwareRandomizer, _cutoff,
                                     _alphaCorrection, _showVectorDirections, _anaglyphEnabled,
                                     (ispc::vec3f&)_anaglyphIpdOffset);
}

VectorFieldsRenderer::VectorFieldsRenderer()
{
    ispcEquivalent = ::ispc::VectorFieldsRenderer_create(this);
}

OSP_REGISTER_RENDERER(VectorFieldsRenderer, vector_fields);
OSP_REGISTER_MATERIAL(vector_fields, core::engine::ospray::AdvancedMaterial, default);
} // namespace rendering
} // namespace bioexplorer
