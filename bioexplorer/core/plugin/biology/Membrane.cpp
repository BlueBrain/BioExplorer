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

#include "Membrane.h"
#include "Protein.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/GeneralSettings.h>
#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>

#include <brayns/engineapi/Material.h>

namespace bioexplorer
{
namespace biology
{
Membrane::Membrane(Scene& scene, const Vector3f& assemblyPosition,
                   const Quaterniond& assemblyRotation,
                   const Vector4fs& clippingPlanes)
    : _scene(scene)
    , _assemblyPosition(assemblyPosition)
    , _assemblyRotation(assemblyRotation)
    , _clippingPlanes(clippingPlanes)
{
}

Membrane::~Membrane()
{
    for (const auto& protein : _proteins)
        _scene.removeModel(protein.second->getModelDescriptor()->getModelID());
}

} // namespace biology
} // namespace bioexplorer
