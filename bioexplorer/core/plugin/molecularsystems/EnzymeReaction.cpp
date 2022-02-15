/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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

#include "EnzymeReaction.h"

#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>
#include <plugin/molecularsystems/Protein.h>

#include <brayns/engineapi/Model.h>
#include <brayns/engineapi/Scene.h>

namespace bioexplorer
{
namespace molecularsystems
{
using namespace common;
using namespace details;

EnzymeReaction::EnzymeReaction(Scene& scene,
                               const EnzymeReactionDetails& details,
                               ProteinPtr enzyme, ProteinPtr substrate,
                               ProteinPtr product)
    : _scene(scene)
    , _details(details)
{
    _enzyme = enzyme;
    _substrate = substrate;
    _product = product;
}

void EnzymeReaction::setProgress(const uint64_t instanceId,
                                 const double progress)
{
    auto substrateModelDescriptor = _substrate->getModelDescriptor();
    auto& substrateInstances = substrateModelDescriptor->getInstances();
    if (instanceId > substrateInstances.size())
        PLUGIN_THROW("Instance id is out of range for substrate");

    auto enzymeModelDescriptor = _enzyme->getModelDescriptor();
    auto& enzymeInstances = enzymeModelDescriptor->getInstances();
    if (instanceId > enzymeInstances.size())
        PLUGIN_THROW("Instance id is out of range for enzyme");

    auto productModelDescriptor = _product->getModelDescriptor();
    auto& productInstances = productModelDescriptor->getInstances();
    if (instanceId > productInstances.size())
        PLUGIN_THROW("Instance id is out of range for product");

    auto substrateInstance = substrateModelDescriptor->getInstance(instanceId);
    auto substrateTransformation =
        (instanceId == 0 ? substrateModelDescriptor->getTransformation()
                         : substrateInstance->getTransformation());
    const Vector3d substrateTranslation =
        substrateTransformation.getTranslation();

    auto enzymeInstance = enzymeModelDescriptor->getInstance(instanceId);
    auto enzymeTransformation =
        (instanceId == 0 ? enzymeModelDescriptor->getTransformation()
                         : enzymeInstance->getTransformation());

    enzymeTransformation.setTranslation(
        substrateTranslation + Vector3d(10.0 * (progress - 0.5), 0.0, 0.0));
    if (instanceId == 0)
        enzymeModelDescriptor->setTransformation(enzymeTransformation);
    enzymeInstance->setTransformation(enzymeTransformation);

    auto productInstance = productModelDescriptor->getInstance(instanceId);
    if (instanceId == 0)
        productModelDescriptor->setTransformation(substrateTransformation);
    productInstance->setTransformation(substrateTransformation);

    productInstance->setVisible(progress > 0.5);
    substrateInstance->setVisible(progress <= 0.5);

    _scene.markModified(false);
}

} // namespace molecularsystems
} // namespace bioexplorer
