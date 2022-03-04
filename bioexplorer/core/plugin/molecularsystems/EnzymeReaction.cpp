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

#include <plugin/common/Assembly.h>
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
                               AssemblyPtr enzymeAssembly, ProteinPtr enzyme,
                               Proteins& substrates, Proteins& products)
    : _scene(scene)
    , _details(details)
    , _enzymeAssembly(enzymeAssembly)
    , _enzyme(enzyme)
    , _substrates(substrates)
    , _products(products)
{
}

void EnzymeReaction::setProgress(const uint64_t instanceId,
                                 const double progress)
{
    auto enzymeModelDescriptor = _enzyme->getModelDescriptor();
    auto& enzymeInstances = enzymeModelDescriptor->getInstances();
    if (instanceId > enzymeInstances.size())
        PLUGIN_THROW("Instance id is out of range for enzyme");
    auto enzymeInstance = enzymeModelDescriptor->getInstance(instanceId);

    Transformation enzymeTransformation;
    auto modelInstanceId =
        ModelInstanceId(enzymeModelDescriptor->getModelID(), instanceId);
    if (_enzymeInitialTransformations.find(modelInstanceId) ==
        _enzymeInitialTransformations.end())
    {
        enzymeTransformation =
            (instanceId == 0 ? enzymeModelDescriptor->getTransformation()
                             : enzymeInstance->getTransformation());
        _enzymeInitialTransformations[modelInstanceId] = enzymeTransformation;
    }
    else
        enzymeTransformation = _enzymeInitialTransformations[modelInstanceId];
    const auto enzymeTranslation = enzymeTransformation.getTranslation();

    // Substrates (Flying in)
    int64_t animationSequenceIndex = 0;
    const double animationSequenceInterval = 0.1;
    Transformations substrateTransformations;
    Vector3d averageSubstrateDirection;
    for (auto& substrate : _substrates)
    {
        const auto animationProgress =
            progress + animationSequenceIndex * animationSequenceInterval;
        auto modelDescriptor = substrate->getModelDescriptor();
        auto& instances = modelDescriptor->getInstances();
        if (instanceId > instances.size())
            PLUGIN_THROW("Instance id is out of range for substrate");

        auto instance = modelDescriptor->getInstance(instanceId);
        auto modelId = modelDescriptor->getModelID();
        Transformation transformation;

        modelInstanceId = ModelInstanceId(modelId, instanceId);
        if (_substrateInitialTransformations.find(modelInstanceId) ==
            _substrateInitialTransformations.end())
        {
            auto transformation =
                (instanceId == 0 ? modelDescriptor->getTransformation()
                                 : instance->getTransformation());
            _substrateInitialTransformations[modelInstanceId] = transformation;
        }
        else
            transformation = _substrateInitialTransformations[modelInstanceId];

        auto translation = transformation.getTranslation();
        averageSubstrateDirection += normalize(translation - enzymeTranslation);

        if (progress < 0.5)
        {
            const double indexedProgress =
                std::max(0.0, 1.0 - 2.0 * animationProgress);
            transformation.setTranslation(enzymeTranslation +
                                          (translation - enzymeTranslation) *
                                              indexedProgress);
            transformation.setRotation(_getMoleculeRotation(progress));

            if (instanceId == 0)
                modelDescriptor->setTransformation(transformation);
            instance->setTransformation(transformation);
        }
        instance->setVisible(progress < 0.5);
        ++animationSequenceIndex;
    }
    averageSubstrateDirection /= _substrates.size();

    // Products (Flying out)
    animationSequenceIndex = -1;
    Transformations productTransformations;
    Vector3d averageProductDirection;
    for (auto& product : _products)
    {
        const auto animationProgress =
            progress + animationSequenceIndex * animationSequenceInterval;
        auto modelDescriptor = product->getModelDescriptor();
        auto& instances = modelDescriptor->getInstances();
        if (instanceId > instances.size())
            PLUGIN_THROW("Instance id is out of range for product");

        auto instance = modelDescriptor->getInstance(instanceId);
        Transformation transformation;

        auto modelId = modelDescriptor->getModelID();
        modelInstanceId = ModelInstanceId(modelId, instanceId);
        if (_productInitialTransformations.find(modelInstanceId) ==
            _productInitialTransformations.end())
        {
            transformation =
                (instanceId == 0 ? modelDescriptor->getTransformation()
                                 : instance->getTransformation());
            _productInitialTransformations[modelInstanceId] = transformation;
        }
        else
            transformation = _productInitialTransformations[modelInstanceId];

        auto translation = transformation.getTranslation();
        averageProductDirection += normalize(translation - enzymeTranslation);

        if (progress >= 0.5)
        {
            const double indexedProgress =
                std::min(0.0, -2.0 * (animationProgress - 0.5));
            transformation.setTranslation(enzymeTranslation +
                                          (translation - enzymeTranslation) *
                                              indexedProgress);
            transformation.setRotation(_getMoleculeRotation(progress));

            if (instanceId == 0)
                modelDescriptor->setTransformation(transformation);
            instance->setTransformation(transformation);
        }
        instance->setVisible(progress >= 0.5);
        --animationSequenceIndex;
    }
    averageProductDirection /= _products.size();

    const auto enzymeAnimationDetails = _enzyme->getAnimationDetails();
    srand(enzymeAnimationDetails.seed);

    // Synchronize random numbers with the ones used to create the enzyme
    // initial positions. This is used by the getShape()->getTransformation()
    // call
    for (uint64_t i = 0; i < instanceId * 3; ++i)
        rand();

    // Enzyme rotation according to substrates and products positions
    Transformations transformations;
    transformations.push_back(_enzymeAssembly->getTransformation());
    transformations.push_back(_enzymeAssembly->getShape()->getTransformation(
        instanceId, enzymeInstances.size(), enzymeAnimationDetails));
    transformations.push_back(_enzyme->getTransformation());

    Transformation enzymeAlignmentTransformation;
    const auto rotation =
        slerp(safeQuatlookAt(averageSubstrateDirection),
              safeQuatlookAt(averageProductDirection), progress);

    enzymeAlignmentTransformation.setRotation(rotation);
    transformations.push_back(enzymeAlignmentTransformation);
    const auto finalTransformation = combineTransformations(transformations);

    if (instanceId == 0)
        enzymeModelDescriptor->setTransformation(finalTransformation);
    enzymeInstance->setTransformation(finalTransformation);

    _scene.markModified(false);
}

Quaterniond EnzymeReaction::_getMoleculeRotation(
    const double progress, const double rotationSpeed) const
{
    const double angle = rotationSpeed * progress * M_PI;
    const double roll = cos(angle * 0.91);
    const double yaw = cos(angle * 1.25);
    const double pitch = sin(angle * 0.27);
    return Quaterniond({yaw, pitch, roll});
}

} // namespace molecularsystems
} // namespace bioexplorer