/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

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

#include "EnzymeReaction.h"

#include <science/common/Assembly.h>
#include <science/common/Logs.h>
#include <science/common/Utils.h>
#include <science/molecularsystems/Protein.h>

#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Scene.h>

using namespace core;

namespace bioexplorer
{
namespace molecularsystems
{
using namespace common;
using namespace details;

EnzymeReaction::EnzymeReaction(core::Scene& scene, const EnzymeReactionDetails& details, AssemblyPtr enzymeAssembly,
                               ProteinPtr enzyme, Proteins& substrates, Proteins& products)
    : _scene(scene)
    , _details(details)
    , _enzymeAssembly(enzymeAssembly)
    , _enzyme(enzyme)
    , _substrates(substrates)
    , _products(products)
{
}

/**
 * @brief This code belongs to a function which is a part of an EnzymeReaction class. The function is responsible for
updating the animation state of the enzyme reaction given the current progress. The progress is a value between 0 and 1
which indicates how far the reaction has progressed.

The code first retrieves the model descriptor for the enzyme object, and the list of enzyme instances. It then checks if
the instance ID supplied is within the range of instances and retrieves the transformation of the enzyme instance ideal
for reaction progress animation.

Then, for each substrate and product molecule in the reaction, the code retrieves the instance for the corresponding
molecule. It also calculates the average direction of all substrate and product molecules relative to the enzyme
molecule.

The code then propagates the transformation to the substrate and product molecules by appropriately setting the
transformations and visibility of the instance based on animation progress. Along the way, it checks if the supplied ID
is within the valid range of molecule instances.

After this, it retrieves the animation details for the enzyme molecule and uses this to synchronize random numbers used
in the initial position of the enzyme object. It then calculates the rotation of the enzyme using the slerp function and
sets the new transformation of the encrypted scene accordingly.

Finally, it marks the scene as modified to indicate that a new animation has been rendered.
 *
 * @param instanceId
 * @param progress
 */
void EnzymeReaction::setProgress(const uint64_t instanceId, const double progress)
{
    auto enzymeModelDescriptor = _enzyme->getModelDescriptor();
    auto& enzymeInstances = enzymeModelDescriptor->getInstances();
    if (instanceId > enzymeInstances.size())
        PLUGIN_THROW("Instance id is out of range for enzyme");
    auto enzymeInstance = enzymeModelDescriptor->getInstance(instanceId);

    Transformation enzymeTransformation;
    auto modelInstanceId = ModelInstanceId(enzymeModelDescriptor->getModelID(), instanceId);
    if (_enzymeInitialTransformations.find(modelInstanceId) == _enzymeInitialTransformations.end())
    {
        enzymeTransformation =
            (instanceId == 0 ? enzymeModelDescriptor->getTransformation() : enzymeInstance->getTransformation());
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
        const auto animationProgress = progress + animationSequenceIndex * animationSequenceInterval;
        auto modelDescriptor = substrate->getModelDescriptor();
        auto& instances = modelDescriptor->getInstances();
        if (instanceId > instances.size())
            PLUGIN_THROW("Instance id is out of range for substrate");

        auto instance = modelDescriptor->getInstance(instanceId);
        auto modelId = modelDescriptor->getModelID();
        Transformation transformation;

        modelInstanceId = ModelInstanceId(modelId, instanceId);
        if (_substrateInitialTransformations.find(modelInstanceId) == _substrateInitialTransformations.end())
        {
            auto transformation =
                (instanceId == 0 ? modelDescriptor->getTransformation() : instance->getTransformation());
            _substrateInitialTransformations[modelInstanceId] = transformation;
        }
        else
            transformation = _substrateInitialTransformations[modelInstanceId];

        auto translation = transformation.getTranslation();
        averageSubstrateDirection += normalize(translation - enzymeTranslation);

        if (progress < 0.5)
        {
            const double indexedProgress = std::max(0.0, 1.0 - 2.0 * animationProgress);
            transformation.setTranslation(enzymeTranslation + (translation - enzymeTranslation) * indexedProgress);
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
        const auto animationProgress = progress + animationSequenceIndex * animationSequenceInterval;
        auto modelDescriptor = product->getModelDescriptor();
        auto& instances = modelDescriptor->getInstances();
        if (instanceId > instances.size())
            PLUGIN_THROW("Instance id is out of range for product");

        auto instance = modelDescriptor->getInstance(instanceId);
        Transformation transformation;

        auto modelId = modelDescriptor->getModelID();
        modelInstanceId = ModelInstanceId(modelId, instanceId);
        if (_productInitialTransformations.find(modelInstanceId) == _productInitialTransformations.end())
        {
            transformation = (instanceId == 0 ? modelDescriptor->getTransformation() : instance->getTransformation());
            _productInitialTransformations[modelInstanceId] = transformation;
        }
        else
            transformation = _productInitialTransformations[modelInstanceId];

        auto translation = transformation.getTranslation();
        averageProductDirection += normalize(translation - enzymeTranslation);

        if (progress >= 0.5)
        {
            const double indexedProgress = std::min(0.0, -2.0 * (animationProgress - 0.5));
            transformation.setTranslation(enzymeTranslation + (translation - enzymeTranslation) * indexedProgress);
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
    transformations.push_back(
        _enzymeAssembly->getShape()->getTransformation(instanceId, enzymeInstances.size(), enzymeAnimationDetails));
    transformations.push_back(_enzyme->getTransformation());

    Transformation enzymeAlignmentTransformation;
    const auto rotation =
        slerp(safeQuatlookAt(averageSubstrateDirection), safeQuatlookAt(averageProductDirection), progress);

    enzymeAlignmentTransformation.setRotation(rotation);
    transformations.push_back(enzymeAlignmentTransformation);
    const auto finalTransformation = combineTransformations(transformations);

    if (instanceId == 0)
        enzymeModelDescriptor->setTransformation(finalTransformation);
    enzymeInstance->setTransformation(finalTransformation);

    _scene.markModified(false);
}

/**
 * @brief This is a method called _getMoleculeRotation that belongs to a class
named EnzymeReaction. This method returns a quaternion representation of a
molecule's rotation based on the progress of a reaction and a given rotation
speed.


First, the method calculates an angle by multiplying the progress and the
rotation speed with π. Then, it calculates the cosine and sine of this angle for
the roll, yaw, and pitch axes of the molecule's rotation. Finally, it returns a
Quaterniond object with the calculated roll, pitch, and yaw values.
 *
 * @param progress
 * @param rotationSpeed
 * @return Quaterniond
 */
Quaterniond EnzymeReaction::_getMoleculeRotation(const double progress, const double rotationSpeed) const
{
    const double angle = rotationSpeed * progress * M_PI;
    const double roll = cos(angle * 0.91);
    const double yaw = cos(angle * 1.25);
    const double pitch = sin(angle * 0.27);
    return Quaterniond({yaw, pitch, roll});
}

} // namespace molecularsystems
} // namespace bioexplorer
