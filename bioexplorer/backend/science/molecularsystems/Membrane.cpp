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

#include "Membrane.h"
#include "Protein.h"

#include <science/common/GeneralSettings.h>
#include <science/common/Logs.h>
#include <science/common/Node.h>
#include <science/common/Utils.h>
#include <science/common/shapes/Shape.h>

#include <platform/core/engineapi/Material.h>
#include <platform/core/engineapi/Model.h>

using namespace core;

namespace bioexplorer
{
using namespace common;
using namespace details;

namespace molecularsystems
{
Membrane::Membrane(const MembraneDetails& details, Scene& scene, const Vector3d& assemblyPosition,
                   const Quaterniond& assemblyRotation, const ShapePtr shape, const ProteinMap& transmembraneProteins)
    : SDFGeometries(NO_GRID_ALIGNMENT, assemblyPosition, assemblyRotation)
    , _scene(scene)
    , _details(details)
    , _nbOccurrences{0}
    , _transmembraneProteins(transmembraneProteins)
    , _shape(shape)
{
    // Lipid models
    strings lipidPDBIds = split(_details.lipidPDBIds, CONTENTS_DELIMITER);
    strings lipidContents = split(_details.lipidContents, CONTENTS_DELIMITER);

    double lipidAverageSize = 0.0;
    size_t i = 0;
    for (const auto& lipidContent : lipidContents)
    {
        ProteinDetails pd;
        pd.assemblyName = _details.assemblyName;
        pd.name = _getElementNameFromId(i);
        pd.pdbId = lipidPDBIds[i];
        pd.contents = lipidContent;
        pd.recenter = true;
        pd.atomRadiusMultiplier = _details.atomRadiusMultiplier;
        pd.representation = _details.representation;
        pd.loadBonds = _details.loadBonds;
        pd.loadNonPolymerChemicals = _details.loadNonPolymerChemicals;

        // Create model
        ProteinPtr lipid(new Protein(_scene, pd));
        const auto& lipidSize = lipid->getBounds().getSize();
        lipidAverageSize += std::min(lipidSize.x, std::min(lipidSize.y, lipidSize.z));
        _lipids[pd.name] = std::move(lipid);
        ++i;
    }
    lipidAverageSize /= lipidContents.size();
    lipidAverageSize /= _details.lipidDensity;

    _nbOccurrences = _shape->getSurface() / lipidAverageSize * 2.0; // WHY DO I HAVE TO DOUBLE IT!

    _processInstances();

    // Add models to the scene
    for (size_t i = 0; i < lipidContents.size(); ++i)
        _scene.addModel(_lipids[_getElementNameFromId(i)]->getModelDescriptor());
}

Membrane::~Membrane()
{
    for (const auto& lipid : _lipids)
        _scene.removeModel(lipid.second->getModelDescriptor()->getModelID());
}

double Membrane::_getDisplacementValue(const DisplacementElement&)
{
    return 0.0;
}

void Membrane::_processInstances()
{
    const auto rotation = doublesToQuaterniond(_details.lipidRotation);
    const auto MolecularSystemAnimationDetails = doublesToMolecularSystemAnimationDetails(_details.animationParams);
    srand(MolecularSystemAnimationDetails.seed);

    std::map<size_t, size_t> instanceCounts;
    for (size_t i = 0; i < _lipids.size(); ++i)
        instanceCounts[i] = 0;

    for (uint64_t occurrence = 0; occurrence < _nbOccurrences; ++occurrence)
    {
        try
        {
            const size_t id = occurrence % _lipids.size();
            auto lipid = _lipids[_getElementNameFromId(id)];
            auto md = lipid->getModelDescriptor();

            const auto& model = md->getModel();
            const auto& bounds = model.getBounds();
            const Vector3d& center = bounds.getCenter();

            Transformations transformations;

            Transformation assemblyTransformation;
            assemblyTransformation.setTranslation(_position);
            assemblyTransformation.setRotation(_rotation);
            transformations.push_back(assemblyTransformation);

            const auto shapeTransformation =
                _shape->getTransformation(occurrence, _nbOccurrences, MolecularSystemAnimationDetails);
            transformations.push_back(shapeTransformation);

            Transformation lipidTransformation;
            lipidTransformation.setRotation(rotation);
            transformations.push_back(lipidTransformation);

            const Transformation finalTransformation = combineTransformations(transformations);
            const auto& finalTranslation = finalTransformation.getTranslation();
            const auto& finalRotation = finalTransformation.getRotation();

            // Collision with trans-membrane proteins
            bool collision = false;
            for (const auto& protein : _transmembraneProteins)
            {
                const auto transMembraneRadius = protein.second->getTransMembraneRadius();
                const auto transMembraneOffset = protein.second->getTransMembraneOffset();
                auto modelDescriptor = protein.second->getModelDescriptor();
                const auto& instances = modelDescriptor->getInstances();
                const auto& instanceSize = modelDescriptor->getModel().getBounds().getSize();
                for (const auto& instance : instances)
                {
                    const auto& tf = instance.getTransformation();
                    const Vector3d proteinBase =
                        finalTranslation + transMembraneOffset * normalize(finalRotation * UP_VECTOR);
                    if (length(finalTranslation - tf.getTranslation()) < transMembraneRadius)
                    {
                        collision = true;
                        break;
                    }
                }
            }
            if (collision)
                continue;

            if (instanceCounts[id] == 0)
                md->setTransformation(finalTransformation);
            const ModelInstance instance(true, false, finalTransformation);
            md->addInstance(instance);

            instanceCounts[id] = instanceCounts[id] + 1;
        }
        catch (const std::runtime_error&)
        {
            // Instance is clipped
        }
    }
}

std::string Membrane::_getElementNameFromId(const size_t id) const
{
    return _details.assemblyName + "_Membrane_" + std::to_string(id);
}

} // namespace molecularsystems
} // namespace bioexplorer
