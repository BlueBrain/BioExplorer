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

#include "Membrane.h"
#include "Protein.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/GeneralSettings.h>
#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>
#include <plugin/common/shapes/Shape.h>

#include <brayns/engineapi/Material.h>

namespace bioexplorer
{
namespace biology
{
Membrane::Membrane(const MembraneDetails& details, Scene& scene,
                   const Vector3d& assemblyPosition,
                   const Quaterniond& assemblyRotation, const ShapePtr shape,
                   const ProteinMap& transmembraneProteins)
    : _scene(scene)
    , _details(details)
    , _nbOccurrences{0}
    , _transmembraneProteins(transmembraneProteins)
    , _assemblyPosition(assemblyPosition)
    , _assemblyRotation(assemblyRotation)
    , _shape(shape)
{
    // Lipid models
    std::vector<std::string> lipidContents =
        split(_details.lipidContents, CONTENTS_DELIMITER);

    double lipidAverageSize = 0.0;
    size_t i = 0;
    for (const auto& lipidContent : lipidContents)
    {
        ProteinDetails pd;
        pd.assemblyName = _details.assemblyName;
        pd.name = _getElementNameFromId(i);
        pd.contents = lipidContent;
        pd.recenter = true;
        pd.atomRadiusMultiplier = _details.atomRadiusMultiplier;
        pd.representation = _details.representation;
        pd.loadBonds = _details.loadBonds;
        pd.loadNonPolymerChemicals = _details.loadNonPolymerChemicals;

        // Create model
        ProteinPtr lipid(new Protein(_scene, pd));
        const auto& lipidSize = lipid->getBounds().getSize();
        lipidAverageSize +=
            std::min(lipidSize.x, std::min(lipidSize.y, lipidSize.z));
        _lipids[pd.name] = std::move(lipid);
        ++i;
    }
    lipidAverageSize /= lipidContents.size();
    lipidAverageSize /= _details.lipidDensity;

    _nbOccurrences = _shape->getSurface() / lipidAverageSize *
                     2.0; // WHY DO I HAVE TO DOUBLE IT!

    _processInstances();

    // Add models to the scene
    for (size_t i = 0; i < lipidContents.size(); ++i)
        _scene.addModel(
            _lipids[_getElementNameFromId(i)]->getModelDescriptor());
}

Membrane::~Membrane()
{
    for (const auto& lipid : _lipids)
        _scene.removeModel(lipid.second->getModelDescriptor()->getModelID());
}

void Membrane::_processInstances()
{
    const auto rotation = doublesToQuaterniond(_details.lipidRotation);
    const auto animationDetails =
        doublesToAnimationDetails(_details.animationParams);
    srand(animationDetails.seed);

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
            assemblyTransformation.setTranslation(_assemblyPosition);
            assemblyTransformation.setRotation(_assemblyRotation);
            transformations.push_back(assemblyTransformation);

            const auto shapeTransformation =
                _shape->getTransformation(occurrence, _nbOccurrences,
                                          animationDetails);
            transformations.push_back(shapeTransformation);

            Transformation lipidTransformation;
            lipidTransformation.setRotation(rotation);
            transformations.push_back(lipidTransformation);

            const Transformation finalTransformation =
                combineTransformations(transformations);
            const auto& finalTranslation = finalTransformation.getTranslation();
            const auto& finalRotation = finalTransformation.getRotation();

            // Collision with trans-membrane proteins
            bool collision = false;
            for (const auto& protein : _transmembraneProteins)
            {
                const auto transMembraneRadius =
                    protein.second->getTransMembraneRadius();
                const auto transMembraneOffset =
                    protein.second->getTransMembraneOffset();
                auto modelDescriptor = protein.second->getModelDescriptor();
                const auto& instances = modelDescriptor->getInstances();
                const auto& instanceSize =
                    modelDescriptor->getModel().getBounds().getSize();
                for (const auto& instance : instances)
                {
                    const auto& tf = instance.getTransformation();
                    const Vector3d proteinBase =
                        finalTranslation +
                        transMembraneOffset *
                            normalize(finalRotation * UP_VECTOR);
                    if (length(finalTranslation - tf.getTranslation()) <
                        transMembraneRadius)
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

} // namespace biology
} // namespace bioexplorer
