/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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

#include "RNASequence.h"
#include "Protein.h"

#include <plugin/common/GeneralSettings.h>
#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>
#include <brayns/engineapi/Scene.h>

namespace bioexplorer
{
namespace molecularsystems
{
using namespace common;

/**
 * @brief Structure representing a nucleotid
 *
 */
struct Nucleotid
{
    /** Index */
    size_t index;
    /** Long name */
    std::string name;
    /** Color */
    Vector3d color;
};
typedef std::map<char, Nucleotid> NucleotidMap;

/**
 * @brief Map of nucleotids indexed by short name
 *
 */
NucleotidMap nucleotidMap{{'A', {0, "Adenine", {0.f, 0.f, 1.f}}},
                          {'U', {1, "Uracile", {0.f, 1.f, 0.f}}},
                          {'G', {2, "Guanine", {1.f, 0.f, 0.f}}},
                          {'T', {3, "Thymine", {1.f, 0.f, 1.f}}},
                          {'C', {4, "Cytosine", {1.f, 1.f, 0.f}}}};

RNASequence::RNASequence(Scene& scene, const RNASequenceDetails& details, const Vector4ds& clippingPlanes,
                         const Vector3d& assemblyPosition, const Quaterniond& assemblyRotation)
    : Node()
    , _scene(scene)
    , _details(details)
    , _assemblyPosition(assemblyPosition)
    , _assemblyRotation(assemblyRotation)
{
    const bool processAsProtein = !_details.proteinContents.empty();
    const auto position = doublesToVector3d(_details.position);
    const auto rotation = doublesToQuaterniond(_details.rotation);
    const std::string& sequence = _details.contents;
    _nbElements = sequence.length();

    const auto shapeParams = doublesToVector2d(_details.shapeParams);
    const auto valuesRange = doublesToVector2d(_details.valuesRange);
    const auto curveParams = doublesToVector3d(_details.curveParams);
    const auto MolecularSystemAnimationDetails = doublesToMolecularSystemAnimationDetails(_details.animationParams);

    PLUGIN_INFO(3, "Loading RNA sequence " << details.name << " from " << details.contents);
    PLUGIN_INFO(3, "- Shape params        : " << shapeParams);
    PLUGIN_INFO(3, "- Values range        : " << valuesRange);
    PLUGIN_INFO(3, "- Curve parameters    : " << curveParams);
    PLUGIN_INFO(3, "- Position            : " << position);
    PLUGIN_INFO(3, "- RNA Sequence length : " << _nbElements);

    _shape =
        RNAShapePtr(new RNAShape(clippingPlanes, _details.shape, _nbElements, shapeParams, valuesRange, curveParams));

    if (processAsProtein)
        _buildRNAAsProteinInstances(rotation);
    else
        _buildRNAAsCurve(rotation);
}

void RNASequence::_buildRNAAsCurve(const Quaterniond& rotation)
{
    const auto& sequence = _details.contents;
    const auto MolecularSystemAnimationDetails = doublesToMolecularSystemAnimationDetails(_details.animationParams);
    const auto shapeParams = doublesToVector2d(_details.shapeParams);
    const auto radius = shapeParams.y;

    auto model = _scene.createModel();

    size_t materialId = 0;
    for (const auto& nucleotid : nucleotidMap)
    {
        auto material = model->createMaterial(materialId, nucleotid.second.name);
        brayns::PropertyMap props;
        props.setProperty(
            {MATERIAL_PROPERTY_CHAMELEON_MODE, static_cast<int>(MaterialChameleonMode::undefined_chameleon_mode)});
        material->setDiffuseColor(nucleotid.second.color);
        material->updateProperties(props);
        ++materialId;
    }
    PLUGIN_INFO(3, "Created " << materialId << " materials");

    const auto occurrences = _nbElements;
    for (uint64_t occurrence = 0; occurrence < occurrences - 1; ++occurrence)
    {
        const char letter = sequence[occurrence];
        if (nucleotidMap.find(letter) != nucleotidMap.end())
        {
            const auto& codon = nucleotidMap[letter];
            const auto materialId = codon.index;

            const auto src = _shape->getTransformation(occurrence, occurrences, MolecularSystemAnimationDetails, 0.f);
            const auto dst =
                _shape->getTransformation(occurrence + 1, occurrences, MolecularSystemAnimationDetails, 0.f);

            model->addCylinder(materialId, {src.getTranslation(), dst.getTranslation(), static_cast<float>(radius)});
        }
    }

    // Metadata
    ModelMetadata metadata;
    metadata[METADATA_ASSEMBLY] = _details.assemblyName;
    metadata["RNA sequence"] = sequence;
    _modelDescriptor = std::make_shared<ModelDescriptor>(std::move(model), _details.name, metadata);
    if (_modelDescriptor && !GeneralSettings::getInstance()->getModelVisibilityOnCreation())
        _modelDescriptor->setVisible(false);
}

void RNASequence::_buildRNAAsProteinInstances(const Quaterniond& rotation)
{
    const auto& sequence = _details.contents;
    const auto MolecularSystemAnimationDetails = doublesToMolecularSystemAnimationDetails(_details.animationParams);
    const size_t nbElements = sequence.length();
    Vector3d position = Vector3d(0.f);

    // Load protein
    ModelPtr model{nullptr};
    const std::string proteinName = _details.assemblyName + "_RNA sequence";
    ProteinDetails pd;
    pd.assemblyName = _details.assemblyName;
    pd.name = proteinName;
    pd.pdbId = _details.pdbId;
    pd.contents = _details.proteinContents;
    pd.recenter = true;
    pd.atomRadiusMultiplier = _details.atomRadiusMultiplier;
    pd.representation = _details.representation;

    _protein = ProteinPtr(new Protein(_scene, pd));
    _modelDescriptor = _protein->getModelDescriptor();

    const auto proteinBounds = _protein->getBounds().getSize();
    const double proteinSize = std::min(proteinBounds.x, std::min(proteinBounds.y, proteinBounds.z));
    double proteinSpacing = 0.f;

    Vector3d previousTranslation;

    const auto occurrences = _nbElements;
    uint64_t nbInstances = 0;
    for (uint64_t occurrence = 0; occurrence < occurrences; ++occurrence)
    {
        try
        {
            Transformations transformations;

            Transformation assemblyTransformation;
            assemblyTransformation.setTranslation(_assemblyPosition);
            assemblyTransformation.setRotation(_assemblyRotation);
            transformations.push_back(assemblyTransformation);

            const auto shapeTransformation =
                _shape->getTransformation(occurrence, occurrences, MolecularSystemAnimationDetails, 0.f);
            transformations.push_back(shapeTransformation);

            const Transformation finalTransformation = combineTransformations(transformations);

            const Vector3d translation = finalTransformation.getTranslation();

            if (nbInstances == 0)
                _modelDescriptor->setTransformation(finalTransformation);
            else
            {
                proteinSpacing += length(previousTranslation - translation);
                if (proteinSpacing < proteinSize)
                    continue;
            }
            previousTranslation = translation;
            ++nbInstances;

            const ModelInstance instance(true, false, finalTransformation);
            _modelDescriptor->addInstance(instance);
            proteinSpacing = 0.f;
        }
        catch (const std::runtime_error&)
        {
            // Instance is clipped
        }
    }
}
} // namespace molecularsystems
} // namespace bioexplorer
