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

#include "RNASequence.h"
#include "Protein.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/GeneralSettings.h>
#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>
#include <plugin/common/shapes/RNAShape.h>

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Scene.h>

namespace bioexplorer
{
namespace biology
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
    Vector3f color;
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

RNASequence::RNASequence(Scene& scene, const RNASequenceDetails& details,
                         const Vector3f& assemblyPosition,
                         const Quaterniond& assemblyRotation)
    : Node()
    , _scene(scene)
    , _details(details)
    , _assemblyPosition(assemblyPosition)
    , _assemblyRotation(assemblyRotation)
{
    const bool processAsProtein = !_details.proteinContents.empty();
    const auto position = floatsToVector3f(_details.position);
    const auto rotation = floatsToQuaterniond(_details.rotation);
    const std::string& sequence = _details.contents;
    _nbElements = sequence.length();

    const auto shapeParams = floatsToVector2f(_details.shapeParams);
    const auto valuesRange = floatsToVector2f(_details.valuesRange);
    const auto curveParams = floatsToVector3f(_details.curveParams);
    const auto randDetails =
        floatsToRandomizationDetails(_details.randomParams);

    PLUGIN_INFO("Loading RNA sequence " << details.name << " from "
                                        << details.contents);
    PLUGIN_INFO("- Shape params        : " << shapeParams);
    PLUGIN_INFO("- Values range        : " << valuesRange);
    PLUGIN_INFO("- Curve parameters    : " << curveParams);
    PLUGIN_INFO("- Position            : " << position);
    PLUGIN_INFO("- RNA Sequence length : " << _nbElements);

    _shape = RNAShapePtr(new RNAShape(Vector4fs(), _details.shape, _nbElements,
                                      shapeParams, valuesRange, curveParams));

    if (processAsProtein)
        _buildRNAAsProteinInstances(rotation);
    else
        _buildRNAAsCurve(rotation);
}

void RNASequence::_buildRNAAsCurve(const Quaterniond& rotation)
{
#if 0
    auto model = _scene.createModel();

    size_t materialId = 0;
    for (const auto& nucleotid : nucleotidMap)
    {
        auto material =
            model->createMaterial(materialId, nucleotid.second.name);
        brayns::PropertyMap props;
        props.setProperty({MATERIAL_PROPERTY_SHADING_MODE,
                           static_cast<int>(MaterialShadingMode::basic)});
        props.setProperty({MATERIAL_PROPERTY_USER_PARAMETER, 1.0});
        props.setProperty(
            {MATERIAL_PROPERTY_CHAMELEON_MODE,
             static_cast<int>(
                 MaterialChameleonMode::undefined_chameleon_mode)});
        material->setDiffuseColor(nucleotid.second.color);
        material->updateProperties(props);
        ++materialId;
    }
    PLUGIN_INFO("Created " << materialId << " materials");

    const float uStep = (U.y - U.x) / U.z;
    const float vStep = (V.y - V.x) / V.z;

    const std::string& sequence = _details.contents;
    const size_t nbElements = sequence.length();
    const auto position = floatsToVector3f(_details.position);
    const auto radius = _details.assemblyParams[5];

    size_t elementId = 0;
    for (float v(V.x); v < V.y; v += vStep)
    {
        for (float u(U.x); u < U.y; u += uStep)
        {
            Vector3f src, dst;
            _getSegment(u, v, uStep, src, dst);

            const char letter = sequence[elementId];
            if (nucleotidMap.find(letter) != nucleotidMap.end())
            {
                const auto& codon = nucleotidMap[letter];
                const auto materialId = codon.index;

                const Vector3f assemblyPosition =
                    Vector3f(_assemblyRotation * Vector3d(_assemblyPosition));
                const Vector3f translationSrc =
                    assemblyPosition + position +
                    Vector3f(_assemblyRotation * rotation * Vector3d(src));
                const Vector3f translationDst =
                    assemblyPosition + position +
                    Vector3f(_assemblyRotation * rotation * Vector3d(dst));

                model->addCylinder(materialId,
                                   {translationSrc, translationDst, radius});
                if (elementId == 0)
                    model->addSphere(materialId, {translationSrc, radius});
                if (elementId == nbElements - 1)
                    model->addSphere(materialId, {translationDst, radius});
            }

            if (elementId >= nbElements)
                break;
            ++elementId;
        }
    }

    // Metadata
    ModelMetadata metadata;
    metadata[METADATA_ASSEMBLY] = _details.assemblyName;
    metadata["RNA sequence"] = sequence;
    _modelDescriptor =
        std::make_shared<ModelDescriptor>(std::move(model), _details.name,
                                          metadata);
    if (_modelDescriptor &&
        !GeneralSettings::getInstance()->getModelVisibilityOnCreation())
        _modelDescriptor->setVisible(false);
#endif
}

void RNASequence::_buildRNAAsProteinInstances(const Quaterniond& rotation)
{
    const auto& sequence = _details.contents;
    const auto randDetails =
        floatsToRandomizationDetails(_details.randomParams);
    const size_t nbElements = sequence.length();
    Vector3f position = Vector3f(0.f);

    // Load protein
    ModelPtr model{nullptr};
    const std::string proteinName = _details.assemblyName + "_RNA sequence";
    ProteinDetails pd;
    pd.assemblyName = _details.assemblyName;
    pd.name = proteinName;
    pd.contents = _details.proteinContents;
    pd.recenter = true;

    _protein = ProteinPtr(new Protein(_scene, pd));
    _modelDescriptor = _protein->getModelDescriptor();

    const auto occurrences = _nbElements;
    for (uint64_t occurrence = 0; occurrence < occurrences; ++occurrence)
    {
        const auto transformation =
            _shape->getTransformation(occurrence, occurrences, randDetails,
                                      0.f);

        const char letter = sequence[occurrence];
        if (nucleotidMap.find(letter) == nucleotidMap.end())
            continue;

        if (occurrence == 0)
            _modelDescriptor->setTransformation(transformation);
        const ModelInstance instance(true, false, transformation);
        _modelDescriptor->addInstance(instance);
    }

    // size_t elementId = 0;
    // for (float v(V.x); v < V.y; v += vStep)
    // {
    //     for (float u(U.x); u < U.y; u += uStep)
    //     {
    //         Vector3f src, dst;
    //         _getSegment(u, v, uStep, src, dst);

    //         const char letter = sequence[elementId];
    //         if (nucleotidMap.find(letter) != nucleotidMap.end())
    //         {
    //             const Vector3f direction = normalize(dst - src);
    //             const Vector3f normal = cross(UP_VECTOR, direction);
    //             if (elementId % 50 == 0)
    //             {
    //                 Transformation finalTransformation;

    //                 float upOffset = 0.f;
    //                 if (randDetails.positionSeed != 0)
    //                     upOffset =
    //                         randDetails.positionStrength *
    //                         rnd3((randDetails.positionSeed + elementId) *
    //                         10);

    //                 const Vector3f translation =
    //                     Vector3f(_assemblyRotation *
    //                              Vector3d(_assemblyPosition)) +
    //                     position +
    //                     Vector3f(_assemblyRotation * rotation *
    //                              Vector3d(src + normal * upOffset));
    //                 finalTransformation.setTranslation(translation);

    //                 Quaterniond instanceRotation =
    //                     glm::quatLookAt(normal, UP_VECTOR);
    //                 if (randDetails.rotationSeed != 0)
    //                     instanceRotation =
    //                         weightedRandomRotation(randDetails.rotationSeed,
    //                                                elementId,
    //                                                instanceRotation,
    //                                                randDetails.rotationStrength);
    //                 finalTransformation.setRotation(rotation *
    //                                                 instanceRotation);

    //                 if (elementId == 0)
    //                     _modelDescriptor->setTransformation(
    //                         finalTransformation);
    //                 const ModelInstance instance(true, false,
    //                                              finalTransformation);
    //                 _modelDescriptor->addInstance(instance);
    //             }
    //         }

    //         if (elementId >= nbElements)
    //             break;
    //         ++elementId;
    //     }
    // }
}

} // namespace biology
} // namespace bioexplorer
