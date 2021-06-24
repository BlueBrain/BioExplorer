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
#include <plugin/common/Shapes.h>
#include <plugin/common/Utils.h>

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
    const Quaterniond rotation = floatsToQuaterniond(_details.rotation);
    const std::string& sequence = _details.contents;
    const size_t nbElements = sequence.length();
    PLUGIN_INFO("DNA Sequence length: " << nbElements);

    const auto& params = _details.assemblyParams;
    auto randInfo =
        floatsToRandomizationDetails(params, 0,
                                     PositionRandomizationType::radial);

    Vector3f U{details.range[0], details.range[1], nbElements};
    Vector3f V{details.range[0], details.range[1], nbElements};

    switch (details.shape)
    {
    case RNAShape::moebius:
        U = {2.f * M_PI, 4.f * M_PI, nbElements};
        V = {-0.4f, 0.4f, 1.f};
        break;
    case RNAShape::heart:
        U = {0.f, 2.f * M_PI, nbElements};
        V = {0.f, 1.f, 1.f};
        break;
    default:
        break;
    }

    const float uStep = (U.y - U.x) / U.z;
    const float vStep = (V.y - V.x) / V.z;

    if (processAsProtein)
        _buildRNAAsProteinInstances(U, V, rotation, randInfo);
    else
        _buildRNAAsCurve(U, V, rotation, randInfo);
}

void RNASequence::_buildRNAAsCurve(const Vector3f& U, const Vector3f& V,
                                   const Quaterniond& rotation,
                                   const RandomizationDetails& randInfo)
{
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
}

void RNASequence::_buildRNAAsProteinInstances(
    const Vector3f& U, const Vector3f& V, const Quaterniond& rotation,
    const RandomizationDetails& randInfo)
{
    const auto& sequence = _details.contents;
    const size_t nbElements = sequence.length();

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

    const float uStep = (U.y - U.x) / U.z;
    const float vStep = (V.y - V.x) / V.z;

    const auto position = floatsToVector3f(_details.position);

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
                const Vector3f direction = normalize(dst - src);
                const Vector3f normal = cross(UP_VECTOR, direction);
                if (elementId % 50 == 0)
                {
                    Transformation finalTransformation;

                    float upOffset = 0.f;
                    if (randInfo.positionSeed != 0 &&
                        randInfo.randomizationType ==
                            PositionRandomizationType::radial)
                        upOffset =
                            randInfo.positionStrength *
                            rnd3((randInfo.positionSeed + elementId) * 10);

                    const Vector3f translation =
                        Vector3f(_assemblyRotation *
                                 Vector3d(_assemblyPosition)) +
                        position +
                        Vector3f(_assemblyRotation * rotation *
                                 Vector3d(src + normal * upOffset));
                    finalTransformation.setTranslation(translation);

                    Quaterniond instanceRotation =
                        glm::quatLookAt(normal, UP_VECTOR);
                    if (randInfo.rotationSeed != 0)
                        instanceRotation =
                            weightedRandomRotation(randInfo.rotationSeed,
                                                   elementId, instanceRotation,
                                                   randInfo.rotationStrength);
                    finalTransformation.setRotation(rotation *
                                                    instanceRotation);

                    if (elementId == 0)
                        _modelDescriptor->setTransformation(
                            finalTransformation);
                    const ModelInstance instance(true, false,
                                                 finalTransformation);
                    _modelDescriptor->addInstance(instance);
                }
            }

            if (elementId >= nbElements)
                break;
            ++elementId;
        }
    }
}

void RNASequence::_getSegment(const float u, const float v, const float uStep,
                              Vector3f& src, Vector3f& dst)
{
    const auto radius = _details.assemblyParams[0];
    const auto params = floatsToVector3f(_details.params);
    switch (_details.shape)
    {
    case RNAShape::moebius:
    {
        src = _moebius(radius, u, v);
        dst = _moebius(radius, u + uStep, v);
        break;
    }
    case RNAShape::torus:
    {
        src = _torus(radius, u, params);
        dst = _torus(radius, u + uStep, params);
        break;
    }
    case RNAShape::star:
    {
        src = _star(radius, u);
        dst = _star(radius, u + uStep);
        break;
    }
    case RNAShape::spring:
    {
        src = _spring(radius, u, params);
        dst = _spring(radius, u + uStep, params);
        break;
    }
    case RNAShape::trefoilKnot:
    {
        src = _trefoilKnot(radius, u, params);
        dst = _trefoilKnot(radius, u + uStep, params);
        break;
    }
    case RNAShape::heart:
    {
        src = _heart(radius, u);
        dst = _heart(radius, u + uStep);
        break;
    }
    case RNAShape::thing:
    {
        src = _thing(radius, u, params);
        dst = _thing(radius, u + uStep, params);
        break;
    }
    default:
        PLUGIN_THROW("Undefined shape");
        break;
    }
}

Vector3f RNASequence::_trefoilKnot(const float radius, const float t,
                                   const Vector3f& params)
{
    return {radius * ((sin(t) + 2.f * sin(params.x * t))) / 3.f,
            radius * ((cos(t) - 2.f * cos(params.y * t))) / 3.f,
            radius * (-sin(params.z * t))};
}

Vector3f RNASequence::_torus(const float radius, const float t,
                             const Vector3f& params)
{
    return {radius * (cos(t) + params.x * cos(params.y * t) * cos(t)),
            radius * (sin(t) + params.x * cos(params.y * t) * sin(t)),
            radius * params.x * sin(params.y * t)};
}

Vector3f RNASequence::_star(const float radius, const float t)
{
    return {radius * (2.f * sin(3.f * t) * cos(t)),
            radius * (2.f * sin(3.f * t) * sin(t)), radius * sin(3.f * t)};
}

Vector3f RNASequence::_spring(const float radius, const float t,
                              const Vector3f& params)
{
    return {radius * cos(t) + (radius + params.x * cos(params.y * t)) * cos(t),
            radius * sin(t) + (radius + params.x * cos(params.y * t)) * sin(t),
            radius * t + params.x * sin(params.y * t)};
}

Vector3f RNASequence::_heart(const float radius, const float u)
{
    return {radius * 4.f * pow(sin(u), 3.f),
            radius * 0.25f *
                (13.f * cos(u) - 5.f * cos(2.f * u) - 2.f * cos(3.f * u) -
                 cos(4.f * u)),
            0.f};
}

Vector3f RNASequence::_thing(const float radius, const float t,
                             const Vector3f& params)
{
    return {radius * (sin(t) + params.x * sin(params.y * t)),
            radius * (cos(t) - params.x * cos(params.y * t)),
            radius * (-sin(params.z * t))};
}

Vector3f RNASequence::_moebius(const float radius, const float u, const float v)
{
    return {4.f * radius * (cos(u) + v * cos(u / 2.f) * cos(u)),
            4.f * radius * (sin(u) + v * cos(u / 2.f) * sin(u)),
            8.f * radius * (v * sin(u / 2.f))};
}
} // namespace biology
} // namespace bioexplorer
