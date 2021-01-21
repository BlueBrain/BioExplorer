/* Copyright (c) 2020, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "RNASequence.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Scene.h>

namespace bioexplorer
{
struct Codon
{
    size_t index;
    std::string name;
    Vector3f defaultColor;
};
typedef std::map<char, Codon> CondonMap;

CondonMap nucleotidMap{{'A', {0, "Adenine", {0.f, 0.f, 1.f}}},
                       {'U', {1, "Uracile", {0.f, 1.f, 0.f}}},
                       {'G', {2, "Guanine", {1.f, 0.f, 0.f}}},
                       {'C', {3, "Cytosine", {1.f, 1.f, 0.f}}}};

RNASequence::RNASequence(Scene& scene, const RNASequenceDescriptor& rd,
                         const Vector2f& range = {0.f, 2.f * M_PI},
                         const Vector3f& params = {1.f, 1.f, 1.f},
                         const Vector3f& position = Vector3f())
    : Node()
{
    const auto& sequence = rd.contents;
    const size_t nbElements = sequence.length();

    auto model = scene.createModel();

    size_t materialId = 0;
    for (const auto& nucleotid : nucleotidMap)
    {
        auto material =
            model->createMaterial(materialId, nucleotid.second.name);
        brayns::PropertyMap props;
        props.setProperty({MATERIAL_PROPERTY_SHADING_MODE,
                           static_cast<int>(MaterialShadingMode::basic)});
        props.setProperty({MATERIAL_PROPERTY_USER_PARAMETER, 1.0});
        material->setDiffuseColor(nucleotid.second.defaultColor);
        material->updateProperties(props);
        ++materialId;
    }

    const size_t nbMaterials = materialId;

    PLUGIN_INFO << "Sequence total length: " << nbElements << std::endl;
    PLUGIN_INFO << "Created " << nbMaterials << " materials" << std::endl;

    Vector3f U{range.x, range.y, nbElements};
    Vector3f V{range.x, range.y, nbElements};

    switch (rd.shape)
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

    size_t elementId = 0;
    for (float v(V.x); v < V.y; v += vStep)
    {
        for (float u(U.x); u < U.y; u += uStep)
        {
            Vector3f src;
            Vector3f dst;
            switch (rd.shape)
            {
            case RNAShape::moebius:
            {
                src = _moebius(rd.assemblyParams[0], u, v);
                dst = _moebius(rd.assemblyParams[0], u + uStep, v);
                break;
            }
            case RNAShape::torus:
            {
                src = _torus(rd.assemblyParams[0], u, params);
                dst = _torus(rd.assemblyParams[0], u + uStep, params);
                break;
            }
            case RNAShape::star:
            {
                src = _star(rd.assemblyParams[0], u);
                dst = _star(rd.assemblyParams[0], u + uStep);
                break;
            }
            case RNAShape::spring:
            {
                src = _spring(rd.assemblyParams[0], u);
                dst = _spring(rd.assemblyParams[0], u + uStep);
                break;
            }
            case RNAShape::trefoilKnot:
            {
                src = _trefoilKnot(rd.assemblyParams[0], u, params);
                dst = _trefoilKnot(rd.assemblyParams[0], u + uStep, params);
                break;
            }
            case RNAShape::heart:
            {
                src = _heart(rd.assemblyParams[0], u);
                dst = _heart(rd.assemblyParams[0], u + uStep);
                break;
            }
            case RNAShape::thing:
            {
                src = _thing(rd.assemblyParams[0], u, params);
                dst = _thing(rd.assemblyParams[0], u + uStep, params);
                break;
            }
            default:
                PLUGIN_THROW(std::runtime_error("Undefined shape"));
                break;
            }

            const char letter = sequence[elementId];
            const auto& codon = nucleotidMap[letter];
            const auto materialId = codon.index;
            const auto radius = rd.assemblyParams[1];
            model->addCylinder(materialId,
                               {position + src, position + dst, radius});

            if (elementId == 0)
                model->addSphere(materialId, {position + src, radius});
            if (elementId == nbElements - 1)
                model->addSphere(materialId, {position + dst, radius});

            if (elementId == nbElements)
                break;

            ++elementId;
        }
    }

    // Metadata
    ModelMetadata metadata;
    metadata[METADATA_ASSEMBLY] = rd.assemblyName;
    metadata["RNA sequence"] = sequence;
    _modelDescriptor =
        std::make_shared<ModelDescriptor>(std::move(model), rd.name, metadata);
}

Vector3f RNASequence::_trefoilKnot(const float radius, const float t,
                                   const Vector3f& params)
{
    return {radius * (sin(t) + 2.f * sin(params.x * t)),
            radius * (cos(t) - 2.f * cos(params.y * t)),
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

Vector3f RNASequence::_spring(const float radius, const float t)
{
    return {radius * cos(t), radius * sin(t), radius * cos(t)};
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
} // namespace bioexplorer
