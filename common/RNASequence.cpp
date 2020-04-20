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

#include <common/log.h>
#include <common/utils.h>

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Scene.h>

#include <fstream>

struct Codon
{
    size_t index;
    std::string name;
    brayns::Vector3f defaultColor;
};
typedef std::map<char, Codon> CondonMap;

CondonMap codonMap{{'A', {0, "Adenine", {0.f, 0.f, 1.f}}},
                   {'U', {1, "Uracile", {0.f, 1.f, 0.f}}},
                   {'G', {2, "Guanine", {1.f, 0.f, 0.f}}},
                   {'C', {3, "Cytosine", {1.f, 1.f, 0.f}}}};

RNASequence::RNASequence(brayns::Scene& scene, const std::string& name,
                         const std::string& filename, const RNAShape shape,
                         const float assemblyRadius, const float radius,
                         const brayns::Vector2f& range = {0.f, 2.f * M_PI},
                         const brayns::Vector3f& params = {1.f, 1.f, 1.f})
{
    std::ifstream file(filename.c_str());
    if (!file.is_open())
        throw std::runtime_error("Could not open " + filename);

    std::vector<size_t> sequenceIndices;
    std::string sequence;
    while (file.good())
    {
        std::string line;
        std::getline(file, line);
        std::string n = line.substr(0, 6);
        n = trim(n);
        std::string s = line.substr(6);
        s = trim(s);
        const size_t index = sequence.length() + s.length();
        sequenceIndices.push_back(index);
        sequence += s;
        _rnaSequenceMap[n] = s;
    }
    file.close();
    const size_t nbElements = sequence.length();

    auto model = scene.createModel();

    size_t materialId = 0;
    for (const auto rnaSequence : _rnaSequenceMap)
        for (const auto& codon : codonMap)
        {
            auto material =
                model->createMaterial(materialId, codon.second.name);
            material->setDiffuseColor(codon.second.defaultColor);
            ++materialId;
        }

    const size_t nbMaterials = materialId;
    PLUGIN_INFO << "Sequence total length: " << nbElements << std::endl;
    PLUGIN_INFO << "Created " << nbMaterials << " materials for "
                << sequenceIndices.size() << " RNA sequences" << std::endl;

    brayns::Vector3f U{range.x, range.y, nbElements};
    brayns::Vector3f V{range.x, range.y, nbElements};

    switch (shape)
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

    size_t sequenceCount = 0;
    size_t elementId = 0;
    for (float v(V.x); v < V.y; v += vStep)
    {
        for (float u(U.x); u < U.y; u += uStep)
        {
            if (elementId >= sequence.length())
                break;

            brayns::Vector3f p0, p1;
            switch (shape)
            {
            case RNAShape::moebius:
            {
                p0 = _moebius(assemblyRadius, u, v);
                p1 = _moebius(assemblyRadius, u + uStep, v);
                break;
            }
            case RNAShape::torus:
            {
                p0 = _torus(assemblyRadius, u, params);
                p1 = _torus(assemblyRadius, u + uStep, params);
                break;
            }
            case RNAShape::star:
            {
                p0 = _star(assemblyRadius, u);
                p1 = _star(assemblyRadius, u + uStep);
                break;
            }
            case RNAShape::spring:
            {
                p0 = _spring(assemblyRadius, u);
                p1 = _spring(assemblyRadius, u + uStep);
                break;
            }
            case RNAShape::trefoilKnot:
            {
                p0 = _trefoilKnot(assemblyRadius, u, params);
                p1 = _trefoilKnot(assemblyRadius, u + uStep, params);
                break;
            }
            case RNAShape::heart:
            {
                p0 = _heart(assemblyRadius, u);
                p1 = _heart(assemblyRadius, u + uStep);
                break;
            }
            case RNAShape::thing:
            {
                p0 = _thing(assemblyRadius, u, params);
                p1 = _thing(assemblyRadius, u + uStep, params);
                break;
            }
            default:
                PLUGIN_THROW(std::runtime_error("Undefined shape"));
                break;
            }

            const char letter = sequence[elementId];
            const auto& codon = codonMap[letter];
            const size_t materialId =
                sequenceCount * codonMap.size() + codon.index;

            if (materialId >= nbMaterials)
                PLUGIN_THROW(std::runtime_error("Invalid material Id: " +
                                                std::to_string(materialId)));
            model->addCylinder(materialId, {{p0.x, p0.y, p0.z},
                                            {p1.x, p1.y, p1.z},
                                            radius});
            //            model->addSphere(materialId, {{p1.x, p1.y, p1.z},
            //            radius});
            ++elementId;
            if (elementId > sequenceIndices[sequenceCount])
                ++sequenceCount;
        }
    }

    // Metadata
    brayns::ModelMetadata metadata;
    for (const auto& s : _rnaSequenceMap)
        metadata[s.first] = s.second;

    _modelDescriptor =
        std::make_shared<brayns::ModelDescriptor>(std::move(model), name,
                                                  filename, metadata);
}

brayns::Vector3f RNASequence::_trefoilKnot(const float radius, const float t,
                                           const brayns::Vector3f& params)
{
    return {radius * (sin(t) + 2.f * sin(params.x * t)),
            radius * (cos(t) - 2.f * cos(params.y * t)),
            radius * (-sin(params.z * t))};
}

brayns::Vector3f RNASequence::_torus(const float radius, const float t,
                                     const brayns::Vector3f& params)
{
    return {radius * (cos(t) + params.x * cos(params.y * t) * cos(t)),
            radius * (sin(t) + params.x * cos(params.y * t) * sin(t)),
            radius * params.x * sin(params.y * t)};
}

brayns::Vector3f RNASequence::_star(const float radius, const float t)
{
    return {radius * (2.f * sin(3.f * t) * cos(t)),
            radius * (2.f * sin(3.f * t) * sin(t)), radius * sin(3.f * t)};
}

brayns::Vector3f RNASequence::_spring(const float radius, const float t)
{
    return {radius * cos(t), radius * sin(t), radius * cos(t)};
}

brayns::Vector3f RNASequence::_heart(const float radius, const float u)
{
    return {radius * 4.f * pow(sin(u), 3.f),
            radius * 0.25f *
                (13.f * cos(u) - 5.f * cos(2.f * u) - 2.f * cos(3.f * u) -
                 cos(4.f * u)),
            0.f};
}

brayns::Vector3f RNASequence::_thing(const float radius, const float t,
                                     const brayns::Vector3f& params)
{
    return {radius * (sin(t) + params.x * sin(params.y * t)),
            radius * (cos(t) - params.x * cos(params.y * t)),
            radius * (-sin(params.z * t))};
}

brayns::Vector3f RNASequence::_moebius(const float radius, const float u,
                                       const float v)
{
    return {4.f * radius * (cos(u) + v * cos(u / 2.f) * cos(u)),
            4.f * radius * (sin(u) + v * cos(u / 2.f) * sin(u)),
            8.f * radius * (v * sin(u / 2.f))};
}
