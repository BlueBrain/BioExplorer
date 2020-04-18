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

CondonMap codonMap{{'a', {0, "Adenine", {0.f, 0.f, 1.f}}},
                   {'u', {1, "Uracile", {0.f, 1.f, 0.f}}},
                   {'g', {2, "Guanine", {1.f, 0.f, 0.f}}},
                   {'c', {3, "cytosine", {1.f, 1.f, 0.f}}}};

RNASequence::RNASequence(brayns::Scene& scene, const std::string& filename,
                         const RNAShape shape, const float assemblyRadius,
                         const float radius)
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
        sequenceIndices.push_back(sequence.length() + line.length());
        sequence += line;
    }
    file.close();

    auto model = scene.createModel();

    size_t materialId = 0;
    for (const auto sequenceIndex : sequenceIndices)
        for (const auto& codon : codonMap)
        {
            auto material =
                model->createMaterial(materialId, codon.second.name);
            material->setDiffuseColor(codon.second.defaultColor);
            ++materialId;
        }

    const size_t nbElements = sequence.length();

    PLUGIN_INFO << "Sequence length: " << nbElements << std::endl;

    brayns::Vector3f U, V;
    switch (shape)
    {
    case RNAShape::moebius:
        U.x = 2.f * static_cast<float>(M_PI);
        U.y = 4.f * static_cast<float>(M_PI);
        U.z = nbElements;
        V.x = -0.4f;
        V.y = 0.4f;
        V.z = 1.f;
        break;
    case RNAShape::heart:
        U.x = 0.f;
        U.y = 2.f * static_cast<float>(M_PI);
        U.z = nbElements;
        V.x = 0.f;
        V.y = 1.f;
        V.z = 1.f;
        break;
    default:
        U.x = 0.f;
        U.y = 2.f * static_cast<float>(M_PI);
        U.z = nbElements;
        V.x = 0.f;
        V.y = 1.f;
        V.z = 1.f;
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
                p0 = _torus(assemblyRadius, u);
                p1 = _torus(assemblyRadius, u + uStep);
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
                p0 = _trefoilKnot(assemblyRadius, u);
                p1 = _trefoilKnot(assemblyRadius, u + uStep);
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
                const brayns::Vector3f a{1.f + rand() % 4, 1.f + rand() % 4,
                                         1.f + rand() % 4};
                p0 = _thing(assemblyRadius, u, a);
                p1 = _thing(assemblyRadius, u + uStep, a);
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
            model->addCylinder(materialId, {{p0.x, p0.y, p0.z},
                                            {p1.x, p1.y, p1.z},
                                            radius});
            model->addSphere(materialId, {{p1.x, p1.y, p1.z}, radius});
            ++elementId;
            if (elementId >= sequenceIndices[sequenceCount])
                ++sequenceCount;
        }
    }

    _modelDescriptor =
        std::make_shared<brayns::ModelDescriptor>(std::move(model), filename);
}

brayns::Vector3f RNASequence::_trefoilKnot(float R, float t)
{
    return {R * (sin(t) + 2.f * sin(2.f * t)),
            R * (cos(t) - 2.f * cos(2.f * t)), R * (-sin(3.f * t))};
}

brayns::Vector3f RNASequence::_torus(float R, float t)
{
    return {R * (3.f * cos(t) + cos(10.f * t) * cos(t)),
            R * (3.f * sin(t) + cos(10.f * t) * sin(t)), R * sin(10.f * t)};
}

brayns::Vector3f RNASequence::_star(float R, float t)
{
    return {R * (2.f * sin(3.f * t) * cos(t)),
            R * (2.f * sin(3.f * t) * sin(t)), R * sin(3.f * t)};
}

brayns::Vector3f RNASequence::_spring(float R, float t)
{
    return {R * cos(t), R * sin(t), R * cos(t)};
}

brayns::Vector3f RNASequence::_heart(float R, float u)
{
    return {R * 4.f * pow(sin(u), 3.f),
            R * 0.25f *
                (13 * cos(u) - 5 * cos(2.f * u) - 2.f * cos(3.f * u) -
                 cos(4.f * u)),
            0.f};
}

brayns::Vector3f RNASequence::_thing(float R, float t,
                                     const brayns::Vector3f& a)
{
    return {R * (sin(t) + a.x * sin(a.y * t)),
            R * (cos(t) - a.x * cos(a.y * t)), R * (-sin(a.z * t))};
}

brayns::Vector3f RNASequence::_moebius(float R, float u, float v)
{
    return {4.f * R * (cos(u) + v * cos(u / 2.f) * cos(u)),
            4.f * R * (sin(u) + v * cos(u / 2.f) * sin(u)),
            8.f * R * (v * sin(u / 2.f))};
}
