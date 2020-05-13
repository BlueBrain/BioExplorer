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

#include "Membrane.h"
#include "Protein.h"

#include <common/log.h>
#include <common/utils.h>

namespace bioexplorer
{
Membrane::Membrane(Scene &scene, const MembraneDescriptor &descriptor,
                   const Vector4fs &clippingPlanes,
                   const OccupiedDirections &occupiedDirections)
    : _scene(scene)
    , _descriptor(descriptor)
    , _clippingPlanes(clippingPlanes)
    , _occupiedDirections(occupiedDirections)
{
    std::vector<std::string> proteinContents;
    proteinContents.push_back(descriptor.content1);
    if (!descriptor.content2.empty())
        proteinContents.push_back(descriptor.content2);
    if (!descriptor.content3.empty())
        proteinContents.push_back(descriptor.content3);
    if (!descriptor.content4.empty())
        proteinContents.push_back(descriptor.content4);

    // Load proteins
    size_t i = 0;
    for (const auto &content : proteinContents)
    {
        ProteinDescriptor pd;
        pd.assemblyName = descriptor.assemblyName;
        pd.name = std::to_string(i);
        pd.contents = content;
        pd.chainIds = descriptor.chainIds;
        pd.recenter = descriptor.recenter;
        pd.loadBonds = descriptor.loadBonds;
        pd.randomSeed = descriptor.randomSeed;
        pd.occurrences = 1;
        pd.orientation = descriptor.orientation;
        pd.assemblyRadius = descriptor.assemblyRadius;
        pd.atomRadiusMultiplier = descriptor.atomRadiusMultiplier;
        pd.representation = descriptor.representation;
        pd.locationCutoffAngle = descriptor.locationCutoffAngle;
        pd.loadNonPolymerChemicals = descriptor.loadNonPolymerChemicals;
        pd.positionRandomizationType = descriptor.positionRandomizationType;

        ProteinPtr protein(new Protein(_scene, pd));
        auto modelDescriptor = protein->getModelDescriptor();
        _proteins[pd.name] = std::move(protein);
        _scene.addModel(modelDescriptor);
        ++i;
    }

    // Assemble proteins
    _processInstances();
}

Membrane::~Membrane()
{
    for (const auto &protein : _proteins)
        _scene.removeModel(protein.second->getModelDescriptor()->getModelID());
}

void Membrane::_processInstances()
{
    const float offset = 2.f / _descriptor.occurrences;
    const float increment = M_PI * (3.f - sqrt(5.f));

    srand(_descriptor.randomSeed);
    size_t rnd{1};
    if (_descriptor.randomSeed != 0 && _descriptor.positionRandomizationType ==
                                           PositionRandomizationType::circular)
        rnd = rand() % _descriptor.occurrences;

    const Quaterniond orientation = {_descriptor.orientation[0],
                                     _descriptor.orientation[1],
                                     _descriptor.orientation[2],
                                     _descriptor.orientation[3]};

    std::map<size_t, size_t> instanceCounts;
    for (size_t i = 0; i < _proteins.size(); ++i)
        instanceCounts[i] = 0;

    for (size_t i = 0; i < _descriptor.occurrences; ++i)
    {
        const size_t proteinId = rand() % _proteins.size();
        auto md = _proteins[std::to_string(proteinId)]->getModelDescriptor();

        const auto &model = md->getModel();
        const auto &bounds = model.getBounds();
        const Vector3f &center = bounds.getCenter();

        // Randomizer
        float radius = _descriptor.assemblyRadius;
        if (_descriptor.randomSeed != 0 &&
            _descriptor.positionRandomizationType ==
                PositionRandomizationType::radial)
            radius *= 1.f + (float(rand() % 20) / 1000.f);

        // Sphere filling
        const float y = ((i * offset) - 1.f) + (offset / 2.f);
        const float r = sqrt(1.f - pow(y, 2.f));
        const float phi = ((i + rnd) % _descriptor.occurrences) * increment;
        const float x = cos(phi) * r;
        const float z = sin(phi) * r;
        const Vector3f direction = {x, y, z};

        // Clipping planes
        const Vector3f position = radius * direction;
        if (isClipped(position, _clippingPlanes))
            continue;

        // Remove membrane where proteins are. This is currently done
        // according to the vector orientation
        bool occupied{false};
        if (_descriptor.locationCutoffAngle != 0.f)
            for (const auto &occupiedDirection : _occupiedDirections)
                if (dot(direction, occupiedDirection.first) >
                    occupiedDirection.second)
                {
                    occupied = true;
                    break;
                }
        if (occupied)
            continue;

        // Final transformation
        Transformation tf;
        tf.setTranslation(_position + position - center);

        Quaterniond assemblyOrientation =
            glm::quatLookAt(direction, {0.f, 1.f, 0.f});

        tf.setRotation(assemblyOrientation * orientation);

        if (instanceCounts[proteinId] == 0)
            md->setTransformation(tf);
        const ModelInstance instance(true, false, tf);
        md->addInstance(instance);

        // Save initial transformation
        _transformations[_descriptor.name].push_back(tf);

        instanceCounts[proteinId] = instanceCounts[proteinId] + 1;
    }

    for (size_t i = 0; i < _proteins.size(); ++i)
        PLUGIN_INFO << "Instances for " << i << " : " << instanceCounts[i]
                    << std::endl;
}
} // namespace bioexplorer
