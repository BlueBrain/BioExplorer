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

#include "Assembly.h"

#include <common/Glycans.h>
#include <common/Mesh.h>
#include <common/Protein.h>
#include <common/RNASequence.h>
#include <common/log.h>

Assembly::Assembly(brayns::Scene &scene, const AssemblyDescriptor &ad)
    : _scene(scene)
    , _halfStructure(ad.halfStructure)
{
    if (ad.position.size() != 3)
        throw std::runtime_error(
            "Position must be a sequence of 3 float values");
    _position = {ad.position[0], ad.position[1], ad.position[2]};
    PLUGIN_INFO << "Add assembly " << ad.name << " at position " << _position
                << (ad.halfStructure ? " (half structure only)" : "")
                << std::endl;
}

Assembly::~Assembly()
{
    for (const auto &protein : _proteins)
        _scene.removeModel(protein.second->getModelDescriptor()->getModelID());
    for (const auto &mesh : _meshes)
        _scene.removeModel(mesh.second->getModelDescriptor()->getModelID());
}

void Assembly::addProtein(const ProteinDescriptor &pd)
{
    ProteinPtr protein(new Protein(_scene, pd));
    auto modelDescriptor = protein->getModelDescriptor();

    const brayns::Quaterniond orientation = {pd.orientation[0],
                                             pd.orientation[1],
                                             pd.orientation[2],
                                             pd.orientation[3]};
    _processInstances(modelDescriptor, pd.assemblyRadius, pd.occurrences,
                      pd.randomSeed, orientation, ModelContentType::pdb,
                      pd.locationCutoffAngle);

    _proteins[pd.name] = std::move(protein);
    _scene.addModel(modelDescriptor);
}

void Assembly::addGlycans(const GlycansDescriptor &gd)
{
    // Get information from target protein (attributes, number of instances,
    // glycosylation sites, etc)
    const auto it = _proteins.find(gd.proteinName);
    if (it == _proteins.end())
        throw std::runtime_error("Target protein " + gd.proteinName +
                                 " not registered");

    const auto targetProtein = (*it).second;

    Vector3fs positions;
    Quaternions rotations;
    targetProtein->getGlycosilationSites(positions, rotations);
    const auto pd = targetProtein->getDescriptor();

    if (positions.empty())
        throw std::runtime_error("No glycosylation site was found on " +
                                 gd.proteinName);

    // Create glycans and attach them to the glycosylation sites of the target
    // protein
    GlycansPtr glycans(new Glycans(_scene, gd, positions, rotations));
    auto modelDescriptor = glycans->getModelDescriptor();
    const brayns::Quaterniond orientation = {pd.orientation[0],
                                             pd.orientation[1],
                                             pd.orientation[2],
                                             pd.orientation[3]};
    _processInstances(modelDescriptor, pd.assemblyRadius, pd.occurrences,
                      pd.randomSeed, orientation, ModelContentType::pdb,
                      pd.locationCutoffAngle);

    _glycans[gd.name] = std::move(glycans);
    _scene.addModel(modelDescriptor);
}

void Assembly::addMesh(const MeshDescriptor &md)
{
    MeshPtr mesh(new Mesh(_scene, md));
    auto modelDescriptor = mesh->getModelDescriptor();

    const brayns::Quaterniond orientation = {md.orientation[0],
                                             md.orientation[1],
                                             md.orientation[2],
                                             md.orientation[3]};

    _processInstances(modelDescriptor, md.assemblyRadius, md.occurrences,
                      md.randomSeed, orientation, ModelContentType::obj);

    _meshes[md.name] = std::move(mesh);
    _scene.addModel(modelDescriptor);
}

void Assembly::_processInstances(brayns::ModelDescriptorPtr md,
                                 const float assemblyRadius,
                                 const size_t occurrences,
                                 const size_t randomSeed,
                                 const brayns::Quaterniond &orientation,
                                 const ModelContentType &modelType,
                                 const float locationCutoffAngle)
{
    const auto &model = md->getModel();
    const auto &bounds = model.getBounds();
    const brayns::Vector3f &center = bounds.getCenter();

    const float offset = 2.f / occurrences;
    const float increment = M_PI * (3.f - sqrt(5.f));

    srand(randomSeed);
    size_t rnd{1};
    if (randomSeed != 0 && modelType == ModelContentType::pdb)
        rnd = rand() % occurrences;

    size_t instanceCount = 0;
    for (size_t i = 0; i < occurrences; ++i)
    {
        // Randomizer
        float radius = assemblyRadius;
        if (randomSeed != 0 && modelType == ModelContentType::obj)
            radius *= 1.f + (float(rand() % 20) / 1000.f);

        // Sphere filling
        const float y = ((i * offset) - 1.f) + (offset / 2.f);
        const float r = sqrt(1.f - pow(y, 2.f));
        const float phi = ((i + rnd) % occurrences) * increment;
        const float x = cos(phi) * r;
        const float z = sin(phi) * r;
        const brayns::Vector3f direction = {x, y, z};

        // Remove membrane where proteins are. This is currently done
        // according to the vector orientation
        bool occupied{false};
        if (modelType != ModelContentType::pdb)
            for (const auto &occupiedDirection : _occupiedDirections)
                if (dot(direction, occupiedDirection.first) >
                    occupiedDirection.second)
                {
                    occupied = true;
                    break;
                }
        if (occupied)
            continue;

        // Half structure
        if (_halfStructure &&
            (direction.x > 0.f && direction.y > 0.f && direction.z > 0.f))
            continue;

        // Final transformation
        brayns::Transformation tf;
        const brayns::Vector3f position = radius * direction;
        tf.setTranslation(_position + position - center);
        tf.setRotationCenter(_position + center);

        brayns::Quaterniond assemblyOrientation =
            glm::quatLookAt(direction, {0.f, 1.f, 0.f});

        tf.setRotation(assemblyOrientation * orientation);

        if (instanceCount == 0)
            md->setTransformation(tf);
        const brayns::ModelInstance instance(true, false, tf);
        md->addInstance(instance);
        ++instanceCount;

        // Store occupied direction
        if (modelType == ModelContentType::pdb)
            _occupiedDirections.push_back({direction, locationCutoffAngle});
    }
}

void Assembly::setColorScheme(const ColorSchemeDescriptor &csd)
{
    auto it = _proteins.find(csd.name);
    if (it != _proteins.end())
    {
        Palette palette;
        for (size_t i = 0; i < csd.palette.size(); i += 3)
            palette.push_back(
                {csd.palette[i], csd.palette[i + 1], csd.palette[i + 2]});

        (*it).second->setColorScheme(csd.colorScheme, palette);
    }
}

void Assembly::setAminoAcidSequence(const AminoAcidSequenceDescriptor &aasd)
{
    auto it = _proteins.find(aasd.name);
    if (it != _proteins.end())
        (*it).second->setAminoAcidSequence(aasd.aminoAcidSequence);
    else
        throw std::runtime_error("Protein not found: " + aasd.name);
}

std::string Assembly::getAminoAcidSequences(
    const AminoAcidSequencesDescriptor &aasd) const
{
    PLUGIN_INFO << "Returning sequences from protein " << aasd.name
                << std::endl;

    std::string response;
    auto it = _proteins.find(aasd.name);
    if (it != _proteins.end())
    {
        for (const auto &sequence : (*it).second->getSequencesAsString())
        {
            if (!response.empty())
                response += "\n";
            response += sequence.second;
        }
    }
    else
        throw std::runtime_error("Protein not found: " + aasd.name);
    return response;
}

void Assembly::addRNASequence(const RNASequenceDescriptor &rnad)
{
    if (rnad.range.size() != 2)
        throw std::runtime_error("Invalid range");
    const brayns::Vector2f range{rnad.range[0], rnad.range[1]};

    if (rnad.params.size() != 3)
        throw std::runtime_error("Invalid params");

    const brayns::Vector3f params{rnad.params[0], rnad.params[1],
                                  rnad.params[2]};

    PLUGIN_INFO << "Loading RNA sequence " << rnad.name << " from "
                << rnad.contents << std::endl;
    PLUGIN_INFO << "Assembly radius: " << rnad.assemblyRadius << std::endl;
    PLUGIN_INFO << "RNA radius     : " << rnad.radius << std::endl;
    PLUGIN_INFO << "Range          : " << range << std::endl;
    PLUGIN_INFO << "Params         : " << params << std::endl;

    RNASequence rnaSequence(_scene, rnad, range, params);
    const auto modelDescriptor = rnaSequence.getModelDescriptor();
    _scene.addModel(modelDescriptor);
}
