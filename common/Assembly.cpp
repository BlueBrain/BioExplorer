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
#include <common/Membrane.h>
#include <common/Mesh.h>
#include <common/Protein.h>
#include <common/RNASequence.h>
#include <common/log.h>
#include <common/utils.h>

namespace bioexplorer
{
Assembly::Assembly(Scene &scene, const AssemblyDescriptor &ad)
    : _scene(scene)
    , _halfStructure(ad.halfStructure)
{
    if (ad.position.size() != 3)
        throw std::runtime_error(
            "Position must be a sequence of 3 float values");
    _position = {ad.position[0], ad.position[1], ad.position[2]};

    if (ad.clippingPlanes.size() % 4 != 0)
        throw std::runtime_error(
            "Clipping planes must be defined by 4 float values");
    const auto &cp = ad.clippingPlanes;
    for (size_t i = 0; i < cp.size(); i += 4)
        _clippingPlanes.push_back({cp[i], cp[i + 1], cp[i + 2], cp[i + 3]});

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

    const Quaterniond orientation = {pd.orientation[0], pd.orientation[1],
                                     pd.orientation[2], pd.orientation[3]};
    _processInstances(modelDescriptor, pd.name, pd.assemblyRadius,
                      pd.occurrences, pd.randomSeed, orientation,
                      pd.positionRandomizationType, pd.locationCutoffAngle);

    _proteins[pd.name] = std::move(protein);
    _scene.addModel(modelDescriptor);
}

void Assembly::addMembrane(const MembraneDescriptor &md)
{
    if (_membrane != nullptr)
        throw std::runtime_error("Assembly already has a membrane");

    MembranePtr membrane(
        new Membrane(_scene, md, _clippingPlanes, _occupiedDirections));
    _membrane = std::move(membrane);
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
    targetProtein->getGlycosilationSites(positions, rotations, gd.siteIndices);
    const auto pd = targetProtein->getDescriptor();

    if (positions.empty())
        throw std::runtime_error("No glycosylation site was found on " +
                                 gd.proteinName);

    // Create glycans and attach them to the glycosylation sites of the target
    // protein
    GlycansPtr glycans(new Glycans(_scene, gd, positions, rotations));
    auto modelDescriptor = glycans->getModelDescriptor();
    const Quaterniond orientation = {pd.orientation[0], pd.orientation[1],
                                     pd.orientation[2], pd.orientation[3]};
    _processInstances(modelDescriptor, pd.name, pd.assemblyRadius,
                      pd.occurrences, pd.randomSeed, orientation,
                      PositionRandomizationType::circular, 0.f);

    _glycans[gd.name] = std::move(glycans);
    _scene.addModel(modelDescriptor);
}

void Assembly::addMesh(const MeshDescriptor &md)
{
    MeshPtr mesh(new Mesh(_scene, md));
    auto modelDescriptor = mesh->getModelDescriptor();

    const Quaterniond orientation = {md.orientation[0], md.orientation[1],
                                     md.orientation[2], md.orientation[3]};

    _processInstances(modelDescriptor, md.name, md.assemblyRadius,
                      md.occurrences, md.randomSeed, orientation,
                      md.positionRandomizationType, md.locationCutoffAngle);

    _meshes[md.name] = std::move(mesh);
    _scene.addModel(modelDescriptor);
}

void Assembly::_processInstances(
    ModelDescriptorPtr md, const std::string &name, const float assemblyRadius,
    const size_t occurrences, const size_t randomSeed,
    const Quaterniond &orientation,
    const PositionRandomizationType &randomizationType,
    const float locationCutoffAngle)
{
    const auto &model = md->getModel();
    const auto &bounds = model.getBounds();
    const Vector3f &center = bounds.getCenter();

    const float offset = 2.f / occurrences;
    const float increment = M_PI * (3.f - sqrt(5.f));

    srand(randomSeed);
    size_t rnd{1};
    if (randomSeed != 0 &&
        randomizationType == PositionRandomizationType::circular)
        rnd = rand() % occurrences;

    size_t instanceCount = 0;
    for (size_t i = 0; i < occurrences; ++i)
    {
        // Randomizer
        float radius = assemblyRadius;
        if (randomSeed != 0 &&
            randomizationType == PositionRandomizationType::radial)
            radius *= 1.f + (float(rand() % 20) / 1000.f);

        // Sphere filling
        const float y = ((i * offset) - 1.f) + (offset / 2.f);
        const float r = sqrt(1.f - pow(y, 2.f));
        const float phi = ((i + rnd) % occurrences) * increment;
        const float x = cos(phi) * r;
        const float z = sin(phi) * r;
        const Vector3f direction = {x, y, z};

        // Clipping planes
        const Vector3f position = radius * direction;
        if (isClipped(position, _clippingPlanes))
            continue;

        // Half structure
        if (_halfStructure &&
            (direction.x > 0.f && direction.y > 0.f && direction.z > 0.f))
            continue;

        // Remove membrane where proteins are. This is currently done
        // according to the vector orientation
        bool occupied{false};
        if (locationCutoffAngle != 0.f)
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

        if (instanceCount == 0)
            md->setTransformation(tf);
        const ModelInstance instance(true, false, tf);
        md->addInstance(instance);

        _transformations[name].push_back(tf);

        // Store occupied direction
        if (locationCutoffAngle != 0.f)
            _occupiedDirections.push_back({direction, locationCutoffAngle});

        ++instanceCount;
    }
}

void Assembly::applyTransformations(const AssemblyTransformationsDescriptor &at)
{
    if (at.transformations.size() % 13 != 0)
        throw std::runtime_error(
            "Invalid number of floats in the list of transformations");

    ModelDescriptorPtr modelDescriptor{nullptr};

    auto it = _proteins.find(at.name);
    if (it != _proteins.end())
        modelDescriptor = (*it).second->getModelDescriptor();
    else
    {
        auto it = _meshes.find(at.name);
        if (it != _meshes.end())
            modelDescriptor = (*it).second->getModelDescriptor();
        else
        {
            auto it = _glycans.find(at.name);
            if (it != _glycans.end())
                modelDescriptor = (*it).second->getModelDescriptor();
            else
                throw std::runtime_error("Element " + at.name +
                                         " is not registered in assembly " +
                                         at.assemblyName);
        }
    }

    std::vector<Transformation> transformations;
    const auto &tfs = at.transformations;
    for (size_t i = 0; i < tfs.size(); i += 13)
    {
        Transformation tf;
        tf.setTranslation({tfs[i], tfs[i + 1], tfs[i + 2]});
        tf.setRotationCenter({tfs[i + 3], tfs[i + 4], tfs[i + 5]});
        tf.setRotation({tfs[i + 6], tfs[i + 7], tfs[i + 8], tfs[i + 9]});
        tf.setScale({tfs[i + 10], tfs[i + 11], tfs[i + 12]});
        transformations.push_back(tf);
    }

    const auto nbInstances = modelDescriptor->getInstances().size();
    const auto &initialTransformations = _transformations[at.name];
    for (size_t i = 0; i < transformations.size() && i < nbInstances; ++i)
    {
        const auto &otf = transformations[i];
        const auto &itf = initialTransformations[i];

        Transformation tf;
        tf.setTranslation(itf.getTranslation() + otf.getTranslation());
        tf.setRotationCenter(itf.getRotationCenter() + otf.getRotationCenter());
        tf.setRotation(itf.getRotation() * otf.getRotation());
        tf.setScale(itf.getScale() * otf.getScale());

        auto instance = modelDescriptor->getInstance(i);
        if (i == 0)
            modelDescriptor->setTransformation(tf);
        instance->setTransformation(tf);
    }
    _scene.markModified();
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

void Assembly::setAminoAcidSequenceAsString(
    const AminoAcidSequenceAsStringDescriptor &aasd)
{
    auto it = _proteins.find(aasd.name);
    if (it != _proteins.end())
        (*it).second->setAminoAcidSequenceAsString(aasd.sequence);
    else
        throw std::runtime_error("Protein not found: " + aasd.name);
}

void Assembly::setAminoAcidSequenceAsRange(
    const AminoAcidSequenceAsRangeDescriptor &aasd)
{
    auto it = _proteins.find(aasd.name);
    if (it != _proteins.end())
        (*it).second->setAminoAcidSequenceAsRange(
            {aasd.range[0], aasd.range[1]});
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
    const Vector2f range{rnad.range[0], rnad.range[1]};

    if (rnad.params.size() != 3)
        throw std::runtime_error("Invalid params");

    const Vector3f params{rnad.params[0], rnad.params[1], rnad.params[2]};

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
} // namespace bioexplorer
