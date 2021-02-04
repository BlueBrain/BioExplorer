/* Copyright (c) 2020-2021, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: cyrille.favreau@epfl.ch
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

#include <plugin/bioexplorer/Glycans.h>
#include <plugin/bioexplorer/Membrane.h>
#include <plugin/bioexplorer/Mesh.h>
#include <plugin/bioexplorer/Protein.h>
#include <plugin/bioexplorer/RNASequence.h>
#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>

namespace bioexplorer
{
Assembly::Assembly(Scene &scene, const AssemblyDescriptor &ad)
    : _descriptor(ad)
    , _scene(scene)
{
    if (ad.position.size() != 3)
        PLUGIN_THROW(std::runtime_error(
            "Position must be a sequence of 3 float values"));
    _position = {ad.position[0], ad.position[1], ad.position[2]};

    if (ad.clippingPlanes.size() % 4 != 0)
        PLUGIN_THROW(std::runtime_error(
            "Clipping planes must be defined by 4 float values"));
    const auto &cp = ad.clippingPlanes;
    for (size_t i = 0; i < cp.size(); i += 4)
        _clippingPlanes.push_back({cp[i], cp[i + 1], cp[i + 2], cp[i + 3]});

    PLUGIN_INFO << "Add assembly " << ad.name << " at position " << _position
                << std::endl;
}

Assembly::~Assembly()
{
    for (const auto &protein : _proteins)
    {
        const auto modelId = protein.second->getModelDescriptor()->getModelID();
        PLUGIN_INFO << "Removing protein <" << modelId << "><" << protein.first
                    << "> from assembly <" << _descriptor.name << ">"
                    << std::endl;
        _scene.removeModel(protein.second->getModelDescriptor()->getModelID());
    }
    for (const auto &mesh : _meshes)
    {
        const auto modelId = mesh.second->getModelDescriptor()->getModelID();
        PLUGIN_INFO << "Removing mesh <" << modelId << "><" << mesh.first
                    << "> from assembly <" << _descriptor.name << ">"
                    << std::endl;
        _scene.removeModel(mesh.second->getModelDescriptor()->getModelID());
    }
    for (const auto &glycan : _glycans)
    {
        const auto modelId = glycan.second->getModelDescriptor()->getModelID();
        PLUGIN_INFO << "Removing glycan <" << modelId << "><" << glycan.first
                    << "> from assembly <" << _descriptor.name << ">"
                    << std::endl;
        _scene.removeModel(modelId);
    }
    if (_rnaSequence)
    {
        const auto modelId = _rnaSequence->getModelDescriptor()->getModelID();
        PLUGIN_INFO << "Removing <" << modelId
                    << "><RNA sequence> from assembly <" << _descriptor.name
                    << ">" << std::endl;
        _scene.removeModel(modelId);
    }
}

void Assembly::addProtein(const ProteinDescriptor &pd)
{
    ProteinPtr protein(new Protein(_scene, pd));
    auto modelDescriptor = protein->getModelDescriptor();

    const Vector3f position = {pd.position[0], pd.position[1], pd.position[2]};
    const Quaterniond orientation = {pd.orientation[0], pd.orientation[1],
                                     pd.orientation[2], pd.orientation[3]};
    _processInstances(modelDescriptor, pd.name, pd.shape, pd.assemblyParams,
                      pd.occurrences, pd.allowedOccurrences, pd.randomSeed,
                      position, orientation, pd.positionRandomizationType,
                      pd.locationCutoffAngle);

    _proteins[pd.name] = std::move(protein);
    _scene.addModel(modelDescriptor);
    PLUGIN_INFO << "Number of instances   : "
                << modelDescriptor->getInstances().size() << std::endl;
}

void Assembly::addMembrane(const MembraneDescriptor &md)
{
    if (_membrane != nullptr)
        PLUGIN_THROW(std::runtime_error("Assembly already has a membrane"));

    MembranePtr membrane(new Membrane(_scene, md, _position, _clippingPlanes,
                                      _occupiedDirections));
    _membrane = std::move(membrane);
}

void Assembly::addGlycans(const SugarsDescriptor &sd)
{
    // Get information from target protein (attributes, number of instances,
    // glycosylation sites, etc)
    const auto it = _proteins.find(sd.proteinName);
    if (it == _proteins.end())
    {
        std::string s;
        for (const auto &protein : _proteins)
            s += "[" + protein.first + "]";
        PLUGIN_THROW(std::runtime_error("Target protein " + sd.proteinName +
                                        " not registered in assembly " +
                                        sd.assemblyName +
                                        ". Registered proteins are " + s));
    }

    PLUGIN_INFO << "Adding glycans to protein " << sd.proteinName << std::endl;
    const auto targetProtein = (*it).second;

    Vector3fs glycanPositions;
    Quaternions glycanOrientations;
    targetProtein->getGlycosilationSites(glycanPositions, glycanOrientations,
                                         sd.siteIndices);
    const auto pd = targetProtein->getDescriptor();

    if (glycanPositions.empty())
        PLUGIN_THROW(std::runtime_error("No glycosylation site was found on " +
                                        sd.proteinName));

    // Create glycans and attach them to the glycosylation sites of the target
    // protein
    GlycansPtr glycans(new Glycans(_scene, sd));
    auto modelDescriptor = glycans->getModelDescriptor();
    const Vector3f proteinPosition = {pd.position[0], pd.position[1],
                                      pd.position[2]};
    const Quaterniond proteinOrientation = {pd.orientation[0],
                                            pd.orientation[1],
                                            pd.orientation[2],
                                            pd.orientation[3]};
    _processInstances(modelDescriptor, pd.name, pd.shape, pd.assemblyParams,
                      pd.occurrences, sd.allowedOccurrences, pd.randomSeed,
                      proteinPosition, proteinOrientation,
                      PositionRandomizationType::circular, 0.f, glycanPositions,
                      glycanOrientations);

    _glycans[sd.name] = std::move(glycans);
    _scene.addModel(modelDescriptor);
}

void Assembly::addSugars(const SugarsDescriptor &sd)
{
    // Get information from target protein (attributes, number of instances,
    // glycosylation sites, etc)
    const auto it = _proteins.find(sd.proteinName);
    if (it == _proteins.end())
    {
        std::string s;
        for (const auto &protein : _proteins)
            s += "[" + protein.first + "]";
        PLUGIN_THROW(std::runtime_error("Target protein " + sd.proteinName +
                                        " not registered in assembly " +
                                        sd.assemblyName +
                                        ". Registered proteins are " + s));
    }

    const auto targetProtein = (*it).second;

    Vector3fs positions;
    Quaternions orientations;
    targetProtein->getSugarBindingSites(positions, orientations, sd.siteIndices,
                                        sd.chainIds);
    const auto pd = targetProtein->getDescriptor();

    const Quaterniond sugarOrientation = {sd.orientation[0], sd.orientation[1],
                                          sd.orientation[2], sd.orientation[3]};
    for (auto &orientation : orientations)
        orientation = orientation * sugarOrientation;

    if (positions.empty())
        PLUGIN_THROW(std::runtime_error("No sugar binding site was found on " +
                                        sd.name));

    PLUGIN_INFO << positions.size() << " sugar sites found on "
                << sd.proteinName << std::endl;

    GlycansPtr glucoses(new Glycans(_scene, sd));
    auto modelDescriptor = glucoses->getModelDescriptor();
    const Vector3f position = {pd.position[0], pd.position[1], pd.position[2]};
    const Quaterniond proteinOrientation = {pd.orientation[0],
                                            pd.orientation[1],
                                            pd.orientation[2],
                                            pd.orientation[3]};
    _processInstances(modelDescriptor, pd.name, pd.shape,
                      {pd.assemblyParams[0] /* * 1.025f */,
                       pd.assemblyParams[1]},
                      pd.occurrences, sd.allowedOccurrences, pd.randomSeed,
                      position, proteinOrientation,
                      PositionRandomizationType::circular, 0.f, positions,
                      orientations);

    _glycans[sd.name] = std::move(glucoses);
    _scene.addModel(modelDescriptor);
}

void Assembly::addMesh(const MeshDescriptor &md)
{
    MeshPtr mesh(new Mesh(_scene, md));
    auto modelDescriptor = mesh->getModelDescriptor();
    _meshes[md.name] = std::move(mesh);
    _scene.addModel(modelDescriptor);
}

void Assembly::_processInstances(
    ModelDescriptorPtr md, const std::string &name, const AssemblyShape shape,
    const floats &assemblyParams, const size_t occurrences,
    const size_ts &allowedOccurrences, const size_t randomSeed,
    const Vector3f &proteinPosition, const Quaterniond &proteinOrientation,
    const PositionRandomizationType &randomizationType,
    const float locationCutoffAngle, const Vector3fs &glycanPositions,
    const Quaternions &glycanOrientations)
{
    const float offset = 2.f / occurrences;
    const float increment = M_PI * (3.f - sqrt(5.f));

    srand(randomSeed);
    size_t rnd{1};
    if (occurrences != 0 && randomSeed != 0 &&
        randomizationType == PositionRandomizationType::circular)
        rnd = rand() % occurrences;

#ifdef DEBUG
    model.addCone(0, {{0, 0, 0}, {0, 0, 1}, 0.1f, 0.f});
#endif

    uint64_t instanceCount = 0;
    for (uint64_t i = 0; i < occurrences; ++i)
    {
        if (!allowedOccurrences.empty() &&
            std::find(allowedOccurrences.begin(), allowedOccurrences.end(),
                      i) == allowedOccurrences.end())
            continue;

        Vector3f pos;
        Vector3f dir;
        switch (shape)
        {
        case AssemblyShape::spherical:
        {
            getSphericalPosition(rnd, assemblyParams[0], assemblyParams[1],
                                 randomizationType, randomSeed, i, occurrences,
                                 proteinPosition, pos, dir);
            break;
        }
        case AssemblyShape::sinusoidal:
        {
            const auto assemblySize = assemblyParams[0];
            const auto assemblyHeight = assemblyParams[1];
            getSinosoidalPosition(assemblySize, assemblyHeight,
                                  randomizationType, randomSeed,
                                  proteinPosition, pos, dir);
            break;
        }
        case AssemblyShape::cubic:
        {
            const auto assemblySize = assemblyParams[0];
            getCubicPosition(assemblySize, proteinPosition, pos, dir);
            break;
        }
        case AssemblyShape::fan:
        {
            const auto assemblyRadius = assemblyParams[0];
            getFanPosition(rnd, assemblyRadius, randomizationType, randomSeed,
                           i, occurrences, proteinPosition, pos, dir);
            break;
        }
        case AssemblyShape::bezier:
        {
            const Vector3fs points = {
                {1, 391, 0},   {25, 411, 0},  {48, 446, 0},  {58, 468, 0},
                {70, 495, 0},  {83, 523, 0},  {110, 535, 0}, {157, 517, 0},
                {181, 506, 0}, {214, 501, 0}, {216, 473, 0}, {204, 456, 0},
                {223, 411, 0}, {241, 382, 0}, {261, 372, 0}, {297, 402, 0},
                {308, 433, 0}, {327, 454, 0}, {355, 454, 0}, {389, 446, 0},
                {406, 433, 0}, {431, 426, 0}, {458, 443, 0}, {478, 466, 0},
                {518, 463, 0}, {559, 464, 0}, {584, 478, 0}, {582, 503, 0},
                {550, 533, 0}, {540, 550, 0}, {540, 574, 0}, {560, 572, 0},
                {599, 575, 0}, {629, 550, 0}, {666, 548, 0}, {696, 548, 0},
                {701, 582, 0}, {701, 614, 0}, {683, 639, 0}, {653, 647, 0},
                {632, 651, 0}, {597, 666, 0}, {570, 701, 0}, {564, 731, 0},
                {559, 770, 0}, {565, 799, 0}, {577, 819, 0}, {611, 820, 0},
                {661, 809, 0}, {683, 787, 0}, {700, 768, 0}, {735, 758, 0},
                {763, 768, 0}, {788, 792, 0}, {780, 820, 0}, {770, 859, 0},
                {740, 882, 0}, {705, 911, 0}, {688, 931, 0}, {646, 973, 0},
                {611, 992, 0}, {585, 1022, 0}};
            const auto assemblySize = assemblyParams[0];
            getBezierPosition(points, assemblySize,
                              float(i) / float(occurrences), pos, dir);
            break;
        }
        default:
            const auto assemblySize = assemblyParams[0];
            getPlanarPosition(assemblySize, randomizationType, randomSeed,
                              proteinPosition, pos, dir);
            break;
        }

        // Clipping planes
        if (isClipped(pos, _clippingPlanes))
            continue;

        // Remove membrane where proteins are. This is currently done according
        // to the vector orientation
        bool occupied{false};
        if (locationCutoffAngle != 0.f)
            for (const auto &occupiedDirection : _occupiedDirections)
                if (dot(dir, occupiedDirection.first) >
                    occupiedDirection.second)
                {
                    occupied = true;
                    break;
                }
        if (occupied)
            continue;

        // Final transformation
        Vector3fs localPositions = glycanPositions;
        Quaternions localOrientations = glycanOrientations;
        if (glycanPositions.empty())
        {
            localPositions.push_back({0.f, 0.f, 0.f});
            localOrientations.push_back({0.f, 0.f, 0.f, 1.f});
        }

        Quaterniond instanceOrientation = glm::quatLookAt(dir, {0.f, 0.f, 1.f});
        for (size_t i = 0; i < localPositions.size(); ++i)
        {
            const auto &localPosition = localPositions[i];
            const auto &localOrientation = localOrientations[i];

            const Vector3f rotatedLocalPosition =
                glm::toMat3(instanceOrientation) *
                glm::toMat3(proteinOrientation) * localPosition;
            const auto translation = _position + pos + rotatedLocalPosition;
            const auto rotation = instanceOrientation * proteinOrientation;

            Transformation tf;
            tf.setTranslation(translation);
            tf.setRotation(rotation);

            if (instanceCount == 0)
                md->setTransformation(tf);
            const ModelInstance instance(true, false, tf);
            md->addInstance(instance);

#if 0
        // Save initial transformation for later use
         _transformations[name].push_back(tf);
#endif

            // Store occupied direction
            if (locationCutoffAngle != 0.f)
                _occupiedDirections.push_back({dir, locationCutoffAngle});

            ++instanceCount;
        }
    }
}

void Assembly::setColorScheme(const ColorSchemeDescriptor &csd)
{
    if (csd.palette.size() < 3 || csd.palette.size() % 3 != 0)
        PLUGIN_THROW(std::runtime_error("Invalid palette size"));

    ProteinPtr protein{nullptr};
    auto itProtein = _proteins.find(csd.name);
    if (itProtein != _proteins.end())
        protein = (*itProtein).second;
    else
    {
        auto itMesh = _meshes.find(csd.name);
        if (itMesh != _meshes.end())
            protein = (*itMesh).second->getProtein();
        else if (_membrane)
        {
            const auto membraneProteins = _membrane->getProteins();
            auto it = membraneProteins.find(csd.name);
            if (it != membraneProteins.end())
                protein = (*it).second;
        }
    }

    if (protein)
    {
        Palette palette;
        for (size_t i = 0; i < csd.palette.size(); i += 3)
            palette.push_back(
                {csd.palette[i], csd.palette[i + 1], csd.palette[i + 2]});

        PLUGIN_INFO << "Applying color scheme to protein " << csd.name
                    << " on assembly " << csd.assemblyName << std::endl;
        protein->setColorScheme(csd.colorScheme, palette, csd.chainIds);

        _scene.markModified();
    }
    else
        PLUGIN_ERROR << "Protein " << csd.name << " not found on assembly "
                     << csd.assemblyName << std::endl;
}

void Assembly::setAminoAcidSequenceAsString(
    const AminoAcidSequenceAsStringDescriptor &aasd)
{
    auto it = _proteins.find(aasd.name);
    if (it != _proteins.end())
        (*it).second->setAminoAcidSequenceAsString(aasd.sequence);
    else
        PLUGIN_THROW(std::runtime_error("Protein not found: " + aasd.name));
}

void Assembly::setAminoAcidSequenceAsRange(
    const AminoAcidSequenceAsRangeDescriptor &aasd)
{
    auto it = _proteins.find(aasd.name);
    if (it != _proteins.end())
        (*it).second->setAminoAcidSequenceAsRange(
            {aasd.range[0], aasd.range[1]});
    else
        PLUGIN_THROW(std::runtime_error("Protein not found: " + aasd.name));
}

std::string Assembly::getAminoAcidInformation(
    const AminoAcidInformationDescriptor &aasd) const
{
    PLUGIN_INFO << "Returning Amino Acid information from protein " << aasd.name
                << std::endl;

    std::string response;
    auto it = _proteins.find(aasd.name);
    if (it != _proteins.end())
    {
        // Sequences
        for (const auto &sequence : (*it).second->getSequencesAsString())
        {
            if (!response.empty())
                response += "\n";
            response += sequence.second;
        }

        // Glycosylation sites
        const auto &sites = (*it).second->getGlycosylationSites({});
        for (const auto &site : sites)
        {
            std::string s;
            for (const auto &index : site.second)
            {
                if (!s.empty())
                    s += ",";
                s += std::to_string(index + 1); // Site indices start a 1, not 0
            }
            response += "\n" + s;
        }
    }
    else
        PLUGIN_THROW(std::runtime_error("Protein not found: " + aasd.name));

    return response;
}

void Assembly::addRNASequence(const RNASequenceDescriptor &rnad)
{
    auto rd = rnad;
    if (rd.range.size() != 2)
        PLUGIN_THROW(std::runtime_error("Invalid range"));
    const Vector2f range{rd.range[0], rd.range[1]};

    if (rd.params.size() != 3)
        PLUGIN_THROW(std::runtime_error("Invalid params"));

    if (rd.position.size() != 3)
        PLUGIN_THROW(std::runtime_error("Invalid position"));

    const Vector3f params{rd.params[0], rd.params[1], rd.params[2]};

    PLUGIN_INFO << "Loading RNA sequence " << rd.name << " from " << rd.contents
                << std::endl;
    PLUGIN_INFO << "Assembly radius: " << rd.assemblyParams[0] << std::endl;
    PLUGIN_INFO << "RNA radius     : " << rd.assemblyParams[1] << std::endl;
    PLUGIN_INFO << "Range          : " << rd.range[0] << ", " << rd.range[1]
                << std::endl;
    PLUGIN_INFO << "Params         : " << rd.params[0] << ", " << rd.params[1]
                << ", " << rd.params[2] << std::endl;
    PLUGIN_INFO << "Position       : " << rd.position[0] << ", "
                << rd.position[1] << ", " << rd.position[2] << std::endl;

    rd.position[0] += _position.x;
    rd.position[1] += _position.y;
    rd.position[2] += _position.z;
    _rnaSequence = RNASequencePtr(new RNASequence(_scene, rd));
    const auto modelDescriptor = _rnaSequence->getModelDescriptor();
    _scene.addModel(modelDescriptor);
}
} // namespace bioexplorer
