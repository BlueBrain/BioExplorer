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

#include "Assembly.h"

#include <plugin/bioexplorer/Membrane.h>
#include <plugin/bioexplorer/MeshBasedMembrane.h>
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

    if (ad.orientation.size() != 4)
        PLUGIN_THROW(std::runtime_error(
            "Orientation must be a sequence of 4 float values"));

    if (ad.clippingPlanes.size() % 4 != 0)
        PLUGIN_THROW(std::runtime_error(
            "Clipping planes must be defined by 4 float values"));
    const auto &cp = ad.clippingPlanes;
    for (size_t i = 0; i < cp.size(); i += 4)
        _clippingPlanes.push_back({cp[i], cp[i + 1], cp[i + 2], cp[i + 3]});

    PLUGIN_INFO << "Adding assembly [" << ad.name << "]" << std::endl;
}

Assembly::~Assembly()
{
    for (const auto &protein : _proteins)
    {
        const auto modelId = protein.second->getModelDescriptor()->getModelID();
        PLUGIN_INFO << "Removing protein [" << modelId << "] [" << protein.first
                    << "] from assembly [" << _descriptor.name << "]"
                    << std::endl;
        _scene.removeModel(protein.second->getModelDescriptor()->getModelID());
    }
    for (const auto &meshBasedMembrane : _meshBasedMembranes)
    {
        const auto modelId =
            meshBasedMembrane.second->getModelDescriptor()->getModelID();
        PLUGIN_INFO << "Removing mesh [" << modelId << "] ["
                    << meshBasedMembrane.first << "] from assembly ["
                    << _descriptor.name << "]" << std::endl;
        _scene.removeModel(
            meshBasedMembrane.second->getModelDescriptor()->getModelID());
    }
    if (_rnaSequence)
    {
        const auto modelId = _rnaSequence->getModelDescriptor()->getModelID();
        PLUGIN_INFO << "Removing RNA sequence [" << modelId
                    << "] from assembly [" << _descriptor.name << "]"
                    << std::endl;
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
                      pd.occurrences, position, orientation,
                      pd.allowedOccurrences, pd.randomSeed,
                      pd.positionRandomizationType, pd.locationCutoffAngle);

    _proteins[pd.name] = std::move(protein);
    _scene.addModel(modelDescriptor);
    PLUGIN_INFO << "Number of instances   : "
                << modelDescriptor->getInstances().size() << std::endl;
}

void Assembly::addMembrane(const MembraneDescriptor &md)
{
    if (_membrane != nullptr)
        PLUGIN_THROW(std::runtime_error("Assembly already has a membrane"));

    const Vector3f position = {_descriptor.position[0], _descriptor.position[1],
                               _descriptor.position[2]};
    const Quaterniond orientation = {_descriptor.orientation[0],
                                     _descriptor.orientation[1],
                                     _descriptor.orientation[2],
                                     _descriptor.orientation[3]};

    MembranePtr membrane(new Membrane(_scene, md, position, orientation,
                                      _clippingPlanes, _occupiedDirections));
    _membrane = std::move(membrane);
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
    PLUGIN_INFO << "Adding sugars to protein " << sd.proteinName << std::endl;
    const auto targetProtein = (*it).second;
    targetProtein->addSugars(sd);
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
    targetProtein->addGlycans(sd);
}

void Assembly::addMeshBasedMembrane(const MeshBasedMembraneDescriptor &md)
{
    MeshBasedMembranePtr meshBaseMembrane(new MeshBasedMembrane(_scene, md));
    auto modelDescriptor = meshBaseMembrane->getModelDescriptor();
    _meshBasedMembranes[md.name] = std::move(meshBaseMembrane);
    _scene.addModel(modelDescriptor);
}

void Assembly::_processInstances(
    ModelDescriptorPtr md, const std::string &name, const AssemblyShape shape,
    const floats &assemblyParams, const size_t occurrences,
    const Vector3f &position, const Quaterniond &orientation,
    const size_ts &allowedOccurrences, const size_t randomSeed,
    const PositionRandomizationType &randomizationType,
    const float locationCutoffAngle)
{
    const float offset = 2.f / occurrences;
    const float increment = M_PI * (3.f - sqrt(5.f));

    srand(randomSeed);
    size_t rnd{1};
    if (occurrences != 0 && randomSeed != 0 &&
        randomizationType == PositionRandomizationType::circular)
        rnd = rand() % occurrences;

    const Quaterniond assemblyOrientation = {_descriptor.orientation[0],
                                             _descriptor.orientation[1],
                                             _descriptor.orientation[2],
                                             _descriptor.orientation[3]};
    const Vector3f assemblyPosition = {_descriptor.position[0],
                                       _descriptor.position[1],
                                       _descriptor.position[2]};

    uint64_t count = 0;
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
                                 position, pos, dir);
            break;
        }
        case AssemblyShape::sinusoidal:
        {
            const auto assemblySize = assemblyParams[0];
            const auto assemblyHeight = assemblyParams[1];
            getSinosoidalPosition(assemblySize, assemblyHeight,
                                  randomizationType, randomSeed, position, pos,
                                  dir);
            break;
        }
        case AssemblyShape::cubic:
        {
            const auto assemblySize = assemblyParams[0];
            getCubicPosition(assemblySize, position, pos, dir);
            break;
        }
        case AssemblyShape::fan:
        {
            const auto assemblyRadius = assemblyParams[0];
            getFanPosition(rnd, assemblyRadius, randomizationType, randomSeed,
                           i, occurrences, position, pos, dir);
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
                              position, pos, dir);
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

        const Quaterniond instanceOrientation = glm::quatLookAt(dir, UP_VECTOR);

        Transformation tf;
        tf.setTranslation(assemblyPosition +
                          Vector3f(assemblyOrientation * Vector3d(pos)));
        tf.setRotation(assemblyOrientation * instanceOrientation * orientation);

        if (count == 0)
            md->setTransformation(tf);
        const ModelInstance instance(true, false, tf);
        md->addInstance(instance);

        // Store occupied direction
        if (locationCutoffAngle != 0.f)
            _occupiedDirections.push_back({dir, locationCutoffAngle});

        ++count;
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
        auto itMesh = _meshBasedMembranes.find(csd.name);
        if (itMesh != _meshBasedMembranes.end())
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
    const AminoAcidSequenceAsRangesDescriptor &aasd)
{
    auto it = _proteins.find(aasd.name);
    if (it != _proteins.end())
    {
        Vector2uis ranges;
        for (size_t i = 0; i < aasd.ranges.size(); i += 2)
            ranges.push_back({aasd.ranges[i], aasd.ranges[i + 1]});

        (*it).second->setAminoAcidSequenceAsRanges(ranges);
    }
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

void Assembly::setAminoAcid(const SetAminoAcid &aminoAcid)
{
    auto it = _proteins.find(aminoAcid.name);
    if (it != _proteins.end())
        (*it).second->setAminoAcid(aminoAcid);
    else
        PLUGIN_THROW(
            std::runtime_error("Protein not found: " + aminoAcid.name));
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

    for (size_t i = 0; i < 3; ++i)
        rd.position[i] += _descriptor.position[i];

    _rnaSequence = RNASequencePtr(new RNASequence(_scene, rd));
    const auto modelDescriptor = _rnaSequence->getModelDescriptor();
    _scene.addModel(modelDescriptor);
}
} // namespace bioexplorer
