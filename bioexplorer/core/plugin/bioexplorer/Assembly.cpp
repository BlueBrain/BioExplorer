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
#include <plugin/common/GeneralSettings.h>
#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>

namespace bioexplorer
{
Assembly::Assembly(Scene &scene, const AssemblyDetails &descriptor)
    : _details(descriptor)
    , _scene(scene)
{
    if (descriptor.position.size() != 3)
        PLUGIN_THROW("Position must be a sequence of 3 float values");

    if (descriptor.rotation.size() != 4)
        PLUGIN_THROW("rotation must be a sequence of 4 float values");

    if (descriptor.clippingPlanes.size() % 4 != 0)
        PLUGIN_THROW("Clipping planes must be defined by 4 float values");
    const auto &cp = descriptor.clippingPlanes;
    for (size_t i = 0; i < cp.size(); i += 4)
        _clippingPlanes.push_back({cp[i], cp[i + 1], cp[i + 2], cp[i + 3]});

    PLUGIN_INFO << "Adding assembly [" << descriptor.name << "]" << std::endl;
}

Assembly::~Assembly()
{
    for (const auto &protein : _proteins)
    {
        const auto modelId = protein.second->getModelDescriptor()->getModelID();
        PLUGIN_INFO << "Removing protein [" << modelId << "] [" << protein.first
                    << "] from assembly [" << _details.name << "]" << std::endl;
        _scene.removeModel(protein.second->getModelDescriptor()->getModelID());
    }
    for (const auto &meshBasedMembrane : _meshBasedMembranes)
    {
        const auto modelId =
            meshBasedMembrane.second->getModelDescriptor()->getModelID();
        PLUGIN_INFO << "Removing mesh [" << modelId << "] ["
                    << meshBasedMembrane.first << "] from assembly ["
                    << _details.name << "]" << std::endl;
        _scene.removeModel(
            meshBasedMembrane.second->getModelDescriptor()->getModelID());
    }
    if (_rnaSequence)
    {
        const auto modelId = _rnaSequence->getModelDescriptor()->getModelID();
        PLUGIN_INFO << "Removing RNA sequence [" << modelId
                    << "] from assembly [" << _details.name << "]" << std::endl;
        _scene.removeModel(modelId);
    }
}

void Assembly::addProtein(const ProteinDetails &pd)
{
    ProteinPtr protein(new Protein(_scene, pd));
    auto modelDescriptor = protein->getModelDescriptor();

    const Vector3f position = {pd.position[0], pd.position[1], pd.position[2]};
    const Quaterniond rotation = {pd.rotation[0], pd.rotation[1],
                                  pd.rotation[2], pd.rotation[3]};

    _processInstances(modelDescriptor, pd.name, pd.shape, pd.assemblyParams,
                      pd.occurrences, position, rotation, pd.allowedOccurrences,
                      pd.randomSeed, pd.positionRandomizationType);

    _proteins[pd.name] = std::move(protein);
    _scene.addModel(modelDescriptor);
    PLUGIN_INFO << "Number of instances   : "
                << modelDescriptor->getInstances().size() << std::endl;
}

void Assembly::addMembrane(const MembraneDetails &md)
{
    if (_membrane != nullptr)
        PLUGIN_THROW("Assembly already has a membrane");

    const Vector3f position = {_details.position[0], _details.position[1],
                               _details.position[2]};
    const Quaterniond rotation = {_details.rotation[0], _details.rotation[1],
                                  _details.rotation[2], _details.rotation[3]};

    MembranePtr membrane(
        new Membrane(_scene, md, position, rotation, _clippingPlanes));
    _membrane = std::move(membrane);
}

void Assembly::addSugars(const SugarsDetails &sd)
{
    // Get information from target protein (attributes, number of instances,
    // glycosylation sites, etc)
    const auto it = _proteins.find(sd.proteinName);
    if (it == _proteins.end())
    {
        std::string s;
        for (const auto &protein : _proteins)
            s += "[" + protein.first + "]";
        PLUGIN_THROW("Target protein " + sd.proteinName +
                     " not registered in assembly " + sd.assemblyName +
                     ". Registered proteins are " + s);
    }
    PLUGIN_INFO << "Adding sugars to protein " << sd.proteinName << std::endl;
    const auto targetProtein = (*it).second;
    targetProtein->addSugars(sd);
}

void Assembly::addGlycans(const SugarsDetails &sd)
{
    // Get information from target protein (attributes, number of instances,
    // glycosylation sites, etc)
    const auto it = _proteins.find(sd.proteinName);
    if (it == _proteins.end())
    {
        std::string s;
        for (const auto &protein : _proteins)
            s += "[" + protein.first + "]";
        PLUGIN_THROW("Target protein " + sd.proteinName +
                     " not registered in assembly " + sd.assemblyName +
                     ". Registered proteins are " + s);
    }

    PLUGIN_INFO << "Adding glycans to protein " << sd.proteinName << std::endl;
    const auto targetProtein = (*it).second;
    targetProtein->addGlycans(sd);
}

void Assembly::addMeshBasedMembrane(const MeshBasedMembraneDetails &md)
{
    MeshBasedMembranePtr meshBaseMembrane(new MeshBasedMembrane(_scene, md));
    auto modelDescriptor = meshBaseMembrane->getModelDescriptor();
    _meshBasedMembranes[md.name] = std::move(meshBaseMembrane);
    _scene.addModel(modelDescriptor);
}

void Assembly::_processInstances(
    ModelDescriptorPtr md, const std::string &name, const AssemblyShape shape,
    const floats &assemblyParams, const size_t occurrences,
    const Vector3f &position, const Quaterniond &rotation,
    const size_ts &allowedOccurrences, const size_t randomSeed,
    const PositionRandomizationType &randomizationType)
{
    const float offset = 2.f / occurrences;
    const float increment = M_PI * (3.f - sqrt(5.f));

    srand(randomSeed);
    size_t rnd{1};
    if (occurrences != 0 && randomSeed != 0 &&
        randomizationType == PositionRandomizationType::circular)
        rnd = rand() % occurrences;

    const Quaterniond assemblyrotation = {_details.rotation[0],
                                          _details.rotation[1],
                                          _details.rotation[2],
                                          _details.rotation[3]};
    const Vector3f assemblyPosition = {_details.position[0],
                                       _details.position[1],
                                       _details.position[2]};

    // Shape parameters
    const auto &params = assemblyParams;
    const float size = (params.size() > 0 ? params[0] : 0.f);
    RandomizationInformation randInfo;
    randInfo.seed = randomSeed;
    randInfo.randomizationType = randomizationType;
    randInfo.positionStrength = (params.size() > 2 ? params[2] : 0.f);
    randInfo.rotationStrength = (params.size() > 4 ? params[4] : 0.f);
    const float extraParameter = (params.size() > 5 ? params[5] : 0.f);

    // Shape
    uint64_t count = 0;
    for (uint64_t occurence = 0; occurence < occurrences; ++occurence)
    {
        if (!allowedOccurrences.empty() &&
            std::find(allowedOccurrences.begin(), allowedOccurrences.end(),
                      occurence) == allowedOccurrences.end())
            continue;

        randInfo.positionSeed =
            (params.size() > 1 ? (params[1] == 0 ? 0 : params[1] + occurence)
                               : 0);
        randInfo.rotationSeed =
            (params.size() > 3 ? (params[3] == 0 ? 0 : params[3] + occurence)
                               : 0);

        Transformation transformation;
        switch (shape)
        {
        case AssemblyShape::spherical:
        {
            transformation = getSphericalPosition(position, size, occurence,
                                                  occurrences, randInfo);
            break;
        }
        case AssemblyShape::sinusoidal:
        {
            transformation =
                getSinosoidalPosition(position, size, extraParameter, occurence,
                                      randInfo);
            break;
        }
        case AssemblyShape::cubic:
        {
            transformation = getCubicPosition(position, size, randInfo);
            break;
        }
        case AssemblyShape::fan:
        {
            transformation = getFanPosition(position, size, occurence,
                                            occurrences, randInfo);
            break;
        }
        case AssemblyShape::bezier:
        {
            if ((params.size() - 5) % 3 != 0)
                PLUGIN_THROW(
                    "Invalid number of floats in assembly extra parameters");
            Vector3fs points;
            for (uint32_t i = 5; i < params.size(); i += 3)
                points.push_back(
                    Vector3f(params[i], params[i + 1], params[i + 2]));
            const auto assemblySize = assemblyParams[0];
            transformation =
                getBezierPosition(points, assemblySize,
                                  float(occurence) / float(occurrences));
            break;
        }
        case AssemblyShape::spherical_to_planar:
        {
            transformation =
                getSphericalToPlanarPosition(position, size, occurence,
                                             occurrences, randInfo,
                                             extraParameter);
            break;
        }
        default:
            transformation = getPlanarPosition(position, size, randInfo);
            break;
        }

        // Final transformation
        const Vector3f translation =
            assemblyPosition +
            Vector3f(assemblyrotation *
                     Vector3d(transformation.getTranslation()));

        if (isClipped(translation, _clippingPlanes))
            continue;

        Transformation finalTransformation;
        finalTransformation.setTranslation(translation);
        finalTransformation.setRotation(
            assemblyrotation * transformation.getRotation() * rotation);

        if (count == 0)
            md->setTransformation(finalTransformation);
        const ModelInstance instance(true, false, finalTransformation);
        md->addInstance(instance);

        ++count;
    }
}

void Assembly::setColorScheme(const ColorSchemeDetails &csd)
{
    if (csd.palette.size() < 3 || csd.palette.size() % 3 != 0)
        PLUGIN_THROW("Invalid palette size");

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
    const AminoAcidSequenceAsStringDetails &aasd)
{
    auto it = _proteins.find(aasd.name);
    if (it != _proteins.end())
        (*it).second->setAminoAcidSequenceAsString(aasd.sequence);
    else
        PLUGIN_THROW("Protein not found: " + aasd.name);
}

void Assembly::setAminoAcidSequenceAsRange(
    const AminoAcidSequenceAsRangesDetails &aasd)
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
        PLUGIN_THROW("Protein not found: " + aasd.name);
}

std::string Assembly::getAminoAcidInformation(
    const AminoAcidInformationDetails &aasd) const
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
        PLUGIN_THROW("Protein not found: " + aasd.name);

    return response;
}

void Assembly::setAminoAcid(const AminoAcidDetails &aminoAcid)
{
    auto it = _proteins.find(aminoAcid.name);
    if (it != _proteins.end())
        (*it).second->setAminoAcid(aminoAcid);
    else
        PLUGIN_THROW("Protein not found: " + aminoAcid.name);
}

void Assembly::addRNASequence(const RNASequenceDetails &rnad)
{
    auto rd = rnad;
    if (rd.range.size() != 2)
        PLUGIN_THROW("Invalid range");
    const Vector2f range{rd.range[0], rd.range[1]};

    if (rd.params.size() != 3)
        PLUGIN_THROW("Invalid params");

    if (rd.position.size() != 3)
        PLUGIN_THROW("Invalid position");

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
        rd.position[i] += _details.position[i];

    _rnaSequence = RNASequencePtr(new RNASequence(_scene, rd));
    const auto modelDescriptor = _rnaSequence->getModelDescriptor();
    _scene.addModel(modelDescriptor);
}

void Assembly::setProteinInstanceTransformation(
    const ProteinInstanceTransformationDetails &descriptor)
{
    ProteinPtr protein{nullptr};
    auto itProtein = _proteins.find(descriptor.name);
    if (itProtein != _proteins.end())
        protein = (*itProtein).second;
    else
        PLUGIN_THROW("Protein " + descriptor.name + " not found on assembly " +
                     descriptor.assemblyName);

    auto modelDescriptor = protein->getModelDescriptor();

    auto &instances = modelDescriptor->getInstances();
    if (descriptor.instanceIndex >= instances.size())
        PLUGIN_THROW("Invalid instance index (" +
                     std::to_string(descriptor.instanceIndex) +
                     ") for protein " + descriptor.name + " in assembly " +
                     descriptor.assemblyName);

    auto instance = modelDescriptor->getInstance(descriptor.instanceIndex);
    auto &transformation = instance->getTransformation();

    if (descriptor.position.size() != 3)
        PLUGIN_THROW("Invalid number of float for position of protein " +
                     descriptor.name + " in assembly " +
                     descriptor.assemblyName);
    const Vector3f position{descriptor.position[0], descriptor.position[1],
                            descriptor.position[2]};

    if (descriptor.rotation.size() != 4)
        PLUGIN_THROW("Invalid number of float for position of protein " +
                     descriptor.name + " in assembly " +
                     descriptor.assemblyName);
    const Quaterniond rotation{descriptor.rotation[0], descriptor.rotation[1],
                               descriptor.rotation[2], descriptor.rotation[3]};

    PLUGIN_INFO << "Modifying instance " << descriptor.instanceIndex
                << " of protein " << descriptor.name << " in assembly "
                << descriptor.assemblyName << " with position=" << position
                << " and rotation=" << rotation << std::endl;
    Transformation newTransformation = transformation;
    newTransformation.setTranslation(position);
    newTransformation.setRotation(rotation);
    instance->setTransformation(newTransformation);
    if (descriptor.instanceIndex == 0)
        modelDescriptor->setTransformation(newTransformation);

    _scene.markModified();
}

const Transformation Assembly::getProteinInstanceTransformation(
    const ProteinInstanceTransformationDetails &descriptor) const
{
    ProteinPtr protein{nullptr};
    auto itProtein = _proteins.find(descriptor.name);
    if (itProtein != _proteins.end())
        protein = (*itProtein).second;
    else
        PLUGIN_THROW("Protein " + descriptor.name + " not found on assembly " +
                     descriptor.assemblyName);

    auto modelDescriptor = protein->getModelDescriptor();

    auto &instances = modelDescriptor->getInstances();
    if (descriptor.instanceIndex >= instances.size())
        PLUGIN_THROW("Invalid instance index (" +
                     std::to_string(descriptor.instanceIndex) +
                     ") for protein " + descriptor.name + " in assembly " +
                     descriptor.assemblyName);

    auto instance = modelDescriptor->getInstance(descriptor.instanceIndex);
    auto transformation = instance->getTransformation();

    if (descriptor.instanceIndex == 0)
        transformation = modelDescriptor->getTransformation();
    return transformation;
}

} // namespace bioexplorer
