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

#include <plugin/biology/MeshBasedMembrane.h>
#include <plugin/biology/ParametricMembrane.h>
#include <plugin/biology/Protein.h>
#include <plugin/biology/RNASequence.h>
#include <plugin/common/GeneralSettings.h>
#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>

namespace bioexplorer
{
namespace biology
{
using namespace common;

Assembly::Assembly(Scene &scene, const AssemblyDetails &details)
    : _details(details)
    , _scene(scene)
{
    if (details.position.size() != 3)
        PLUGIN_THROW("Position must be a sequence of 3 float values");

    if (details.rotation.size() != 4)
        PLUGIN_THROW("rotation must be a sequence of 4 float values");

    if (details.clippingPlanes.size() % 4 != 0)
        PLUGIN_THROW("Clipping planes must be defined by 4 float values");
    const auto &cp = details.clippingPlanes;
    for (size_t i = 0; i < cp.size(); i += 4)
        _clippingPlanes.push_back({cp[i], cp[i + 1], cp[i + 2], cp[i + 3]});

    PLUGIN_INFO("Adding assembly [" << details.name << "]");
}

Assembly::~Assembly()
{
    for (const auto &protein : _proteins)
    {
        const auto modelId = protein.second->getModelDescriptor()->getModelID();
        PLUGIN_INFO("Removing protein [" << modelId << "] [" << protein.first
                                         << "] from assembly [" << _details.name
                                         << "]");
        _scene.removeModel(protein.second->getModelDescriptor()->getModelID());
    }
    if (_rnaSequence)
    {
        const auto modelId = _rnaSequence->getModelDescriptor()->getModelID();
        PLUGIN_INFO("Removing RNA sequence [" << modelId << "] from assembly ["
                                              << _details.name << "]");
        _scene.removeModel(modelId);
    }
}

void Assembly::addProtein(const ProteinDetails &details)
{
    ProteinPtr protein(new Protein(_scene, details));
    auto modelDescriptor = protein->getModelDescriptor();

    const Vector3f position = {details.position[0], details.position[1],
                               details.position[2]};
    const Quaterniond rotation = {details.rotation[0], details.rotation[1],
                                  details.rotation[2], details.rotation[3]};

    _processInstances(modelDescriptor, details.name, details.shape,
                      details.assemblyParams, details.occurrences, position,
                      rotation, details.allowedOccurrences, details.randomSeed,
                      details.positionRandomizationType);

    _proteins[details.name] = std::move(protein);
    _scene.addModel(modelDescriptor);
    PLUGIN_INFO(
        "Number of instances: " << modelDescriptor->getInstances().size());
}

void Assembly::addParametricMembrane(const ParametricMembraneDetails &details)
{
    if (_membrane)
        PLUGIN_THROW("Assembly already has a parametric membrane");

    const Vector3f position = {_details.position[0], _details.position[1],
                               _details.position[2]};
    const Quaterniond rotation = {_details.rotation[0], _details.rotation[1],
                                  _details.rotation[2], _details.rotation[3]};

    ParametricMembranePtr membrane(new ParametricMembrane(_scene, details));
    _membrane = std::move(membrane);
}

void Assembly::addMeshBasedMembrane(const MeshBasedMembraneDetails &details)
{
    if (_membrane)
        PLUGIN_THROW("Assembly already has a mesh-based membrane");

    MeshBasedMembranePtr membrane(new MeshBasedMembrane(_scene, details));
    _membrane = std::move(membrane);
}

void Assembly::addSugars(const SugarsDetails &details)
{
    // Get information from target protein (attributes, number of instances,
    // glycosylation sites, etc)
    const auto it = _proteins.find(details.proteinName);
    if (it == _proteins.end())
    {
        std::string s;
        for (const auto &protein : _proteins)
            s += "[" + protein.first + "]";
        PLUGIN_THROW("Target protein " + details.proteinName +
                     " not registered in assembly " + details.assemblyName +
                     ". Registered proteins are " + s);
    }
    PLUGIN_INFO("Adding sugars to protein " << details.proteinName);
    const auto targetProtein = (*it).second;
    targetProtein->addSugars(details);
}

void Assembly::addGlycans(const SugarsDetails &details)
{
    // Get information from target protein (attributes, number of instances,
    // glycosylation sites, etc)
    const auto it = _proteins.find(details.proteinName);
    if (it == _proteins.end())
    {
        std::string s;
        for (const auto &protein : _proteins)
            s += "[" + protein.first + "]";
        PLUGIN_THROW("Target protein " + details.proteinName +
                     " not registered in assembly " + details.assemblyName +
                     ". Registered proteins are " + s);
    }

    PLUGIN_INFO("Adding glycans to protein " << details.proteinName);
    const auto targetProtein = (*it).second;
    targetProtein->addGlycans(details);
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
    RandomizationDetails randInfo;
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

        // Clipping planes
        Vector3f translation = Vector3f(
            assemblyrotation * Vector3d(transformation.getTranslation()));
        if (isClipped(translation, _clippingPlanes))
            continue;

        // Final transformation
        translation += assemblyPosition;

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

void Assembly::setColorScheme(const ColorSchemeDetails &details)
{
    if (details.palette.size() < 3 || details.palette.size() % 3 != 0)
        PLUGIN_THROW("Invalid palette size");

    ProteinPtr protein{nullptr};
    const auto itProtein = _proteins.find(details.name);
    if (itProtein != _proteins.end())
        protein = (*itProtein).second;
    else if (_membrane)
    {
        const auto membraneProteins = _membrane->getProteins();
        const auto it =
            membraneProteins.find(details.assemblyName + '_' + details.name);
        if (it != membraneProteins.end())
            protein = (*it).second;
    }

    if (protein)
    {
        Palette palette;
        for (size_t i = 0; i < details.palette.size(); i += 3)
            palette.push_back({details.palette[i], details.palette[i + 1],
                               details.palette[i + 2]});

        PLUGIN_INFO("Applying color scheme to protein "
                    << details.name << " on assembly " << details.assemblyName);
        protein->setColorScheme(details.colorScheme, palette, details.chainIds);

        _scene.markModified();
    }
    else
        PLUGIN_ERROR("Protein " << details.name << " not found on assembly "
                                << details.assemblyName);
}

void Assembly::setAminoAcidSequenceAsString(
    const AminoAcidSequenceAsStringDetails &details)
{
    const auto it = _proteins.find(details.name);
    if (it != _proteins.end())
        (*it).second->setAminoAcidSequenceAsString(details.sequence);
    else
        PLUGIN_THROW("Protein not found: " + details.name);
}

void Assembly::setAminoAcidSequenceAsRange(
    const AminoAcidSequenceAsRangesDetails &details)
{
    const auto it = _proteins.find(details.name);
    if (it != _proteins.end())
    {
        Vector2uis ranges;
        for (size_t i = 0; i < details.ranges.size(); i += 2)
            ranges.push_back({details.ranges[i], details.ranges[i + 1]});

        (*it).second->setAminoAcidSequenceAsRanges(ranges);
    }
    else
        PLUGIN_THROW("Protein not found: " + details.name);
}

const std::string Assembly::getAminoAcidInformation(
    const AminoAcidInformationDetails &details) const
{
    PLUGIN_INFO("Returning Amino Acid information from protein "
                << details.name);

    std::string response;
    const auto it = _proteins.find(details.name);
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
        PLUGIN_THROW("Protein not found: " + details.name);

    return response;
}

void Assembly::setAminoAcid(const AminoAcidDetails &details)
{
    auto it = _proteins.find(details.name);
    if (it != _proteins.end())
        (*it).second->setAminoAcid(details);
    else
        PLUGIN_THROW("Protein not found: " + details.name);
}

void Assembly::addRNASequence(const RNASequenceDetails &details)
{
    auto rd = details;
    if (rd.range.size() != 2)
        PLUGIN_THROW("Invalid range");
    const Vector2f range{rd.range[0], rd.range[1]};

    if (rd.params.size() != 3)
        PLUGIN_THROW("Invalid params");

    if (rd.position.size() != 3)
        PLUGIN_THROW("Invalid position");

    const Vector3f params{rd.params[0], rd.params[1], rd.params[2]};

    PLUGIN_INFO("Loading RNA sequence " << rd.name << " from " << rd.contents);
    PLUGIN_INFO("Assembly radius: " << rd.assemblyParams[0]);
    PLUGIN_INFO("RNA radius     : " << rd.assemblyParams[1]);
    PLUGIN_INFO("Range          : " << rd.range[0] << ", " << rd.range[1]);
    PLUGIN_INFO("Params         : " << rd.params[0] << ", " << rd.params[1]
                                    << ", " << rd.params[2]);
    PLUGIN_INFO("Position       : " << rd.position[0] << ", " << rd.position[1]
                                    << ", " << rd.position[2]);

    for (size_t i = 0; i < 3; ++i)
        rd.position[i] += _details.position[i];

    _rnaSequence = RNASequencePtr(new RNASequence(_scene, rd));
    const auto modelDescriptor = _rnaSequence->getModelDescriptor();
    _scene.addModel(modelDescriptor);
}

void Assembly::setProteinInstanceTransformation(
    const ProteinInstanceTransformationDetails &details)
{
    ProteinPtr protein{nullptr};
    const auto itProtein = _proteins.find(details.name);
    if (itProtein != _proteins.end())
        protein = (*itProtein).second;
    else
        PLUGIN_THROW("Protein " + details.name + " not found on assembly " +
                     details.assemblyName);

    const auto modelDescriptor = protein->getModelDescriptor();

    const auto &instances = modelDescriptor->getInstances();
    if (details.instanceIndex >= instances.size())
        PLUGIN_THROW("Invalid instance index (" +
                     std::to_string(details.instanceIndex) + ") for protein " +
                     details.name + " in assembly " + details.assemblyName);

    const auto instance = modelDescriptor->getInstance(details.instanceIndex);
    const auto &transformation = instance->getTransformation();

    if (details.position.size() != 3)
        PLUGIN_THROW("Invalid number of float for position of protein " +
                     details.name + " in assembly " + details.assemblyName);
    const Vector3f position{details.position[0], details.position[1],
                            details.position[2]};

    if (details.rotation.size() != 4)
        PLUGIN_THROW("Invalid number of float for position of protein " +
                     details.name + " in assembly " + details.assemblyName);
    const Quaterniond rotation{details.rotation[0], details.rotation[1],
                               details.rotation[2], details.rotation[3]};

    PLUGIN_INFO("Modifying instance "
                << details.instanceIndex << " of protein " << details.name
                << " in assembly " << details.assemblyName << " with position="
                << position << " and rotation=" << rotation);
    Transformation newTransformation = transformation;
    newTransformation.setTranslation(position);
    newTransformation.setRotation(rotation);
    instance->setTransformation(newTransformation);
    if (details.instanceIndex == 0)
        modelDescriptor->setTransformation(newTransformation);

    _scene.markModified();
}

const Transformation Assembly::getProteinInstanceTransformation(
    const ProteinInstanceTransformationDetails &details) const
{
    ProteinPtr protein{nullptr};
    const auto itProtein = _proteins.find(details.name);
    if (itProtein != _proteins.end())
        protein = (*itProtein).second;
    else
        PLUGIN_THROW("Protein " + details.name + " not found on assembly " +
                     details.assemblyName);

    const auto modelDescriptor = protein->getModelDescriptor();

    const auto &instances = modelDescriptor->getInstances();
    if (details.instanceIndex >= instances.size())
        PLUGIN_THROW("Invalid instance index (" +
                     std::to_string(details.instanceIndex) + ") for protein " +
                     details.name + " in assembly " + details.assemblyName);

    const auto instance = modelDescriptor->getInstance(details.instanceIndex);
    auto transformation = instance->getTransformation();

    if (details.instanceIndex == 0)
        transformation = modelDescriptor->getTransformation();
    return transformation;
}
} // namespace biology
} // namespace bioexplorer
