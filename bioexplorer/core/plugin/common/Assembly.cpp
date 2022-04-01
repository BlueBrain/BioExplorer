/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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

#include <plugin/common/GeneralSettings.h>
#include <plugin/common/Logs.h>
#include <plugin/common/Node.h>
#include <plugin/common/Utils.h>
#include <plugin/common/shapes/BezierShape.h>
#include <plugin/common/shapes/CubeShape.h>
#include <plugin/common/shapes/FanShape.h>
#include <plugin/common/shapes/HelixShape.h>
#include <plugin/common/shapes/MeshShape.h>
#include <plugin/common/shapes/PlaneShape.h>
#include <plugin/common/shapes/PointShape.h>
#include <plugin/common/shapes/SinusoidShape.h>
#include <plugin/common/shapes/SphereShape.h>
#include <plugin/molecularsystems/EnzymeReaction.h>
#include <plugin/molecularsystems/Membrane.h>
#include <plugin/molecularsystems/Protein.h>
#include <plugin/molecularsystems/RNASequence.h>
#include <plugin/morphologies/Astrocytes.h>
#include <plugin/morphologies/Neurons.h>
#include <plugin/vasculature/Vasculature.h>
#include <plugin/vasculature/VasculatureHandler.h>

#include <brayns/engineapi/Model.h>

namespace bioexplorer
{
namespace common
{
Assembly::Assembly(Scene &scene, const AssemblyDetails &details)
    : _details(details)
    , _scene(scene)
{
    const auto size = doublesToVector3d(details.shapeParams);
    _position = doublesToVector3d(_details.position);
    _rotation = doublesToQuaterniond(_details.rotation);
    _clippingPlanes = doublesToVector4ds(details.clippingPlanes);

    switch (details.shape)
    {
    case AssemblyShape::sphere:
    {
        _shape = ShapePtr(new SphereShape(_clippingPlanes, size.x));
        break;
    }
    case AssemblyShape::sinusoid:
    {
        _shape = ShapePtr(new SinusoidShape(_clippingPlanes, size));
        break;
    }
    case AssemblyShape::cube:
    {
        _shape = ShapePtr(new CubeShape(_clippingPlanes, size));
        break;
    }
    case AssemblyShape::fan:
    {
        _shape = ShapePtr(new FanShape(_clippingPlanes, size.x));
        break;
    }
    case AssemblyShape::plane:
    {
        _shape =
            ShapePtr(new PlaneShape(_clippingPlanes, Vector2f(size.x, size.z)));
        break;
    }
    case AssemblyShape::mesh:
    {
        _shape = ShapePtr(
            new MeshShape(_clippingPlanes, size, _details.shapeMeshContents));
        break;
    }
    case AssemblyShape::helix:
    {
        _shape = ShapePtr(new HelixShape(_clippingPlanes, size.x, size.y));
        break;
    }
    default:
        _shape = ShapePtr(new PointShape(_clippingPlanes));
        break;
    }

    PLUGIN_INFO(3, "Adding assembly [" << details.name << "] at position "
                                       << _position << ", rotation "
                                       << _rotation);
}

Assembly::~Assembly()
{
    for (const auto &protein : _proteins)
    {
        const auto modelId = protein.second->getModelDescriptor()->getModelID();
        PLUGIN_INFO(3, "Removing protein [" << modelId << "] [" << protein.first
                                            << "] from assembly ["
                                            << _details.name << "]");
        _scene.removeModel(protein.second->getModelDescriptor()->getModelID());
    }
    if (_rnaSequence)
    {
        const auto modelId = _rnaSequence->getModelDescriptor()->getModelID();
        PLUGIN_INFO(3, "Removing RNA sequence ["
                           << modelId << "] from assembly [" << _details.name
                           << "]");
        _scene.removeModel(modelId);
    }
    if (_vasculature)
    {
        const auto modelId = _vasculature->getModelDescriptor()->getModelID();
        PLUGIN_INFO(3, "Removing Vasculature ["
                           << modelId << "] from assembly [" << _details.name
                           << "]");
        _scene.removeModel(modelId);
    }
    if (_neurons)
    {
        const auto modelId = _neurons->getModelDescriptor()->getModelID();
        PLUGIN_INFO(3, "Removing Neurons [" << modelId << "] from assembly ["
                                            << _details.name << "]");
        _scene.removeModel(modelId);
    }
    _modelDescriptors.clear();
}

void Assembly::addProtein(const ProteinDetails &details,
                          const AssemblyConstraints &constraints)
{
    ProteinPtr protein(new Protein(_scene, details));
    auto modelDescriptor = protein->getModelDescriptor();
    const auto animationParams =
        doublesToAnimationDetails(details.animationParams);
    const auto proteinPosition = doublesToVector3d(details.position);
    const auto proteinRotation = doublesToQuaterniond(details.rotation);
    const auto transmembraneParams =
        doublesToVector2d(details.transmembraneParams);

    _processInstances(modelDescriptor, details.name, details.occurrences,
                      proteinPosition, proteinRotation,
                      details.allowedOccurrences, animationParams,
                      transmembraneParams.x, constraints);

    _proteins[details.name] = std::move(protein);
    _modelDescriptors.push_back(modelDescriptor);
    _scene.addModel(modelDescriptor);
    PLUGIN_INFO(3, "Number of instances: "
                       << modelDescriptor->getInstances().size());
}

void Assembly::addMembrane(const MembraneDetails &details)
{
    if (_membrane)
        PLUGIN_THROW("Assembly already has a membrane");

    MembranePtr membrane(
        new Membrane(details, _scene, _position, _rotation, _shape, _proteins));
    _membrane = std::move(membrane);
}

void Assembly::addSugar(const SugarDetails &details)
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
    PLUGIN_INFO(3, "Adding sugars to protein " << details.proteinName);
    const auto targetProtein = (*it).second;
    targetProtein->addSugar(details);
}

void Assembly::addGlycan(const SugarDetails &details)
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

    PLUGIN_INFO(3, "Adding glycans to protein " << details.proteinName);
    const auto targetProtein = (*it).second;
    targetProtein->addGlycan(details);
}

void Assembly::_processInstances(ModelDescriptorPtr md, const std::string &name,
                                 const size_t occurrences,
                                 const Vector3d &position,
                                 const Quaterniond &rotation,
                                 const uint64_ts &allowedOccurrences,
                                 const AnimationDetails &animationDetails,
                                 const double offset,
                                 const AssemblyConstraints &constraints)
{
    srand(animationDetails.seed);

    // Shape
    uint64_t count = 0;
    for (uint64_t occurrence = 0; occurrence < occurrences; ++occurrence)
    {
        try
        {
            if (!allowedOccurrences.empty() &&
                std::find(allowedOccurrences.begin(), allowedOccurrences.end(),
                          occurrence) == allowedOccurrences.end())
                continue;

            Transformations transformations;

            Transformation assemblyTransformation;
            assemblyTransformation.setTranslation(_position);
            assemblyTransformation.setRotation(_rotation);
            transformations.push_back(assemblyTransformation);

            Transformation shapeTransformation =
                _shape->getTransformation(occurrence, occurrences,
                                          animationDetails, offset);

            transformations.push_back(shapeTransformation);

            Transformation proteinTransformation;
            proteinTransformation.setTranslation(position);
            proteinTransformation.setRotation(rotation);
            transformations.push_back(proteinTransformation);

            const Transformation finalTransformation =
                combineTransformations(transformations);
            const auto &translation = finalTransformation.getTranslation();

            // Assembly constaints
            bool addInstance = true;
            for (const auto &constraint : constraints)
            {
                if (constraint.first == AssemblyConstraintType::inside &&
                    !constraint.second->isInside(translation))
                    addInstance = false;
                if (constraint.first == AssemblyConstraintType::outside &&
                    constraint.second->isInside(translation))
                    addInstance = false;
            }
            if (!addInstance)
                continue;

            if (count == 0)
                md->setTransformation(finalTransformation);
            const ModelInstance instance(true, false, finalTransformation);
            md->addInstance(instance);
            ++count;
        }
        catch (const std::runtime_error &)
        {
            // Instance is clipped
        }
    }
}

void Assembly::setProteinColorScheme(const ProteinColorSchemeDetails &details)
{
    if (details.palette.size() < 3 || details.palette.size() % 3 != 0)
        PLUGIN_THROW("Invalid palette size");

    ProteinPtr protein{nullptr};
    const auto itProtein = _proteins.find(details.name);
    if (itProtein != _proteins.end())
        protein = (*itProtein).second;
    else if (_membrane)
    {
        const auto membraneLipids = _membrane->getLipids();
        const auto it =
            membraneLipids.find(details.assemblyName + '_' + details.name);
        if (it != membraneLipids.end())
            protein = (*it).second;
    }

    if (protein)
    {
        Palette palette;
        for (size_t i = 0; i < details.palette.size(); i += 3)
            palette.push_back({details.palette[i], details.palette[i + 1],
                               details.palette[i + 2]});

        PLUGIN_INFO(3, "Applying color scheme to protein "
                           << details.name << " on assembly "
                           << details.assemblyName);
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
    PLUGIN_INFO(3, "Returning Amino Acid information from protein "
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

    for (size_t i = 0; i < _details.position.size(); ++i)
        rd.position[i] += _details.position[i];

    _rnaSequence = RNASequencePtr(
        new RNASequence(_scene, rd, _clippingPlanes, _position, _rotation));
    const auto modelDescriptor = _rnaSequence->getModelDescriptor();
    _modelDescriptors.push_back(modelDescriptor);
    _scene.addModel(modelDescriptor);
    auto protein = _rnaSequence->getProtein();
    if (protein)
    {
        const auto name = protein->getDescriptor().name;
        _proteins[name] = std::move(protein);
    }
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

    const auto position = doublesToVector3d(details.position);
    const auto rotation = doublesToQuaterniond(details.rotation);

    PLUGIN_INFO(3, "Modifying instance "
                       << details.instanceIndex << " of protein "
                       << details.name << " in assembly "
                       << details.assemblyName << " with position=" << position
                       << " and rotation=" << rotation);
    Transformation newTransformation = transformation;
    newTransformation.setTranslation(position);
    newTransformation.setRotation(rotation);
    if (details.instanceIndex == 0)
        modelDescriptor->setTransformation(newTransformation);
    instance->setTransformation(newTransformation);

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
    const auto transformation = instance->getTransformation();
    const auto &position = transformation.getTranslation();
    const auto &rotation = transformation.getRotation();

    PLUGIN_INFO(3, "Getting instance "
                       << details.instanceIndex << " of protein "
                       << details.name << " in assembly "
                       << details.assemblyName << " with position=" << position
                       << " and rotation=" << rotation);

    return transformation;
}

bool Assembly::isInside(const Vector3d &location) const
{
    bool result = false;
    if (_shape)
        result |= _shape->isInside(location);
    return result;
}

ProteinInspectionDetails Assembly::inspect(const Vector3d &origin,
                                           const Vector3d &direction,
                                           double &t) const
{
    ProteinInspectionDetails result;
    result.hit = false;
    result.assemblyName = _details.name;

    t = std::numeric_limits<double>::max();

    // Proteins
    for (const auto protein : _proteins)
    {
        const auto md = protein.second->getModelDescriptor();
        const auto &instances = md->getInstances();
        const Vector3d instanceHalfSize =
            protein.second->getBounds().getSize() / 2.0;

        uint64_t count = 0;
        for (const auto &instance : instances)
        {
            const auto instancePosition =
                instance.getTransformation().getTranslation();

            Boxd box;
            box.merge(instancePosition - instanceHalfSize);
            box.merge(instancePosition + instanceHalfSize);

            double tHit;
            if (rayBoxIntersection(origin, direction, box, 0.0, t, tHit))
            {
                result.hit = true;
                if (tHit < t)
                {
                    result.proteinName = protein.second->getDescriptor().name;
                    result.modelId = md->getModelID();
                    result.instanceId = count;
                    result.position = {instancePosition.x, instancePosition.y,
                                       instancePosition.z};
                    t = tHit;
                }
            }
            ++count;
        }
    }

    // Membrane
    if (_membrane)
    {
        for (const auto protein : _membrane->getLipids())
        {
            const auto md = protein.second->getModelDescriptor();
            const auto &instances = md->getInstances();
            const Vector3d instanceHalfSize =
                protein.second->getBounds().getSize() / 2.0;

            uint64_t count = 0;
            for (const auto &instance : instances)
            {
                const auto instancePosition =
                    instance.getTransformation().getTranslation();

                Boxd box;
                box.merge(instancePosition - instanceHalfSize);
                box.merge(instancePosition + instanceHalfSize);

                double tHit;
                if (rayBoxIntersection(origin, direction, box, 0.0, t, tHit))
                {
                    result.hit = true;
                    if (tHit < t)
                    {
                        result.proteinName =
                            protein.second->getDescriptor().name;
                        result.modelId = md->getModelID();
                        result.instanceId = count;
                        result.position = {instancePosition.x,
                                           instancePosition.y,
                                           instancePosition.z};
                        t = tHit;
                    }
                }
                ++count;
            }
        }
    }

    return result;
}

void Assembly::addVasculature(const VasculatureDetails &details)
{
    if (_vasculature)
    {
        auto modelDescriptor = _vasculature->getModelDescriptor();
        if (modelDescriptor)
        {
            const auto modelId = modelDescriptor->getModelID();
            _scene.removeModel(modelId);
        }
    }
    _vasculature.reset(std::move(new Vasculature(_scene, details)));
    _scene.markModified(false);
}

std::string Assembly::getVasculatureInfo() const
{
    auto modelDescriptor = _vasculature->getModelDescriptor();
    Response response;
    if (!_vasculature)
        PLUGIN_THROW("No vasculature is currently defined in assembly " +
                     _details.name);
    std::stringstream s;
    s << "modelId=" << modelDescriptor->getModelID() << CONTENTS_DELIMITER
      << "nbNodes=" << _vasculature->getNbNodes() << CONTENTS_DELIMITER
      << "nbSubGraphs=" << _vasculature->getNbSubGraphs() << CONTENTS_DELIMITER
      << "nbSections=" << _vasculature->getNbSections() << CONTENTS_DELIMITER
      << "nbPairs=" << _vasculature->getNbPairs() << CONTENTS_DELIMITER
      << "nbEntryNodes=" << _vasculature->getNbEntryNodes()
      << CONTENTS_DELIMITER
      << "nbMaxPointsPerSection=" << _vasculature->getNbMaxPointsPerSection();
    return s.str().c_str();
}

void Assembly::setVasculatureColorScheme(
    const VasculatureColorSchemeDetails &details)
{
    if (!_vasculature)
        PLUGIN_THROW("No vasculature currently exists");

    auto modelDescriptor = _vasculature->getModelDescriptor();
    if (modelDescriptor)
    {
        const auto modelId = modelDescriptor->getModelID();
        _scene.removeModel(modelId);
    }

    _vasculature->setColorScheme(details);
    _scene.markModified(false);
}

void Assembly::setVasculatureReport(const VasculatureReportDetails &details)
{
    PLUGIN_INFO(3, "Setting report to vasculature");
    if (!_vasculature)
        PLUGIN_THROW("No vasculature is currently loaded");

    auto modelDescriptor = _vasculature->getModelDescriptor();
    auto handler = std::make_shared<VasculatureHandler>(details);
    auto &model = modelDescriptor->getModel();
    model.setSimulationHandler(handler);
}

void Assembly::setVasculatureRadiusReport(
    const VasculatureRadiusReportDetails &details)
{
    if (!_vasculature && !_astrocytes)
        PLUGIN_THROW("No vasculature nor astrocytes are currently loaded");

    if (_vasculature)
        _vasculature->setRadiusReport(details);
    if (_astrocytes)
        _astrocytes->setVasculatureRadiusReport(details);
}

void Assembly::addAstrocytes(const AstrocytesDetails &details)
{
    if (_astrocytes)
        PLUGIN_THROW("Astrocytes already exists in assembly " +
                     details.assemblyName);

    _astrocytes = AstrocytesPtr(new Astrocytes(_scene, details));
    _scene.markModified(false);
}

void Assembly::addNeurons(const NeuronsDetails &details)
{
    if (_neurons)
        PLUGIN_THROW("Neurons already exists in assembly " +
                     details.assemblyName);

    _neurons = NeuronsPtr(new Neurons(_scene, details));
    _scene.markModified(false);
}

Vector4ds Assembly::getNeuronSectionPoints(const NeuronSectionDetails &details)
{
    if (!_neurons)
        PLUGIN_THROW("No neurons are currently defined in assembly " +
                     details.assemblyName);
    return _neurons->getNeuronSectionPoints(details.neuronId,
                                            details.sectionId);
}

ProteinPtr Assembly::getProtein(const std::string &name)
{
    ProteinPtr protein{nullptr};
    const auto it = _proteins.find(name);
    if (it != _proteins.end())
        protein = (*it).second;
    return protein;
}

Transformation Assembly::getTransformation() const
{
    Transformation transformation;
    transformation.setTranslation(doublesToVector3d(_details.position));
    transformation.setRotation(doublesToQuaterniond(_details.rotation));
    return transformation;
}

void Assembly::addEnzymeReaction(const EnzymeReactionDetails &details,
                                 AssemblyPtr enzymeAssembly, ProteinPtr enzyme,
                                 Proteins &substrates, Proteins &products)
{
    auto enzymeReaction =
        EnzymeReactionPtr(new EnzymeReaction(_scene, details, enzymeAssembly,
                                             enzyme, substrates, products));
    _enzymeReactions[details.name] = enzymeReaction;
}

void Assembly::setEnzymeReactionProgress(
    const EnzymeReactionProgressDetails &details)
{
    if (_enzymeReactions.find(details.name) == _enzymeReactions.end())
        PLUGIN_THROW("Enzyme reaction does not exist in assembly " +
                     _details.name);
    _enzymeReactions[details.name]->setProgress(details.instanceId,
                                                details.progress);
}

} // namespace common
} // namespace bioexplorer
