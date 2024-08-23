/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include "Neurons.h"
#include "CompartmentSimulationHandler.h"
#include "SomaSimulationHandler.h"
#include "SpikeSimulationHandler.h"

#include <science/common/Logs.h>
#include <science/common/Utils.h>
#include <science/common/shapes/Shape.h>

#include <science/io/cache/MemoryCache.h>
#include <science/io/db/DBConnector.h>

#include <platform/core/common/Timer.h>
#include <platform/core/engineapi/Material.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Scene.h>

#ifdef USE_CGAL
#include <science/meshing/PointCloudMesher.h>
#include <science/meshing/SurfaceMesher.h>
#endif

#include <omp.h>

#include <random>

using namespace core;

namespace bioexplorer
{
using namespace details;
using namespace common;
using namespace io;
using namespace db;
#ifdef USE_CGAL
using namespace meshing;
#endif

namespace morphology
{
const uint64_t NB_MYELIN_FREE_SEGMENTS = 4;
const double DEFAULT_ARROW_RADIUS_RATIO = 10.0;
const uint64_t DEFAULT_DEBUG_SYNAPSE_DENSITY_RATIO = 1;
const double MAX_SOMA_RADIUS = 10.0;

std::map<ReportType, std::string> reportTypeAsString = {{ReportType::undefined, "undefined"},
                                                        {ReportType::spike, "spike"},
                                                        {ReportType::soma, "soma"},
                                                        {ReportType::compartment, "compartment"},
                                                        {ReportType::synapse_efficacy, "synapse efficacy"}};

// Mitochondria density per layer
// Source: A simplified morphological classification scheme for pyramidal cells in six layers of primary somatosensory
// cortex of juvenile rats https://www.sciencedirect.com/science/article/pii/S2451830118300293)
const doubles MITOCHONDRIA_DENSITY = {0.0459, 0.0522, 0.064, 0.0774, 0.0575, 0.0403};

Neurons::Neurons(Scene& scene, const NeuronsDetails& details, const Vector3d& assemblyPosition,
                 const Quaterniond& assemblyRotation, const LoaderProgress& callback)
    : Morphologies(details.alignToGrid, assemblyPosition, assemblyRotation, doublesToVector3d(details.scale))
    , _details(details)
    , _scene(scene)
{
    _animationDetails = doublesToCellAnimationDetails(_details.animationParams);
    _spheresRepresentation.enabled = _details.morphologyRepresentation == MorphologyRepresentation::spheres ||
                                     _details.morphologyRepresentation == MorphologyRepresentation::uniform_spheres;
    _spheresRepresentation.uniform = _details.morphologyRepresentation == MorphologyRepresentation::uniform_spheres;
    _spheresRepresentation.radius = _spheresRepresentation.uniform ? _details.radiusMultiplier : 0.f;

    srand(_animationDetails.seed);

    Timer chrono;
    _buildModel(callback);
    PLUGIN_TIMER(chrono.elapsed(), "Neurons loaded");
}

double Neurons::_getDisplacementValue(const DisplacementElement& element)
{
    const auto params = _details.displacementParams;
    switch (element)
    {
    case DisplacementElement::morphology_soma_strength:
        return valueFromDoubles(params, 0, DEFAULT_MORPHOLOGY_SOMA_STRENGTH);
    case DisplacementElement::morphology_soma_frequency:
        return valueFromDoubles(params, 1, DEFAULT_MORPHOLOGY_SOMA_FREQUENCY);
    case DisplacementElement::morphology_section_strength:
        return valueFromDoubles(params, 2, DEFAULT_MORPHOLOGY_SECTION_STRENGTH);
    case DisplacementElement::morphology_section_frequency:
        return valueFromDoubles(params, 3, DEFAULT_MORPHOLOGY_SECTION_FREQUENCY);
    case DisplacementElement::morphology_nucleus_strength:
        return valueFromDoubles(params, 4, DEFAULT_MORPHOLOGY_NUCLEUS_STRENGTH);
    case DisplacementElement::morphology_nucleus_frequency:
        return valueFromDoubles(params, 5, DEFAULT_MORPHOLOGY_NUCLEUS_FREQUENCY);
    case DisplacementElement::morphology_mitochondrion_strength:
        return valueFromDoubles(params, 6, DEFAULT_MORPHOLOGY_MITOCHONDRION_STRENGTH);
    case DisplacementElement::morphology_mitochondrion_frequency:
        return valueFromDoubles(params, 7, DEFAULT_MORPHOLOGY_MITOCHONDRION_FREQUENCY);
    case DisplacementElement::morphology_myelin_steath_strength:
        return valueFromDoubles(params, 8, DEFAULT_MORPHOLOGY_MYELIN_STEATH_STRENGTH);
    case DisplacementElement::morphology_myelin_steath_frequency:
        return valueFromDoubles(params, 9, DEFAULT_MORPHOLOGY_MYELIN_STEATH_FREQUENCY);
    case DisplacementElement::morphology_spine_strength:
        return valueFromDoubles(params, 10, DEFAULT_MORPHOLOGY_SPINE_STRENGTH);
    case DisplacementElement::morphology_spine_frequency:
        return valueFromDoubles(params, 11, DEFAULT_MORPHOLOGY_SPINE_FREQUENCY);
    default:
        PLUGIN_THROW("Invalid displacement element");
    }
}

void Neurons::_logRealismParams()
{
    PLUGIN_INFO(1, "----------------------------------------------------");
    PLUGIN_INFO(1, "Realism level (" << static_cast<uint32_t>(_details.realismLevel) << ")");
    PLUGIN_INFO(1, "- Soma     : " << boolAsString(andCheck(static_cast<uint32_t>(_details.realismLevel),
                                                            static_cast<uint32_t>(MorphologyRealismLevel::soma))));
    PLUGIN_INFO(1, "- Axon     : " << boolAsString(andCheck(static_cast<uint32_t>(_details.realismLevel),
                                                            static_cast<uint32_t>(MorphologyRealismLevel::axon))));
    PLUGIN_INFO(1, "- Dendrite : " << boolAsString(andCheck(static_cast<uint32_t>(_details.realismLevel),
                                                            static_cast<uint32_t>(MorphologyRealismLevel::dendrite))));
    PLUGIN_INFO(1, "- Internals: " << boolAsString(andCheck(static_cast<uint32_t>(_details.realismLevel),
                                                            static_cast<uint32_t>(MorphologyRealismLevel::internals))));
    PLUGIN_INFO(1, "- Externals: " << boolAsString(andCheck(static_cast<uint32_t>(_details.realismLevel),
                                                            static_cast<uint32_t>(MorphologyRealismLevel::externals))));
    PLUGIN_INFO(1, "- Spine    : " << boolAsString(andCheck(static_cast<uint32_t>(_details.realismLevel),
                                                            static_cast<uint32_t>(MorphologyRealismLevel::spine))));
    PLUGIN_INFO(1, "----------------------------------------------------");
}

void Neurons::_buildContours(ThreadSafeContainer& container, const NeuronSomaMap& somas)
{
#ifdef USE_CGAL
    PointCloud pointCloud;
    uint64_t progress = 0;
    size_t materialId = 0;

    for (const auto soma : somas)
    {
        PLUGIN_PROGRESS("Loading somas", progress, somas.size());
        ++progress;

        const Vector3d position = soma.second.position;
        pointCloud[materialId].push_back({position.x, position.y, position.z, _details.radiusMultiplier});
    }

    PointCloudMesher pcm;
    pcm.toConvexHull(container, pointCloud);
#else
    PLUGIN_THROW("The BioExplorer was not compiled with the CGAL library")
#endif
}

void Neurons::_buildSurface(const NeuronSomaMap& somas)
{
#ifdef USE_CGAL
    PointCloud pointCloud;
    uint64_t progress = 0;
    size_t materialId = 0;

    for (const auto soma : somas)
    {
        PLUGIN_PROGRESS("Loading somas", progress, somas.size());
        ++progress;

        const Vector3d position = soma.second.position;
        pointCloud[materialId].push_back({position.x, position.y, position.z, _details.radiusMultiplier});
    }

    SurfaceMesher sm(_uuid);
    _modelDescriptor = sm.generateSurface(_scene, _details.assemblyName, pointCloud[materialId]);
    if (!_modelDescriptor)
        PLUGIN_THROW("Failed to generate surface")

#else
    PLUGIN_THROW("The BioExplorer was not compiled with the CGAL library")
#endif
}

void Neurons::_buildModel(const LoaderProgress& callback)
{
    const auto& connector = DBConnector::getInstance();

    auto model = _scene.createModel();
    std::string sqlNodeFilter = _details.sqlNodeFilter;

    // Neurons
    auto somas = connector.getNeurons(_details.populationName, sqlNodeFilter);

    // Report parameters
    float* voltages = nullptr;
    _neuronsReportParameters = doublesToNeuronsReportParametersDetails(_details.reportParams);
    if (_neuronsReportParameters.reportId != -1)
    {
        _simulationReport = connector.getSimulationReport(_details.populationName, _neuronsReportParameters.reportId);
        _attachSimulationReport(*model, somas.size());
        voltages = static_cast<float*>(
            model->getSimulationHandler()->getFrameData(_neuronsReportParameters.initialSimulationFrame));
        if (!_neuronsReportParameters.loadNonSimulatedNodes)
        {
            auto it = somas.begin();
            while (it != somas.end())
            {
                const auto itg = _simulationReport.guids.find((*it).first);
                if (itg == _simulationReport.guids.end())
                    it = somas.erase(it);
                else
                    ++it;
            }
        }
    }

    if (somas.empty())
        PLUGIN_THROW("Selection returned no nodes");

    PLUGIN_INFO(1, "Building " << somas.size() << " neurons");
    _logRealismParams();

    size_t previousMaterialId = std::numeric_limits<size_t>::max();
    Vector3ui indexOffset;

    const bool somasOnly =
        _details.loadSomas && !_details.loadAxon && !_details.loadApicalDendrites && !_details.loadBasalDendrites;

    ThreadSafeContainers containers;
    if (somasOnly || _details.morphologyRepresentation == MorphologyRepresentation::orientation ||
        _details.morphologyRepresentation == MorphologyRepresentation::contour)
    {
        ThreadSafeContainer container(*model, _alignToGrid, _position, _rotation, _scale);
        switch (_details.morphologyRepresentation)
        {
        case MorphologyRepresentation::orientation:
        {
            _buildOrientations(container, somas);
            break;
        }
        case MorphologyRepresentation::contour:
        {
            _buildContours(container, somas);
            break;
        }
        case MorphologyRepresentation::surface:
        {
            _buildSurface(somas);
            return;
            break;
        }
        default:
        {
            _buildSomasOnly(*model, container, somas);
            break;
        }
        }
        containers.push_back(container);
    }
    else
    {
        const auto nbDBConnections = DBConnector::getInstance().getNbConnections();

        uint64_t neuronIndex;
        volatile bool flag = false;
        std::string flagMessage;
#pragma omp parallel for shared(flag, flagMessage) num_threads(nbDBConnections)
        for (neuronIndex = 0; neuronIndex < somas.size(); ++neuronIndex)
        {
            try
            {
                if (flag)
                    continue;

                if (omp_get_thread_num() == 0)
                {
                    PLUGIN_PROGRESS("Loading neurons...", neuronIndex, somas.size() / nbDBConnections);
                    try
                    {
                        callback.updateProgress("Loading neurons...",
                                                0.5f *
                                                    ((float)neuronIndex / ((float)(somas.size() / nbDBConnections))));
                    }
                    catch (...)
                    {
#pragma omp critical
                        {
                            flag = true;
                        }
                    }
                }

                auto it = std::next(somas.begin(), neuronIndex);
                const auto& soma = it->second;
                ThreadSafeContainer container(*model, _alignToGrid, _position, _rotation, _scale);
                _buildMorphology(container, it->first, soma, neuronIndex, voltages);

#pragma omp critical
                {
                    containers.push_back(container);
                }
            }
            catch (const std::runtime_error& e)
            {
#pragma omp critical
                {
                    flagMessage = e.what();
                    flag = true;
                }
            }
            catch (...)
            {
#pragma omp critical
                {
                    flagMessage = "Loading was canceled";
                    flag = true;
                }
            }
        }
    }

    for (uint64_t i = 0; i < containers.size(); ++i)
    {
        PLUGIN_PROGRESS("- Compiling 3D geometry...", i, containers.size());
        callback.updateProgress("Compiling 3D geometry...", 0.5f + 0.5f * (float)(1 + i) / (float)containers.size());
        auto& container = containers[i];
        container.commitToModel();
    }
    model->applyDefaultColormap();
    if (_neuronsReportParameters.reportId != -1)
        setDefaultTransferFunction(*model, _neuronsReportParameters.valueRange);

    ModelMetadata metadata = {{"Number of Neurons", std::to_string(somas.size())},
                              {"Number of Spines", std::to_string(_nbSpines)},
                              {"SQL node filter", _details.sqlNodeFilter},
                              {"SQL section filter", _details.sqlSectionFilter},
                              {"Max distance to soma", std::to_string(_maxDistanceToSoma)},
                              {"Min soma radius", std::to_string(_minMaxSomaRadius.x)},
                              {"Max soma radius", std::to_string(_minMaxSomaRadius.y)}};

    if (!_simulationReport.description.empty())
        metadata["Simulation " + reportTypeAsString[_simulationReport.type] + " report"] =
            _simulationReport.description;

    _modelDescriptor.reset(new core::ModelDescriptor(std::move(model), _details.assemblyName, metadata));
    if (!_modelDescriptor)
        PLUGIN_THROW("Neurons model could not be created");
}

void Neurons::_buildSomasOnly(Model& model, ThreadSafeContainer& container, const NeuronSomaMap& somas)
{
    uint64_t progress = 0;
    uint64_t i = 0;
    _minMaxSomaRadius = Vector2d(_details.radiusMultiplier, _details.radiusMultiplier);

    float* voltages = nullptr;
    if (_neuronsReportParameters.reportId != -1)
        voltages = static_cast<float*>(
            model.getSimulationHandler()->getFrameData(_neuronsReportParameters.initialSimulationFrame));

    for (const auto soma : somas)
    {
        PLUGIN_PROGRESS("Loading somas", progress, somas.size());
        ++progress;

        const auto useSdf =
            andCheck(static_cast<uint32_t>(_details.realismLevel), static_cast<uint32_t>(MorphologyRealismLevel::soma));
        const auto baseMaterialId =
            _details.populationColorScheme == PopulationColorScheme::id ? i * NB_MATERIALS_PER_MORPHOLOGY : 0;
        if (_details.showMembrane)
        {
            uint64_t somaUserData = NO_USER_DATA;
            const auto neuronId = soma.first;
            switch (_simulationReport.type)
            {
            case ReportType::spike:
            case ReportType::soma:
            {
                if (_simulationReport.guids.empty())
                    somaUserData = neuronId;
                else
                {
                    const auto it = _simulationReport.guids.find(neuronId);
                    somaUserData = (*it).second;
                }
                break;
            }
            }

            auto radius = _details.radiusMultiplier;
            if (voltages && _neuronsReportParameters.voltageScalingEnabled)
                radius = _details.radiusMultiplier * _neuronsReportParameters.voltageScalingRange.x *
                         std::max(0.0, voltages[somaUserData] - _neuronsReportParameters.valueRange.x);

            const Vector3d position = soma.second.position;
            container.addSphere(position, radius, baseMaterialId, useSdf, somaUserData, {},
                                Vector3f(_getDisplacementValue(DisplacementElement::morphology_soma_strength),
                                         _getDisplacementValue(DisplacementElement::morphology_soma_frequency), 0.f));
        }
        if (_details.generateInternals)
        {
            const double mitochondriaDensity =
                (soma.second.layer < MITOCHONDRIA_DENSITY.size() ? MITOCHONDRIA_DENSITY[soma.second.layer] : 0.0);

            const auto useSdf = andCheck(static_cast<uint32_t>(_details.realismLevel),
                                         static_cast<uint32_t>(MorphologyRealismLevel::internals));
            _addSomaInternals(container, baseMaterialId, soma.second.position, _details.radiusMultiplier,
                              mitochondriaDensity, useSdf, _details.radiusMultiplier);
        }
        ++i;
    }
}

void Neurons::_buildOrientations(ThreadSafeContainer& container, const NeuronSomaMap& somas)
{
    const auto radius = _details.radiusMultiplier;
    uint64_t i = 0;
    for (const auto soma : somas)
    {
        PLUGIN_PROGRESS("Loading soma orientations", i, somas.size());
        const auto baseMaterialId =
            _details.populationColorScheme == PopulationColorScheme::id ? i * NB_MATERIALS_PER_MORPHOLOGY : 0;
        _addArrow(container, soma.first, soma.second.position, soma.second.rotation, Vector4d(0, 0, 0, radius * 0.2),
                  Vector4d(0, radius, 0, radius * 0.2), NeuronSectionType::soma, baseMaterialId, 0.0);
        ++i;
    }
}

SectionSynapseMap Neurons::_buildDebugSynapses(const uint64_t neuronId, const SectionMap& sections)
{
    SectionSynapseMap synapses;
    for (const auto& section : sections)
    {
        if (static_cast<NeuronSectionType>(section.second.type) == NeuronSectionType::axon)
            continue;

        // Process points according to representation
        const auto points = _getProcessedSectionPoints(_details.morphologyRepresentation, section.second.points);
        double sectionLength = 0.0;
        doubles segmentEnds;
        for (size_t i = 0; i < points.size() - 1; ++i)
        {
            const double segmentLength = length(points[i + 1] - points[i]);
            sectionLength += segmentLength;
            segmentEnds.push_back(sectionLength);
        }
        const size_t potentialNumberOfSynapses =
            DEFAULT_DEBUG_SYNAPSE_DENSITY_RATIO * sectionLength / DEFAULT_SPINE_RADIUS + 1;
        size_t effectiveNumberOfSynapses = potentialNumberOfSynapses * (0.5 + 0.5 * rnd0());

        for (size_t i = 0; i < effectiveNumberOfSynapses; ++i)
        {
            const double distance = rnd0() * sectionLength;
            size_t segmentId = 0;
            while (distance > segmentEnds[segmentId] && segmentId < segmentEnds.size())
                ++segmentId;

            const auto preSynapticSectionId = section.first;
            const auto preSynapticSegmentId = segmentId;
            Synapse synapse;
            synapse.type = (rand() % 2 == 0 ? MorphologySynapseType::afferent : MorphologySynapseType::efferent);
            synapse.preSynapticSegmentDistance = distance - (segmentId > 0 ? segmentEnds[segmentId - 1] : 0.f);
            synapse.postSynapticNeuronId = neuronId;
            synapse.postSynapticSectionId = 0;
            synapse.postSynapticSegmentId = 0;
            synapse.postSynapticSegmentDistance = 0.0;
            synapses[preSynapticSectionId][preSynapticSegmentId].push_back(synapse);
        }
    }
    return synapses;
}

double Neurons::_addSoma(const uint64_t neuronId, const size_t somaMaterialId, const Section& section,
                         const Vector3d& somaPosition, const Quaterniond& somaRotation, const double somaRadius,
                         const uint64_t somaUserData, const double voltageScaling, ThreadSafeContainer& container,
                         Neighbours& somaNeighbours, Neighbours& sectionNeighbours)
{
    double correctedSomaRadius;
    uint64_t count = 0.0;
    // Sections connected to the soma
    if (_details.showMembrane && _details.loadSomas && section.parentId == SOMA_AS_PARENT)
    {
        auto points = section.points;
        for (uint64_t i = 0; i < points.size(); ++i)
        {
            auto& point = points[i];
            point.x *= voltageScaling;
            point.y *= voltageScaling;
            point.z *= voltageScaling;
        }

        const auto& firstPoint = points[0];
        const auto& lastPoint = points[points.size() - 1];
        auto point = firstPoint;
        if (length(lastPoint) < length(firstPoint))
            point = lastPoint;

        const bool useSdf =
            andCheck(static_cast<uint32_t>(_details.realismLevel), static_cast<uint32_t>(MorphologyRealismLevel::soma));

        const double srcRadius = _getCorrectedRadius(somaRadius * 0.75, _details.radiusMultiplier);
        const double dstRadius = _getCorrectedRadius(point.w * 0.5, _details.radiusMultiplier);

        const auto sectionType = static_cast<NeuronSectionType>(section.type);
        const bool loadSection = (sectionType == NeuronSectionType::axon && _details.loadAxon) ||
                                 (sectionType == NeuronSectionType::basal_dendrite && _details.loadBasalDendrites) ||
                                 (sectionType == NeuronSectionType::apical_dendrite && _details.loadApicalDendrites);

        if (!loadSection)
            return -1.0;

        const Vector3d dst =
            _animatedPosition(Vector4d(somaPosition + somaRotation * Vector3d(point), dstRadius), neuronId);
        const Vector3f displacement = {Vector3f(_getDisplacementValue(DisplacementElement::morphology_soma_strength),
                                                _getDisplacementValue(DisplacementElement::morphology_soma_frequency),
                                                0.f)};

        Vector3d p2 = Vector3d();
        if (voltageScaling == 1.f)
        {
            const Vector3d segmentDirection = normalize(lastPoint - firstPoint);
            const double halfDistanceToSoma = length(Vector3d(point)) * 0.5;
            const Vector3d p1 = Vector3d(point) - halfDistanceToSoma * segmentDirection;
            p2 = p1 * dstRadius / somaRadius * 0.95;
        }
        const auto src = _animatedPosition(Vector4d(somaPosition + somaRotation * p2, srcRadius), neuronId);

        correctedSomaRadius = std::max(correctedSomaRadius, length(p2)) * 2.0;

        if (_spheresRepresentation.enabled)
            container.addConeOfSpheres(src, srcRadius, dst, dstRadius, somaMaterialId, somaUserData,
                                       _spheresRepresentation.radius);
        else
        {
            const uint64_t geometryIndex = container.addCone(src, srcRadius, dst, dstRadius, somaMaterialId, useSdf,
                                                             somaUserData, {}, displacement);
            somaNeighbours.insert(geometryIndex);
            sectionNeighbours.insert(geometryIndex);

            if (!useSdf)
                container.addSphere(dst, dstRadius, somaMaterialId, useSdf, somaUserData);
        }
        ++count;
    }
    return count == 0 ? somaRadius : correctedSomaRadius / count;
}

void Neurons::_buildMorphology(ThreadSafeContainer& container, const uint64_t neuronId, const NeuronSoma& soma,
                               const uint64_t neuronIndex, const float* voltages)
{
    const auto& connector = DBConnector::getInstance();

    const auto& somaRotation = soma.rotation;
    const auto layer = soma.layer;
    const double mitochondriaDensity = (layer < MITOCHONDRIA_DENSITY.size() ? MITOCHONDRIA_DENSITY[layer] : 0.0);

    SectionMap sections;

    // Soma radius
    double somaRadius = _getCorrectedRadius(1.f, _details.radiusMultiplier);
    if (_details.loadAxon || _details.loadApicalDendrites || _details.loadBasalDendrites)
    {
#if 0
        sections = connector.getNeuronSections(_details.populationName, neuronId, _details.sqlSectionFilter);
#else
        sections = MemoryCache::getInstance()->getNeuronSections(connector, neuronId, _details);
#endif
        double count = 0.0;
        for (const auto& section : sections)
            if (section.second.parentId == SOMA_AS_PARENT)
            {
                const Vector3d point = section.second.points[0];
                somaRadius += 0.5 * length(point);
                count += 1.0;
            }
        if (count > 0.0)
            somaRadius /= count;
        _minMaxSomaRadius.x = std::min(_minMaxSomaRadius.x, somaRadius);
        _minMaxSomaRadius.y = std::max(_minMaxSomaRadius.y, somaRadius);
        somaRadius = _getCorrectedRadius(std::min(somaRadius, MAX_SOMA_RADIUS), _details.radiusMultiplier);
    }
    const auto somaPosition = _animatedPosition(Vector4d(soma.position, somaRadius), neuronId);

    size_t baseMaterialId;
    switch (_details.populationColorScheme)
    {
    case PopulationColorScheme::none:
        baseMaterialId = 0;
        break;
    case PopulationColorScheme::id:
        baseMaterialId = neuronIndex * NB_MATERIALS_PER_MORPHOLOGY;
        break;
    }

    size_t somaMaterialId;
    switch (_details.morphologyColorScheme)
    {
    case MorphologyColorScheme::none:
        somaMaterialId = baseMaterialId;
        break;
    case MorphologyColorScheme::section_type:
        somaMaterialId = baseMaterialId + MATERIAL_OFFSET_SOMA;
        break;
    case MorphologyColorScheme::section_orientation:
        somaMaterialId = getMaterialIdFromOrientation({1.0, 1.0, 1.0});
        break;
    case MorphologyColorScheme::distance_to_soma:
        somaMaterialId = 0;
        break;
    }

    // Soma
    uint64_t somaUserData = NO_USER_DATA;
    switch (_simulationReport.type)
    {
    case ReportType::compartment:
    {
        const auto compartments =
            connector.getNeuronSectionCompartments(_details.populationName, _neuronsReportParameters.reportId, neuronId,
                                                   0);
        if (!compartments.empty())
            somaUserData = compartments[0];
        break;
    }
    case ReportType::spike:
    case ReportType::soma:
    {
        if (_simulationReport.guids.empty())
            somaUserData = neuronId + 1;
        else
        {
            const auto it = _simulationReport.guids.find(neuronId);
            somaUserData = (*it).second + 1;
        }
        break;
    }
    }
    float voltageScaling = 1.f;
    if (voltages && _neuronsReportParameters.voltageScalingEnabled)
        voltageScaling = _neuronsReportParameters.voltageScalingRange.x *
                         std::max(0.0, voltages[somaUserData] - _neuronsReportParameters.valueRange.x);

    // Load synapses for all sections
    SectionSynapseMap synapses;
    switch (_details.synapsesType)
    {
    case morphology::MorphologySynapseType::afferent:
    {
        synapses = connector.getNeuronAfferentSynapses(_details.populationName, neuronId);
        break;
    }
    case morphology::MorphologySynapseType::debug:
    {
        synapses = _buildDebugSynapses(neuronId, sections);
        break;
    }
    }

    // Soma as spheres
    double correctedSomaRadius = 0.f;
    if (_spheresRepresentation.enabled)
        correctedSomaRadius = _addSomaAsSpheres(neuronId, somaMaterialId, sections, somaPosition, somaRotation,
                                                somaRadius, somaUserData, _details.radiusMultiplier, container);

    // Sections (dendrites and axon)
    Neighbours somaNeighbours;
    for (const auto& section : sections)
    {
        Neighbours sectionNeighbours;
        const auto sectionType = static_cast<NeuronSectionType>(section.second.type);
        const auto& points = section.second.points;
        bool useSdf = andCheck(static_cast<uint32_t>(_details.realismLevel), static_cast<uint32_t>(sectionType));

        double distanceToSoma = 0.0;
        if (_details.maxDistanceToSoma > 0.0)
            // If maxDistanceToSoma != 0, then compute actual distance from soma
            distanceToSoma = _getDistanceToSoma(sections, section.second);

        if (sectionType == NeuronSectionType::axon && !_details.loadAxon)
            continue;
        if (sectionType == NeuronSectionType::basal_dendrite && !_details.loadBasalDendrites)
            continue;
        if (sectionType == NeuronSectionType::apical_dendrite && !_details.loadApicalDendrites)
            continue;
        if (_details.morphologyRepresentation == MorphologyRepresentation::graph)
        {
            if (distanceToSoma <= _details.maxDistanceToSoma)
                _addArrow(container, neuronIndex, somaPosition, somaRotation, section.second.points[0],
                          section.second.points[section.second.points.size() - 1], sectionType, baseMaterialId,
                          distanceToSoma);
            continue;
        }

        correctedSomaRadius =
            _spheresRepresentation.enabled
                ? somaRadius
                : _addSoma(neuronId, somaMaterialId, section.second, somaPosition, somaRotation, somaRadius,
                           somaUserData, voltageScaling, container, somaNeighbours, sectionNeighbours);
        if (correctedSomaRadius < 0.f)
            continue;

        float parentRadius = section.second.points[0].w;
        if (sections.find(section.second.parentId) != sections.end())
        {
            const auto& parentSection = sections[section.second.parentId];
            const auto& parentSectionPoints = parentSection.points;
            parentRadius = parentSection.points[parentSection.points.size() - 1].w;
        }

        if (distanceToSoma <= _details.maxDistanceToSoma)
            _addSection(container, neuronId, soma.morphologyId, section.first, section.second, somaPosition,
                        somaRotation, parentRadius, baseMaterialId, mitochondriaDensity, somaUserData, synapses,
                        distanceToSoma, sectionNeighbours, voltageScaling);
    }

    if (_details.loadSomas && !_spheresRepresentation.enabled)
    {
        if (_details.showMembrane)
        {
            const bool useSdf = andCheck(static_cast<uint32_t>(_details.realismLevel),
                                         static_cast<uint32_t>(MorphologyRealismLevel::soma));
            somaNeighbours.insert(
                container.addSphere(somaPosition, correctedSomaRadius, somaMaterialId, useSdf, somaUserData, {},
                                    Vector3f(_getDisplacementValue(DisplacementElement::morphology_soma_strength),
                                             _getDisplacementValue(DisplacementElement::morphology_soma_frequency),
                                             0.f)));
            if (useSdf)
                for (const auto index : somaNeighbours)
                    container.setSDFGeometryNeighbours(index, somaNeighbours);
        }
        if (_details.generateInternals)
        {
            const bool useSdf = andCheck(static_cast<uint32_t>(_details.realismLevel),
                                         static_cast<uint32_t>(MorphologyRealismLevel::internals));
            _addSomaInternals(container, baseMaterialId, somaPosition, correctedSomaRadius, mitochondriaDensity, useSdf,
                              _details.radiusMultiplier);
        }
    }

} // namespace morphology

void Neurons::_addVaricosity(Vector4fs& points)
{
    // Reference: The cholinergic innervation develops early and rapidly in the
    // rat cerebral cortex: a quantitative immunocytochemical study
    // https://www.sciencedirect.com/science/article/abs/pii/S030645220100389X
    const uint64_t middlePointIndex = points.size() / 2;
    const auto& startPoint = points[middlePointIndex];
    const auto& endPoint = points[middlePointIndex + 1];
    const double radius = std::min(startPoint.w, startPoint.w);

    const auto sp = Vector3d(startPoint);
    const auto ep = Vector3d(endPoint);

    const Vector3d dir = ep - sp;
    const Vector3d p0 = sp + dir * 0.2;
    const Vector3d p1 =
        sp + dir * 0.5 +
        radius * Vector3d((rand() % 100 - 50) / 100.0, (rand() % 100 - 50) / 100.0, (rand() % 100 - 50) / 100.0);
    const Vector3d p2 = sp + dir * 0.8;

    auto idx = points.begin() + middlePointIndex + 1;
    idx = points.insert(idx, {p2.x, p2.y, p2.z, startPoint.w});
    idx = points.insert(idx, {p1.x, p1.y, p1.z, radius * 2.0});
    points.insert(idx, {p0.x, p0.y, p0.z, endPoint.w});
}

void Neurons::_addArrow(ThreadSafeContainer& container, const uint64_t neuronId, const Vector3d& somaPosition,
                        const Quaterniond& somaRotation, const Vector4d& srcNode, const Vector4d& dstNode,
                        const NeuronSectionType sectionType, const size_t baseMaterialId, const double distanceToSoma)
{
    size_t sectionMaterialId;
    switch (_details.morphologyColorScheme)
    {
    case MorphologyColorScheme::none:
        sectionMaterialId = sectionMaterialId = baseMaterialId;
        break;
    case MorphologyColorScheme::section_type:
        sectionMaterialId = baseMaterialId + static_cast<size_t>(sectionType);
        break;
    case MorphologyColorScheme::section_orientation:
        sectionMaterialId = getMaterialIdFromOrientation(somaRotation * Vector3d(0, 0, 1));
        break;
    case MorphologyColorScheme::distance_to_soma:
        sectionMaterialId = _getMaterialFromDistanceToSoma(_details.maxDistanceToSoma, distanceToSoma);
        break;
    }

    auto src = _animatedPosition(Vector4d(somaPosition + somaRotation * Vector3d(srcNode), srcNode.w), neuronId);
    auto dst = _animatedPosition(Vector4d(somaPosition + somaRotation * Vector3d(dstNode), dstNode.w), neuronId);

    const auto userData = neuronId;
    auto direction = dst - src;
    const auto maxRadius = _details.radiusMultiplier < 0 ? -_details.radiusMultiplier : std::max(srcNode.w, dstNode.w);
    const float radius = _details.radiusMultiplier < 0
                             ? -_details.radiusMultiplier
                             : std::min(length(direction) / 5.0, maxRadius * _details.radiusMultiplier);

    const auto d1 = normalize(direction) * (length(direction) / 2.0 - radius);
    const auto d2 = normalize(direction) * (length(direction) / 2.0 + radius);

    const bool useSdf = false;

    container.addSphere(src, radius * 0.2, sectionMaterialId, useSdf, userData);
    container.addCone(src, radius * 0.2, Vector3f(src + d1 * 0.99), radius * 0.2, sectionMaterialId, useSdf, userData);
    container.addCone(Vector3f(src + d1 * 0.99), radius * 0.2, Vector3f(src + d1), radius, sectionMaterialId, useSdf,
                      userData);
    container.addCone(Vector3f(src + d1), radius, Vector3f(src + d2), radius * 0.2, sectionMaterialId, useSdf,
                      userData);
    container.addCone(Vector3f(src + d2), radius * 0.2, dst, radius * 0.2, sectionMaterialId, useSdf, userData);
    _bounds.merge(src);
    _bounds.merge(dst);
}

void Neurons::_addSection(ThreadSafeContainer& container, const uint64_t neuronId, const uint64_t morphologyId,
                          const uint64_t sectionId, const Section& section, const Vector3d& somaPosition,
                          const Quaterniond& somaRotation, const double parentRadius, const size_t baseMaterialId,
                          const double mitochondriaDensity, const uint64_t somaUserData,
                          const SectionSynapseMap& synapses, const double distanceToSoma, const Neighbours& neighbours,
                          const float voltageScaling)
{
    const auto& connector = DBConnector::getInstance();
    const auto sectionType = static_cast<NeuronSectionType>(section.type);
    bool useSdf = false;
    switch (sectionType)
    {
    case NeuronSectionType::axon:
        useSdf =
            andCheck(static_cast<uint32_t>(_details.realismLevel), static_cast<uint32_t>(MorphologyRealismLevel::axon));
        break;
    case NeuronSectionType::apical_dendrite:
    case NeuronSectionType::basal_dendrite:
        useSdf = andCheck(static_cast<uint32_t>(_details.realismLevel),
                          static_cast<uint32_t>(MorphologyRealismLevel::dendrite));
        break;
    }
    auto userData = NO_USER_DATA;

    auto points = section.points;

    for (auto& point : points)
    {
        point.x *= voltageScaling;
        point.y *= voltageScaling;
        point.z *= voltageScaling;
    }

    points[0].w = parentRadius;

    size_t sectionMaterialId;
    switch (_details.morphologyColorScheme)
    {
    case MorphologyColorScheme::none:
        sectionMaterialId = baseMaterialId;
        break;
    case MorphologyColorScheme::section_type:
        sectionMaterialId = baseMaterialId + section.type;
        break;
    case MorphologyColorScheme::section_orientation:
        sectionMaterialId = getMaterialIdFromOrientation(points[points.size() - 1] - points[0]);
        break;
    }

    // Process points according to representation
    auto localPoints = _getProcessedSectionPoints(_details.morphologyRepresentation, points);

    // Generate varicosities
    const auto middlePointIndex = localPoints.size() / 2;
    const bool addVaricosity = _details.generateVaricosities && sectionType == NeuronSectionType::axon &&
                               localPoints.size() > nbMinSegmentsForVaricosity;
    if (addVaricosity)
        _addVaricosity(localPoints);

    // Section surface and volume
    double sectionLength = 0.0;
    double sectionVolume = 0.0;
    uint64_ts compartments;
    switch (_simulationReport.type)
    {
    case ReportType::undefined:
        userData = NO_USER_DATA;
        break;
    case ReportType::spike:
    case ReportType::soma:
    {
        userData = somaUserData;
        break;
    }
    case ReportType::compartment:
    {
        compartments = connector.getNeuronSectionCompartments(_details.populationName,
                                                              _neuronsReportParameters.reportId, neuronId, sectionId);
        break;
    }
    }

    // Section synapses
    SegmentSynapseMap segmentSynapses;
    const auto it = synapses.find(sectionId);
    if (it != synapses.end())
        segmentSynapses = (*it).second;

    // Section points
    Neighbours sectionNeighbours = neighbours;
    uint64_t previousGeometryIndex = 0;
    for (uint64_t i = 0; i < localPoints.size() - 1; ++i)
    {
        if (!compartments.empty())
        {
            const uint64_t compartmentIndex = i * compartments.size() / localPoints.size();
            userData = compartments[compartmentIndex];
        }

        const auto& srcPoint = localPoints[i];
        const auto& dstPoint = localPoints[i + 1];

        if (srcPoint == dstPoint)
            // It sometimes occurs that points are duplicated, resulting in a zero-length segment that can ignored
            continue;

        const double srcRadius = _getCorrectedRadius(srcPoint.w * 0.5, _details.radiusMultiplier);
        const auto src =
            _animatedPosition(Vector4d(somaPosition + somaRotation * Vector3d(srcPoint), srcRadius), neuronId);

        const double dstRadius = _getCorrectedRadius(dstPoint.w * 0.5, _details.radiusMultiplier);
        const auto dst =
            _animatedPosition(Vector4d(somaPosition + somaRotation * Vector3d(dstPoint), dstRadius), neuronId);
        const double sampleLength = length(dstPoint - srcPoint);
        sectionLength += sampleLength;

        if (_details.showMembrane)
        {
            Vector3f displacement{std::min(std::min(srcRadius, dstRadius) * 0.5f,
                                           _getDisplacementValue(DisplacementElement::morphology_section_strength)),
                                  _getDisplacementValue(DisplacementElement::morphology_section_frequency), 0.f};

            size_t materialId = _details.morphologyColorScheme == MorphologyColorScheme::distance_to_soma
                                    ? _getMaterialFromDistanceToSoma(_details.maxDistanceToSoma, distanceToSoma)

                                    : sectionMaterialId;

            // Varicosity (axon only)
            if (addVaricosity && _details.morphologyColorScheme == MorphologyColorScheme::section_type)
            {
                if (i > middlePointIndex && i < middlePointIndex + 3)
                {
                    materialId = baseMaterialId + MATERIAL_OFFSET_VARICOSITY;
                    displacement =
                        Vector3f(2.f * srcRadius *
                                     _getDisplacementValue(DisplacementElement::morphology_section_strength),
                                 _getDisplacementValue(DisplacementElement::morphology_section_frequency) / srcRadius,
                                 0.f);
                }
                if (i == middlePointIndex + 1 || i == middlePointIndex + 3)
                    sectionNeighbours = {};
                if (i == middlePointIndex + 1)
                    _varicosities[neuronId].push_back(dst);
            }

            // Synapses
            const auto it = segmentSynapses.find(i);
            if (it != segmentSynapses.end())
            {
                const auto synapses = (*it).second;
                PLUGIN_INFO(3,
                            "Adding " << synapses.size() << " spines to segment " << i << " of section " << sectionId);
                for (const auto& synapse : synapses)
                {
                    const size_t spineMaterialId =
                        _details.morphologyColorScheme == MorphologyColorScheme::section_type
                            ? baseMaterialId + (synapse.type == MorphologySynapseType::afferent
                                                    ? MATERIAL_OFFSET_AFFERENT_SYNAPSE
                                                    : MATERIAL_OFFSET_EFFERENT_SYNAPSE)
                            : materialId;
                    const Vector3d segmentDirection = dst - src;
                    const double radiusInSegment =
                        srcRadius + ((1.0 / length(segmentDirection)) * synapse.preSynapticSegmentDistance) *
                                        (dstRadius - srcRadius);
                    const Vector3d positionInSegment =
                        src + normalize(segmentDirection) * synapse.preSynapticSegmentDistance;
                    _addSpine(container, userData, morphologyId, sectionId, synapse, spineMaterialId, positionInSegment,
                              radiusInSegment);
                }
            }

            if (_spheresRepresentation.enabled)
                container.addConeOfSpheres(src, srcRadius, dst, dstRadius, materialId, userData,
                                           _spheresRepresentation.radius);
            else
            {
                if (!useSdf)
                    container.addSphere(dst, dstRadius, materialId, useSdf, userData);

                const uint64_t geometryIndex =
                    container.addCone(src, srcRadius, dst, dstRadius, materialId, useSdf, userData, {}, displacement);
                previousGeometryIndex = geometryIndex;
                sectionNeighbours = {geometryIndex};
            }

            // Stop if distance to soma in greater than the specified max value
            _maxDistanceToSoma = std::max(_maxDistanceToSoma, distanceToSoma + sectionLength);
            if (_details.maxDistanceToSoma > 0.0 && distanceToSoma + sectionLength >= _details.maxDistanceToSoma)
                break;
        }
        sectionVolume += coneVolume(sampleLength, srcRadius, dstRadius);

        _bounds.merge(srcPoint);
        _bounds.merge(dstPoint);
    }

    if (sectionType == NeuronSectionType::axon)
    {
        if (_details.generateInternals)
            _addSectionInternals(container, neuronId, somaPosition, somaRotation, sectionLength, sectionVolume,
                                 localPoints, mitochondriaDensity, baseMaterialId);

        if (_details.generateExternals)
            _addAxonMyelinSheath(container, neuronId, somaPosition, somaRotation, sectionLength, localPoints,
                                 mitochondriaDensity, baseMaterialId);
    }
}

void Neurons::_addSectionInternals(ThreadSafeContainer& container, const uint64_t neuronId,
                                   const Vector3d& somaPosition, const Quaterniond& somaRotation,
                                   const double sectionLength, const double sectionVolume, const Vector4fs& points,
                                   const double mitochondriaDensity, const size_t baseMaterialId)
{
    if (mitochondriaDensity == 0.0)
        return;

    const auto useSdf = andCheck(static_cast<uint32_t>(_details.realismLevel),
                                 static_cast<uint32_t>(MorphologyRealismLevel::internals));

    // Add mitochondria (density is per section, not for the full axon)
    const size_t nbMaxMitochondrionSegments = sectionLength / mitochondrionSegmentSize;
    const double indexRatio = double(points.size()) / double(nbMaxMitochondrionSegments);

    double mitochondriaVolume = 0.0;
    const size_t mitochondrionMaterialId = baseMaterialId + MATERIAL_OFFSET_MITOCHONDRION;

    uint64_t nbSegments = _getNbMitochondrionSegments();
    int64_t mitochondrionSegment = -(rand() % (1 + nbMaxMitochondrionSegments / 10));
    double previousRadius;
    Vector3d previousPosition;

    uint64_t geometryIndex = 0;
    Vector3d randomPosition{points[0].w * (rand() % 100 - 50) / 200.0, points[0].w * (rand() % 100 - 50) / 200.0,
                            points[0].w * (rand() % 100 - 50) / 200.0};
    for (size_t step = 0; step < nbMaxMitochondrionSegments; ++step)
    {
        if (mitochondriaVolume < sectionVolume * mitochondriaDensity && mitochondrionSegment >= 0 &&
            mitochondrionSegment < nbSegments)
        {
            const uint64_t srcIndex = uint64_t(step * indexRatio);
            const uint64_t dstIndex = uint64_t(step * indexRatio) + 1;
            if (dstIndex < points.size())
            {
                const auto srcSample = _animatedPosition(points[srcIndex], neuronId);
                const auto dstSample = _animatedPosition(points[dstIndex], neuronId);
                const double srcRadius = _getCorrectedRadius(points[srcIndex].w * 0.5, _details.radiusMultiplier);
                const Vector3d srcPosition{srcSample.x + srcRadius * (rand() % 100 - 50) / 500.0,
                                           srcSample.y + srcRadius * (rand() % 100 - 50) / 500.0,
                                           srcSample.z + srcRadius * (rand() % 100 - 50) / 500.0};
                const double dstRadius = _getCorrectedRadius(points[dstIndex].w * 0.5, _details.radiusMultiplier);
                const Vector3d dstPosition{dstSample.x + dstRadius * (rand() % 100 - 50) / 500.0,
                                           dstSample.y + dstRadius * (rand() % 100 - 50) / 500.0,
                                           dstSample.z + dstRadius * (rand() % 100 - 50) / 500.0};

                const Vector3d direction = dstPosition - srcPosition;
                const Vector3d position = srcPosition + randomPosition + direction * (step * indexRatio - srcIndex);
                const double radius = (1.0 + (rand() % 1000 - 500) / 5000.0) * mitochondrionRadius *
                                      0.5; // Make twice smaller than in the soma

                Neighbours neighbours;
                if (mitochondrionSegment != 0)
                    neighbours = {geometryIndex};

                if (!useSdf && !_spheresRepresentation.enabled)
                    container.addSphere(somaPosition + somaRotation * position, radius, mitochondrionMaterialId,
                                        NO_USER_DATA);

                if (mitochondrionSegment > 0)
                {
                    Neighbours neighbours = {};
                    if (mitochondrionSegment > 1)
                        neighbours = {geometryIndex};
                    const auto srcPosition =
                        _animatedPosition(Vector4d(somaPosition + somaRotation * position, radius), neuronId);
                    const auto dstPosition =
                        _animatedPosition(Vector4d(somaPosition + somaRotation * previousPosition, previousRadius),
                                          neuronId);

                    if (_spheresRepresentation.enabled)
                        container.addConeOfSpheres(srcPosition, radius, dstPosition, previousRadius,
                                                   mitochondrionMaterialId, NO_USER_DATA,
                                                   _spheresRepresentation.radius);
                    else
                        geometryIndex = container.addCone(
                            srcPosition, radius, dstPosition, previousRadius, mitochondrionMaterialId, useSdf,
                            NO_USER_DATA, neighbours,
                            Vector3f(radius *
                                         _getDisplacementValue(DisplacementElement::morphology_mitochondrion_strength) *
                                         2.0,
                                     radius *
                                         _getDisplacementValue(DisplacementElement::morphology_mitochondrion_frequency),
                                     0.f));

                    mitochondriaVolume += coneVolume(length(position - previousPosition), radius, previousRadius);
                }

                previousPosition = position;
                previousRadius = radius;
            }
        }
        ++mitochondrionSegment;

        if (mitochondrionSegment == nbSegments)
        {
            mitochondrionSegment = -(rand() % (1 + nbMaxMitochondrionSegments / 10));
            nbSegments = _getNbMitochondrionSegments();
            const auto index = uint64_t(step * indexRatio);
            randomPosition =
                Vector3d(points[index].w * (rand() % 100 - 50) / 200.0, points[index].w * (rand() % 100 - 50) / 200.0,
                         points[index].w * (rand() % 100 - 50) / 200.0);
        }
    }
}

void Neurons::_addAxonMyelinSheath(ThreadSafeContainer& container, const uint64_t neuronId,
                                   const Vector3d& somaPosition, const Quaterniond& somaRotation,
                                   const double sectionLength, const Vector4fs& points,
                                   const double mitochondriaDensity, const size_t baseMaterialId)
{
    if (sectionLength == 0 || points.empty())
        return;

    const bool useSdf = andCheck(static_cast<uint32_t>(_details.realismLevel),
                                 static_cast<uint32_t>(MorphologyRealismLevel::externals));

    const size_t myelinSteathMaterialId = baseMaterialId + MATERIAL_OFFSET_MYELIN_SHEATH;

    if (sectionLength < 2 * myelinSteathLength)
        return;

    const uint64_t nbPoints = points.size();
    if (nbPoints < NB_MYELIN_FREE_SEGMENTS)
        return;

    // Average radius for myelin steath
    const auto myelinScale = myelinSteathRadiusRatio;
    double srcRadius = 0.0;
    for (const auto& point : points)
        srcRadius += _getCorrectedRadius(point.w * 0.5 * myelinScale, _details.radiusMultiplier);
    srcRadius /= points.size();

    uint64_t i = NB_MYELIN_FREE_SEGMENTS; // Ignore first 3 segments
    while (i < nbPoints - NB_MYELIN_FREE_SEGMENTS)
    {
        // Start surrounding segments with myelin steaths
        const auto& srcPoint = points[i];
        const auto srcPosition =
            _animatedPosition(Vector4d(somaPosition + somaRotation * Vector3d(srcPoint), srcRadius), neuronId);

        if (!useSdf)
            container.addSphere(srcPosition, srcRadius, myelinSteathMaterialId, NO_USER_DATA);

        double currentLength = 0;
        auto previousPosition = srcPosition;
        auto previousRadius = srcRadius;
        const Vector3f displacement{srcRadius *
                                        _getDisplacementValue(DisplacementElement::morphology_myelin_steath_strength),
                                    _getDisplacementValue(DisplacementElement::morphology_myelin_steath_frequency),
                                    0.f};
        Neighbours neighbours;

        while (currentLength < myelinSteathLength && i < nbPoints - NB_MYELIN_FREE_SEGMENTS)
        {
            ++i;
            const auto& dstPoint = points[i];
            const auto dstRadius = srcRadius;
            const auto dstPosition =
                _animatedPosition(Vector4d(somaPosition + somaRotation * Vector3d(dstPoint), dstRadius), neuronId);

            currentLength += length(dstPosition - previousPosition);
            if (!useSdf && !_spheresRepresentation.enabled)
                container.addSphere(dstPosition, srcRadius, myelinSteathMaterialId, NO_USER_DATA);

            if (_spheresRepresentation.enabled)
                container.addConeOfSpheres(dstPosition, dstRadius, previousPosition, previousRadius,
                                           myelinSteathMaterialId, NO_USER_DATA, _spheresRepresentation.radius);
            else
            {
                const auto geometryIndex =
                    container.addCone(dstPosition, dstRadius, previousPosition, previousRadius, myelinSteathMaterialId,
                                      useSdf, NO_USER_DATA, neighbours, displacement);
                neighbours.insert(geometryIndex);
            }
            previousPosition = dstPosition;
            previousRadius = dstRadius;
        }
        i += NB_MYELIN_FREE_SEGMENTS; // Leave free segments between
                                      // myelin steaths
    }
}

void Neurons::_addSpine(ThreadSafeContainer& container, const uint64_t userData, const uint64_t morphologyId,
                        const uint64_t sectionId, const Synapse& synapse, const size_t SpineMaterialId,
                        const Vector3d& preSynapticSurfacePosition, const double radiusAtSurfacePosition)
{
    const double radius = DEFAULT_SPINE_RADIUS;
    const double spineScale = 0.25;
    const double spineLength = 0.4 + 0.2 * rnd1();

    const auto spineDisplacement =
        Vector3d(_getDisplacementValue(DisplacementElement::morphology_spine_strength),
                 _getDisplacementValue(DisplacementElement::morphology_spine_frequency), 0.0);

    const auto spineSmallRadius = std::max(spineDisplacement.x, radius * spineRadiusRatio * 0.5 * spineScale);
    const auto spineBaseRadius = std::max(spineDisplacement.x, radius * spineRadiusRatio * 0.75 * spineScale);
    const auto spineLargeRadius = std::max(spineDisplacement.x, radius * spineRadiusRatio * 2.5 * spineScale);
    const auto sectionDisplacementAmplitude = _getDisplacementValue(DisplacementElement::morphology_section_strength);

    const auto direction = normalize(Vector3d(rnd1(), rnd1(), rnd1()));
    const auto origin = preSynapticSurfacePosition +
                        normalize(direction) * std::max(0.0, radiusAtSurfacePosition - sectionDisplacementAmplitude);
    const auto target = preSynapticSurfacePosition + normalize(direction) * (radiusAtSurfacePosition + spineLength);

    // Create random shape between origin and target
    auto middle = (target + origin) / 2.0;
    const double d = length(target - origin) / 2.0;
    middle += Vector3f(d * rnd1(), d * rnd1(), d * rnd1());
    const float spineMiddleRadius = std::max(0.01, spineSmallRadius + d * 0.1 * rnd1());

    Neighbours neighbours;

    const bool useSdf =
        andCheck(static_cast<uint32_t>(_details.realismLevel), static_cast<uint32_t>(MorphologyRealismLevel::spine));

    if (!useSdf && !_spheresRepresentation.enabled)
    {
        container.addSphere(target, spineLargeRadius, SpineMaterialId, useSdf, userData);
        neighbours.insert(container.addSphere(middle, spineMiddleRadius, SpineMaterialId, useSdf, userData, neighbours,
                                              spineDisplacement));
    }

    if (middle != origin)
    {
        if (_spheresRepresentation.enabled)
            container.addConeOfSpheres(origin, spineSmallRadius, middle, spineMiddleRadius, SpineMaterialId, userData,
                                       _spheresRepresentation.radius);
        else
            container.addCone(origin, spineSmallRadius, middle, spineMiddleRadius, SpineMaterialId, useSdf, userData,
                              neighbours, spineDisplacement);
    }

    if (middle != target)
    {
        if (_spheresRepresentation.enabled)
            container.addConeOfSpheres(middle, spineMiddleRadius, target, spineLargeRadius, SpineMaterialId, userData,
                                       _spheresRepresentation.radius);
        else
            container.addCone(middle, spineMiddleRadius, target, spineLargeRadius, SpineMaterialId, useSdf, userData,
                              neighbours, spineDisplacement);
    }

    ++_nbSpines;
}

Vector4ds Neurons::getNeuronSectionPoints(const uint64_t neuronId, const uint64_t sectionId)
{
    const auto& connector = DBConnector::getInstance();
    const auto neurons = connector.getNeurons(_details.populationName, "guid=" + std::to_string(neuronId));

    if (neurons.empty())
        PLUGIN_THROW("Neuron " + std::to_string(neuronId) + " does not exist");
    const auto& neuron = neurons.begin()->second;
    const auto sections = connector.getNeuronSections(_details.populationName, neuronId);

    if (sections.empty())
        PLUGIN_THROW("Section " + std::to_string(sectionId) + " does not exist for neuron " + std::to_string(neuronId));
    const auto section = sections.begin()->second;
    Vector4ds points;
    for (const auto& point : section.points)
    {
        const Vector3d position = _scale * (neuron.position + neuron.rotation * Vector3d(point));
        const double radius = point.w;
        points.push_back({position.x, position.y, position.z, radius});
    }
    return points;
}

Vector3ds Neurons::getNeuronVaricosities(const uint64_t neuronId)
{
    if (_varicosities.find(neuronId) == _varicosities.end())
        PLUGIN_THROW("Neuron " + std::to_string(neuronId) + " does not exist");
    return _varicosities[neuronId];
}

void Neurons::_attachSimulationReport(Model& model, const uint64_t nbNeurons)
{
    // Simulation report
    std::string sqlNodeFilter = _details.sqlNodeFilter;
    const auto& connector = DBConnector::getInstance();
    switch (_simulationReport.type)
    {
    case ReportType::undefined:
        PLUGIN_DEBUG("No report attached to the geometry");
        break;
    case ReportType::spike:
    {
        PLUGIN_INFO(1,
                    "Initialize spike simulation handler and restrain "
                    "guids to the simulated ones");
        auto handler =
            std::make_shared<SpikeSimulationHandler>(_details.populationName, _neuronsReportParameters.reportId);
        model.setSimulationHandler(handler);
        break;
    }
    case ReportType::soma:
    {
        PLUGIN_INFO(1,
                    "Initialize soma simulation handler and restrain guids "
                    "to the simulated ones");
        auto handler =
            std::make_shared<SomaSimulationHandler>(_details.populationName, _neuronsReportParameters.reportId);
        model.setSimulationHandler(handler);
        break;
    }
    case ReportType::compartment:
    {
        PLUGIN_INFO(1,
                    "Initialize compartment simulation handler and restrain "
                    "guids to the simulated ones");
        auto handler =
            std::make_shared<CompartmentSimulationHandler>(_details.populationName, _neuronsReportParameters.reportId);
        model.setSimulationHandler(handler);
        break;
    }
    }
}

} // namespace morphology
} // namespace bioexplorer
