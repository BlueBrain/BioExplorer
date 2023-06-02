/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
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

#include "Synapses.h"

#include <plugin/common/Logs.h>
#include <plugin/common/ThreadSafeContainer.h>
#include <plugin/common/Utils.h>

#include <plugin/io/db/DBConnector.h>

// Platform
#include <platform/core/common/CommonTypes.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Scene.h>

namespace bioexplorer
{
namespace morphology
{
using namespace common;
using namespace io;
using namespace db;

Synapses::Synapses(Scene& scene, const SynapsesDetails& details, const Vector3d& assemblyPosition,
                   const Quaterniond& assemblyRotation)
    : Morphologies(0, assemblyPosition, assemblyRotation)
    , _details(details)
    , _scene(scene)
{
    Timer chrono;
    _buildModel();
    PLUGIN_TIMER(chrono.elapsed(), "Synapses model loaded");
}

double Synapses::_getDisplacementValue(const DisplacementElement& element)
{
    const auto params = _details.displacementParams;
    switch (element)
    {
    case DisplacementElement::morphology_spine_strength:
        return valueFromDoubles(params, 0, DEFAULT_MORPHOLOGY_SPINE_STRENGTH);
    case DisplacementElement::morphology_spine_frequency:
        return valueFromDoubles(params, 1, DEFAULT_MORPHOLOGY_SPINE_FREQUENCY);
    default:
        PLUGIN_THROW("Invalid displacement element");
    }
}

void Synapses::_buildModel()
{
    if (_modelDescriptor)
        _scene.removeModel(_modelDescriptor->getModelID());

    auto model = _scene.createModel();
    ThreadSafeContainer container(*model, _alignToGrid, _position, _rotation);

    const auto synapses = DBConnector::getInstance().getSynapses(_details.populationName, _details.sqlFilter);

    const auto nbSynapses = synapses.size();
    const size_t materialId = 0;
    uint64_t i = 0;
    uint64_t progressStep = synapses.size() / 100 + 1;
    for (const auto& synapse : synapses)
    {
        if (_details.representation == SynapseRepresentation::spine)
            _addSpine(container, synapse.first, synapse.second, materialId);
        else
            container.addSphere(synapse.second.preSynapticSurfacePosition,
                                DEFAULT_SPINE_RADIUS * _details.radiusMultiplier, materialId, false, i);
        if (i % progressStep == 0)
            PLUGIN_PROGRESS("Loading " << i << "/" << nbSynapses << " synapses", i, nbSynapses);
        ++i;
    }

    container.commitToModel();
    PLUGIN_INFO(1, "");

    const ModelMetadata metadata = {{"Number of synapses", std::to_string(nbSynapses)},
                                    {"SQL filter", _details.sqlFilter}};

    _modelDescriptor.reset(new core::ModelDescriptor(std::move(model), _details.assemblyName, metadata));
    if (_modelDescriptor)
        _scene.addModel(_modelDescriptor);
    else
        PLUGIN_THROW(
            "Synapse efficacy model could not be created for "
            "population " +
            _details.populationName);
}

void Synapses::_addSpine(ThreadSafeContainer& container, const uint64_t guid, const Synapse& synapse,
                         const size_t SpineMaterialId)
{
    const double radius = _details.radiusMultiplier * DEFAULT_SPINE_RADIUS;

    const auto spineSmallRadius = radius * spineRadiusRatio * 0.5;
    const auto spineBaseRadius = radius * spineRadiusRatio * 0.75;
    const auto spineLargeRadius = radius * spineRadiusRatio * 2.5;

    const auto direction =
        Vector3d((rand() % 200 - 100) / 100.0, (rand() % 200 - 100) / 100.0, (rand() % 200 - 100) / 100.0);
    const auto l = 6.f * radius;

    const auto origin = synapse.postSynapticSurfacePosition;
    // const auto target = origin + normalize(direction) * l;
    const auto target = synapse.preSynapticSurfacePosition;

    // Create random shape between origin and target
    auto middle = (target + origin) / 2.0;
    const double d = length(target - origin) / 1.5;
    const auto id = guid * 4;
    middle += Vector3f(d * rnd2(id), d * rnd2(id + 1), d * rnd2(id + 2));
    const float spineMiddleRadius = spineSmallRadius + d * 0.1 * rnd2(id + 3);

    const auto displacement = Vector3f(_getDisplacementValue(DisplacementElement::morphology_spine_strength),
                                       _getDisplacementValue(DisplacementElement::morphology_spine_frequency), 0.f);

    const bool useSdf =
        andCheck(static_cast<uint32_t>(_details.realismLevel), static_cast<uint32_t>(MorphologyRealismLevel::spine));
    Neighbours neighbours;
    neighbours.insert(container.addSphere(middle, spineMiddleRadius, SpineMaterialId, useSdf, NO_USER_DATA, neighbours,
                                          displacement));
    if (middle != origin)
        container.addCone(origin, spineSmallRadius, middle, spineMiddleRadius, SpineMaterialId, useSdf, NO_USER_DATA,
                          neighbours, displacement);
    if (middle != target)
        container.addCone(middle, spineMiddleRadius, target, spineLargeRadius, SpineMaterialId, useSdf, NO_USER_DATA,
                          neighbours, displacement);
}

} // namespace morphology
} // namespace bioexplorer
