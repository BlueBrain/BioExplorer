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

#include "SynapseEfficacy.h"

#include <plugin/common/Logs.h>
#include <plugin/common/ThreadSafeContainer.h>
#include <plugin/common/Utils.h>

#include <plugin/io/db/DBConnector.h>

#include <brayns/engineapi/Model.h>
#include <brayns/engineapi/Scene.h>

namespace bioexplorer
{
namespace connectomics
{
using namespace common;
using namespace io;
using namespace db;

SynapseEfficacy::SynapseEfficacy(Scene& scene,
                                 const SynapseEfficacyDetails& details)
    : Node()
    , _details(details)
    , _scene(scene)
{
    Timer chrono;
    _buildModel();
    PLUGIN_TIMER(chrono.elapsed(), "Synapse efficacy model loaded");
}

void SynapseEfficacy::_buildModel()
{
    if (_modelDescriptor)
        _scene.removeModel(_modelDescriptor->getModelID());

    auto model = _scene.createModel();
    ThreadSafeContainer container(*model);

    const auto synapsePositions =
        DBConnector::getInstance().getSynapseEfficacyPositions(
            _details.populationName, _details.sqlFilter);

    const auto nbSynapses = synapsePositions.size();
    const size_t materialId = 0;
    const bool useSdf = false;
    uint64_t progressStep = synapsePositions.size() / 100 + 1;
    uint64_t i = 0;
    for (const auto& position : synapsePositions)
    {
        const auto src = getAlignmentToGrid(_details.alignToGrid, position);
        container.addSphere(src, _details.radius, materialId, useSdf, i);
        if (i % progressStep == 0)
            PLUGIN_PROGRESS("Loading " << i << "/" << nbSynapses << " synapses",
                            i, nbSynapses);
        ++i;
    }

    container.commitToModel();
    PLUGIN_INFO(1, "");

    const ModelMetadata metadata = {{"Number of synapses",
                                     std::to_string(nbSynapses)},
                                    {"SQL filter", _details.sqlFilter}};

    _modelDescriptor.reset(new brayns::ModelDescriptor(std::move(model),
                                                       _details.assemblyName,
                                                       metadata));
    if (_modelDescriptor)
        _scene.addModel(_modelDescriptor);
    else
        PLUGIN_THROW(
            "Synapse efficacy model could not be created for "
            "population " +
            _details.populationName);
}
} // namespace connectomics
} // namespace bioexplorer
