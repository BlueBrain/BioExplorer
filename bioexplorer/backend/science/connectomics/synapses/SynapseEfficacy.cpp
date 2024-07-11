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

#include "SynapseEfficacy.h"

#include <science/common/Logs.h>
#include <science/common/ThreadSafeContainer.h>
#include <science/common/Utils.h>

#include <science/io/db/DBConnector.h>

#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Scene.h>

using namespace core;

namespace bioexplorer
{
using namespace common;
using namespace details;
using namespace io;
using namespace db;

namespace connectomics
{
SynapseEfficacy::SynapseEfficacy(Scene& scene, const SynapseEfficacyDetails& details, const Vector3d& position,
                                 const Quaterniond& rotation)
    : SDFGeometries(details.alignToGrid, position, rotation)
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
    ThreadSafeContainer container(*model, _alignToGrid, _position, _rotation);

    const auto synapsePositions =
        DBConnector::getInstance().getSynapseEfficacyPositions(_details.populationName, _details.sqlFilter);

    const auto nbSynapses = synapsePositions.size();
    const size_t materialId = 0;
    const bool useSdf = false;
    uint64_t progressStep = synapsePositions.size() / 100 + 1;
    uint64_t i = 0;
    for (const auto& position : synapsePositions)
    {
        container.addSphere(position, _details.radius, materialId, useSdf, i);
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
} // namespace connectomics
} // namespace bioexplorer
