/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include "Synaptome.h"

#include <science/common/Logs.h>
#include <science/common/ThreadSafeContainer.h>
#include <science/common/Utils.h>

#include <science/io/db/DBConnector.h>

#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Scene.h>

#include <omp.h>

using namespace core;

namespace bioexplorer
{
using namespace common;
using namespace details;
using namespace io;
using namespace db;

namespace connectomics
{
const float springConstant = 0.01f;
const float idealEdgeLength = 100.0f;

Synaptome::Synaptome(Scene& scene, const SynaptomeDetails& details, const Vector3d& position,
                     const Quaterniond& rotation, const LoaderProgress& callback)
    : SDFGeometries(NO_GRID_ALIGNMENT, position, rotation)
    , _details(details)
    , _scene(scene)
{
    Timer chrono;
    _buildModel(callback);
    PLUGIN_TIMER(chrono.elapsed(), "White matter loaded");
}

void Synaptome::_addNode(const uint64_t id, const Vector3f& position, float mass)
{
    _nodes[id] = {position, Vector3f(), mass};
}

void Synaptome::_addEdge(uint64_t source, uint64_t target, const core::Vector3f& direction)
{
    _edges.push_back({source, target});
    _nodes[source].position += direction;
    _nodes[source].mass++;
    _nodes[target].position -= direction;
    _nodes[target].mass++;
}

void Synaptome::_buildModel(const LoaderProgress& callback)
{
    if (_modelDescriptor)
        _scene.removeModel(_modelDescriptor->getModelID());

    auto model = _scene.createModel();

    auto& connector = DBConnector::getInstance();
    const auto nbDBConnections = connector.getNbConnections();
    const auto somas = connector.getNeurons(_details.populationName, _details.sqlNodeFilter);
    const auto synaptome = connector.getSynaptome(_details.populationName, _details.sqlEdgeFilter);

    ThreadSafeContainer container(*model, _alignToGrid, _position, _rotation);
    const size_t nodeMaterialId = 0;
    const size_t edgeMaterialId = 1;
    const bool useSdf = false;
    const float nodeRadius = _details.radius;
    const float edgeRadius = _details.radius / 10.f;

    for (const auto& soma : somas)
        _addNode(soma.first, soma.second.position, 1.f);

    for (const auto& edge : synaptome)
    {
        const auto itSrc = somas.find(edge.first);
        const auto itDst = somas.find(edge.second);
        if (itSrc != somas.end() && itDst != somas.end())
        {
            const auto direction = _details.force * ((*itDst).second.position - (*itSrc).second.position);
            _addEdge(edge.first, edge.second, direction);
        }
    }

    // Draw nodes
    for (const auto& node : _nodes)
        container.addSphere(node.second.position, nodeRadius, nodeMaterialId, useSdf);

    // Draw edges
    for (const auto& edge : _edges)
        container.addCone(_nodes[edge.x].position, edgeRadius, _nodes[edge.y].position, edgeRadius,
                          static_cast<size_t>(_nodes[edge.x].mass + 1), useSdf);
    container.commitToModel();
    model->applyDefaultColormap();

    const ModelMetadata metadata = {{"Number of nodes", std::to_string(_nodes.size())},
                                    {"Number of edges", std::to_string(_edges.size())},
                                    {"SQL node filter", _details.sqlNodeFilter},
                                    {"SQL edge filter", _details.sqlEdgeFilter}};
    _modelDescriptor.reset(new core::ModelDescriptor(std::move(model), _details.assemblyName, metadata));
    if (!_modelDescriptor)
        PLUGIN_THROW(
            "Synaptome model could not be created for "
            "population " +
            _details.populationName);
}
} // namespace connectomics
} // namespace bioexplorer
