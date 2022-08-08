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

#include "Atlas.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/Logs.h>
#include <plugin/common/ThreadSafeContainer.h>
#include <plugin/common/Utils.h>

#include <plugin/io/db/DBConnector.h>

#include <brayns/common/Timer.h>
#include <brayns/engineapi/Model.h>
#include <brayns/engineapi/Scene.h>

namespace bioexplorer
{
namespace atlas
{
using namespace common;
using namespace io;
using namespace db;

Atlas::Atlas(Scene& scene, const AtlasDetails& details)
    : Node(doublesToVector3d(details.scale))
    , _details(details)
    , _scene(scene)
{
    Timer chrono;
    _load();
    PLUGIN_TIMER(chrono.elapsed(), "Atlas loaded");
}

void Atlas::_load()
{
    if (_modelDescriptor)
        _scene.removeModel(_modelDescriptor->getModelID());

    auto model = _scene.createModel();

    auto& connector = DBConnector::getInstance();
    const auto regions = connector.getAtlasRegions(_details.regionSqlFilter);

    ThreadSafeContainers containers;
    uint64_t counter = 0;
    uint64_t nbCells = 0;

    const auto nbDBConnections = DBConnector::getInstance().getNbConnections();
    uint64_t index;
#pragma omp parallel for num_threads(nbDBConnections)
    for (index = 0; index < regions.size(); ++index)
    {
        ThreadSafeContainer container(*model, false);

        const auto region = regions[index];
        if (_details.loadCells)
        {
            const auto cells =
                connector.getAtlasCells(region, _details.cellSqlFilter);
            for (const auto& cell : cells)
                container.addSphere(cell.second.position, _details.cellRadius,
                                    cell.second.region);
#pragma omp critical
            nbCells += cells.size();
        }

        if (_details.loadMeshes)
        {
            const Vector3d position = doublesToVector3d(_details.meshPosition);
            const Quaterniond rotation =
                doublesToQuaterniond(_details.meshRotation);
            const Vector3d scale = doublesToVector3d(_details.meshScale);
            auto mesh = connector.getAtlasMesh(region);
            for (auto& vertex : mesh.vertices)
            {
                const Vector3d p = Vector3d(vertex) + position;
                const Vector3d r = rotation * p;
                vertex = scale * r;
            }
            container.addMesh(region, mesh);
        }

#pragma omp critical
        containers.push_back(container);

#pragma omp critical
        ++counter;

#pragma omp critical
        PLUGIN_PROGRESS("Loading " << regions.size() << " regions", counter,
                        regions.size());
    }

    for (uint64_t i = 0; i < containers.size(); ++i)
    {
        PLUGIN_PROGRESS("- Compiling 3D geometry...", i, containers.size());
        auto& container = containers[i];
        container.commitToModel();
    }
    ModelMetadata metadata = {{"Number of regions",
                               std::to_string(regions.size())},
                              {"Number of cells", std::to_string(nbCells)}};

    _modelDescriptor.reset(new brayns::ModelDescriptor(std::move(model),
                                                       _details.assemblyName,
                                                       metadata));
    if (_modelDescriptor)
        _scene.addModel(_modelDescriptor);
    else
        PLUGIN_THROW("Atlas model could not be created");
}

} // namespace atlas
} // namespace bioexplorer
