/* Copyright (c) 2020-2022, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
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
    if (_details.loadCells)
        _loadCells();
    if (_details.loadMeshes)
        _loadMeshes();
    PLUGIN_TIMER(chrono.elapsed(), "Atlas loaded");
}

void Atlas::_loadCells()
{
    if (_modelDescriptor)
        _scene.removeModel(_modelDescriptor->getModelID());

    auto model = _scene.createModel();

    auto& connector = DBConnector::getInstance();
    const auto regions = connector.getAtlasRegions(_details.regionSqlFilter);

    ThreadSafeContainers containers;
    uint64_t counter = 0;
    uint64_t nbCells = 0;
    uint64_t index;
#pragma omp parallel for
    for (index = 0; index < regions.size(); ++index)
    {
        ThreadSafeContainer container(*model, false);

        const auto region = regions[index];
        const auto cells =
            connector.getAtlasCells(region, _details.cellSqlFilter);
        for (const auto& cell : cells)
            container.addSphere(cell.second.position, _details.cellRadius,
                                cell.second.region);
#pragma omp critical
        nbCells += cells.size();

#pragma omp critical
        containers.push_back(container);

#pragma omp critical
        ++counter;

#pragma omp critical
        PLUGIN_PROGRESS("Loading " << regions.size() << " regions", counter,
                        regions.size());
    }

    for (size_t i = 0; i < containers.size(); ++i)
    {
        const float progress = 1.f + i;
        PLUGIN_PROGRESS("- Compiling 3D geometry...", progress,
                        containers.size());
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

void Atlas::_loadMeshes() {}
} // namespace atlas
} // namespace bioexplorer
