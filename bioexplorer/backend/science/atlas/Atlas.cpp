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

#include "Atlas.h"

#include <science/common/Logs.h>
#include <science/common/ThreadSafeContainer.h>
#include <science/common/Utils.h>

#include <science/io/db/DBConnector.h>

#include <platform/core/common/Timer.h>
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

namespace atlas
{
Atlas::Atlas(Scene& scene, const AtlasDetails& details, const Vector3d& position, const Quaterniond& rotation,
             const LoaderProgress& callback)
    : SDFGeometries(NO_GRID_ALIGNMENT, position, rotation, doublesToVector3d(details.scale))
    , _details(details)
    , _scene(scene)
{
    Timer chrono;
    _buildModel(callback);
    PLUGIN_TIMER(chrono.elapsed(), "Atlas loaded");
}

void Atlas::_buildModel(const LoaderProgress& callback)
{
    if (_modelDescriptor)
        _scene.removeModel(_modelDescriptor->getModelID());

    auto model = _scene.createModel();

    auto& connector = DBConnector::getInstance();
    const auto regions = connector.getAtlasRegions(_details.populationName, _details.regionSqlFilter);
    const bool useSdf = false;

    ThreadSafeContainers containers;
    uint64_t counter = 0;
    uint64_t nbCells = 0;

    const auto nbDBConnections = DBConnector::getInstance().getNbConnections();
    uint64_t index;
    volatile bool flag = false;
    std::string flagMessage;
#pragma omp parallel for shared(flag, flagMessage) num_threads(nbDBConnections)
    for (index = 0; index < regions.size(); ++index)
    {
        try
        {
            if (flag)
                continue;

            if (omp_get_thread_num() == 0)
            {
                PLUGIN_PROGRESS("Loading regions...", index, regions.size());
                try
                {
                    callback.updateProgress("Loading regions...",
                                            0.5f * ((float)index / (float)(regions.size() / nbDBConnections)));
                }
                catch (...)
                {
#pragma omp critical
                    {
                        flag = true;
                    }
                }
            }
            ThreadSafeContainer container(*model, _alignToGrid, _position, _rotation);

            const auto region = regions[index];
            if (_details.loadCells)
            {
                const auto cells = connector.getAtlasCells(_details.populationName, region, _details.cellSqlFilter);
                for (const auto& cell : cells)
                    container.addSphere(cell.second.position, _details.cellRadius, cell.second.region, useSdf);
#pragma omp critical
                {
                    nbCells += cells.size();
                }
            }

            if (_details.loadMeshes)
            {
                const Vector3d position = doublesToVector3d(_details.meshPosition);
                const Quaterniond rotation = doublesToQuaterniond(_details.meshRotation);
                const Vector3d scale = doublesToVector3d(_details.meshScale);
                auto mesh = connector.getAtlasMesh(_details.populationName, region);
                for (auto& vertex : mesh.vertices)
                {
                    const Vector3d p = Vector3d(vertex) + position;
                    const Vector3d r = rotation * p;
                    vertex = scale * r;
                }
                container.addMesh(region, mesh);
            }

#pragma omp critical
            {
                ++counter;

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

    for (uint64_t i = 0; i < containers.size(); ++i)
    {
        PLUGIN_PROGRESS("- Compiling 3D geometry...", i + 1, containers.size());
        callback.updateProgress("Compiling 3D geometry...", 0.5f + 0.5f * (float)(1 + i) / (float)containers.size());
        auto& container = containers[i];
        container.commitToModel();
    }
    model->applyDefaultColormap();

    const ModelMetadata metadata = {{"Number of regions", std::to_string(regions.size())},
                                    {"Number of cells", std::to_string(nbCells)},
                                    {"Cell SQL filter", _details.cellSqlFilter},
                                    {"Region SQL filter", _details.regionSqlFilter}};

    _modelDescriptor.reset(new core::ModelDescriptor(std::move(model), _details.assemblyName, metadata));
    if (!_modelDescriptor)
        PLUGIN_THROW("Atlas model could not be created");
}

} // namespace atlas
} // namespace bioexplorer
