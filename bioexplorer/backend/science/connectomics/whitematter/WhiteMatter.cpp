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

#include "WhiteMatter.h"

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
WhiteMatter::WhiteMatter(Scene& scene, const WhiteMatterDetails& details, const Vector3d& position,
                         const Quaterniond& rotation, const LoaderProgress& callback)
    : SDFGeometries(NO_GRID_ALIGNMENT, position, rotation, doublesToVector3d(details.scale))
    , _details(details)
    , _scene(scene)
{
    Timer chrono;
    _buildModel(callback);
    PLUGIN_TIMER(chrono.elapsed(), "White matter loaded");
}

void WhiteMatter::_buildModel(const LoaderProgress& callback)
{
    if (_modelDescriptor)
        _scene.removeModel(_modelDescriptor->getModelID());

    auto model = _scene.createModel();
    ThreadSafeContainers containers;

    const auto ompThreads = omp_get_max_threads();

    const auto nbDBConnections = DBConnector::getInstance().getNbConnections();
    const auto streamlines =
        DBConnector::getInstance().getWhiteMatterStreamlines(_details.populationName, _details.sqlFilter);

    const auto nbStreamlines = streamlines.size();
    uint64_t index;
    volatile bool flag = false;
    std::string flagMessage;
#pragma omp parallel for shared(flag, flagMessage) num_threads(nbDBConnections)
    for (index = 0; index < nbStreamlines; ++index)
    {
        try
        {
            if (flag)
                continue;

            if (omp_get_thread_num() == 0)
            {
                PLUGIN_PROGRESS("Loading white matter...", index, nbStreamlines);
                try
                {
                    callback.updateProgress("Loading white matter...",
                                            0.5f * ((float)index / (float)(nbStreamlines / nbDBConnections)));
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
            StreamlinesData streamline;

            const float alpha = 1.f;
            uint64_t j = 0;
            Vector3f previousPoint;
            const auto& points = streamlines[index];
            for (const auto& point : points)
            {
                streamline.vertex.push_back({point.x, point.y, point.z, _details.radius});
                streamline.vertexColor.push_back(
                    (j == 0 ? Vector4f(0.f, 0.f, 0.f, alpha)
                            : Vector4f(0.5f + 0.5f * normalize(point - previousPoint), alpha)));
                previousPoint = point;
                ++j;
            }
            const size_t materialId = 0;
            container.addStreamline(materialId, streamline);

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
    for (uint64_t i = 0; i < containers.size(); ++i)
    {
        PLUGIN_PROGRESS("- Compiling 3D geometry...", i + 1, containers.size());
        callback.updateProgress("Compiling 3D geometry...", 0.5f + 0.5f * (float)(1 + i) / (float)containers.size());
        auto& container = containers[i];
        container.commitToModel();
    }

    const ModelMetadata metadata = {{"Number of streamlines", std::to_string(nbStreamlines)},
                                    {"SQL filter", _details.sqlFilter}};
    _modelDescriptor.reset(new core::ModelDescriptor(std::move(model), _details.assemblyName, metadata));
    if (!_modelDescriptor)
        PLUGIN_THROW(
            "WhiteMatter model could not be created for "
            "population " +
            _details.populationName);
}
} // namespace connectomics
} // namespace bioexplorer
