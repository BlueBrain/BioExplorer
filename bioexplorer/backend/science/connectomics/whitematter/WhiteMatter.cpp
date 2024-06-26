/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2024 Blue BrainProject / EPFL
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
#pragma omp parallel for num_threads(nbDBConnections)
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
                    flag = true;
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
            containers.push_back(container);
        }
        catch (const std::runtime_error& e)
        {
#pragma omp critical
            flagMessage = e.what();
#pragma omp critical
            flag = true;
        }
        catch (...)
        {
#pragma omp critical
            flagMessage = "Loading was canceled";
#pragma omp critical
            flag = true;
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
