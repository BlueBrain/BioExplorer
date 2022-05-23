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

#include "WhiteMatter.h"

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
namespace connectomics
{
using namespace common;
using namespace io;
using namespace db;

WhiteMatter::WhiteMatter(Scene& scene, const WhiteMatterDetails& details)
    : Node(doublesToVector3d(details.scale))
    , _details(details)
    , _scene(scene)
{
    Timer chrono;
    _buildModel();
    PLUGIN_TIMER(chrono.elapsed(), "White matter loaded");
}

void WhiteMatter::_addStreamline(ThreadSafeContainer& container,
                                 const Vector3fs& points,
                                 const uint64_t materialId)
{
    StreamlinesData streamline;

    const float alpha = 1.f;
    uint64_t i = 0;
    Vector3f previousPoint;
    for (const auto& point : points)
    {
        streamline.vertex.push_back(
            {point.x, point.y, point.z, _details.radius});
        streamline.vertexColor.push_back(
            (i == 0 ? Vector4f(0.f, 0.f, 0.f, alpha)
                    : Vector4f(0.5f + 0.5f * normalize(point - previousPoint),
                               alpha)));
        previousPoint = point;
        ++i;
    }

    container.addStreamline(materialId, streamline);
}

void WhiteMatter::_buildModel()
{
    if (_modelDescriptor)
        _scene.removeModel(_modelDescriptor->getModelID());

    auto model = _scene.createModel();
    ThreadSafeContainer container(*model, false);

    const auto ompThreads = omp_get_max_threads();

    const auto streamlines =
        DBConnector::getInstance().getWhiteMatterStreamlines(
            _details.populationName, _details.sqlFilter);
    const auto nbStreamlines = streamlines.size();
    for (uint64_t i = 0; i < nbStreamlines; ++i)
    {
        _addStreamline(container, streamlines[i], i);
        PLUGIN_PROGRESS("Loading " << i << "/" << nbStreamlines
                                   << " streamlines",
                        i, nbStreamlines);
    }

    container.commitToModel();
    PLUGIN_INFO(1, "");

    const ModelMetadata metadata = {
        {"Number of streamlines", std::to_string(nbStreamlines)}};

    _modelDescriptor.reset(new brayns::ModelDescriptor(std::move(model),
                                                       _details.assemblyName,
                                                       metadata));
    if (_modelDescriptor)
        _scene.addModel(_modelDescriptor);
    else
        PLUGIN_THROW(
            "WhiteMatter model could not be created for "
            "population " +
            _details.populationName);
}
} // namespace connectomics
} // namespace bioexplorer
