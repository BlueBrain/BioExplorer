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

#include "PointCloudMesher.h"
#include "../../common/Logs.h"

#include "MetaballsGenerator.h"
#include <platform/core/engineapi/Model.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Random.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/convex_hull_3.h>

using namespace core;

namespace sonataexplorer
{
namespace meshing
{
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Polyhedron_3<K> Polyhedron_3;
typedef K::Point_3 Point_3;
typedef K::Segment_3 Segment_3;
typedef K::Triangle_3 Triangle_3;

PointCloudMesher::PointCloudMesher() {}

bool PointCloudMesher::toConvexHull(Model& model, const PointCloud& pointCloud)
{
    bool addModel{false};
    for (const auto& point : pointCloud)
    {
        model.createMaterial(point.first, std::to_string(point.first));

        std::vector<Point_3> points;
        for (const auto& c : point.second)
            points.push_back({c.x, c.y, c.z});

        CGAL::Object obj;
        // compute convex hull of non-collinear points
        CGAL::convex_hull_3(points.begin(), points.end(), obj);
        if (const Polyhedron_3* poly = CGAL::object_cast<Polyhedron_3>(&obj))
        {
            PLUGIN_INFO("The convex hull contains " << poly->size_of_vertices() << " vertices");

            for (auto eit = poly->edges_begin(); eit != poly->edges_end(); ++eit)
            {
                Point_3 a = eit->vertex()->point();
                Point_3 b = eit->opposite()->vertex()->point();
                const Cylinder cylinder(Vector3f(a.x(), a.y(), a.z()), Vector3f(b.x(), b.y(), b.z()), 1.f);
                model.addCylinder(point.first, cylinder);
                addModel = true;
            }
        }
        else
            PLUGIN_ERROR("something else");
    }
    return addModel;
}

bool PointCloudMesher::toMetaballs(Model& model, const PointCloud& pointCloud, const size_t gridSize,
                                   const float threshold)
{
    auto& triangles = model.getTriangleMeshes();
    for (const auto& point : pointCloud)
    {
        if (point.second.empty())
            continue;

        PLUGIN_INFO("Material " << point.first << ", number of balls: " << point.second.size());

        model.createMaterial(point.first, std::to_string(point.first));

        MetaballsGenerator metaballsGenerator;
        metaballsGenerator.generateMesh(point.second, gridSize, threshold, point.first, triangles);
    }
    return !triangles.empty();
}
} // namespace meshing
} // namespace sonataexplorer
