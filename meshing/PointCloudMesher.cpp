/* Copyright (c) 2020, EPFL/Blue Brain Project
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

#include "PointCloudMesher.h"
#include "MetaballsGenerator.h"

#include <common/log.h>

#include <brayns/engineapi/Model.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Random.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/convex_hull_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Polyhedron_3<K> Polyhedron_3;
typedef K::Point_3 Point_3;
typedef K::Segment_3 Segment_3;
typedef K::Triangle_3 Triangle_3;

namespace bioexplorer
{
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
            PLUGIN_INFO << "The convex hull contains "
                        << poly->size_of_vertices() << " vertices" << std::endl;

            for (auto eit = poly->edges_begin(); eit != poly->edges_end();
                 ++eit)
            {
                Point_3 a = eit->vertex()->point();
                Point_3 b = eit->opposite()->vertex()->point();
                model.addCylinder(point.first, {{a.x(), a.y(), a.z()},
                                                {b.x(), b.y(), b.z()},
                                                point.second[0].w});
                addModel = true;
            }
        }
        else
            PLUGIN_ERROR << "something else" << std::endl;
    }
    return addModel;
}

bool PointCloudMesher::toMetaballs(brayns::Model& model,
                                   const PointCloud& pointCloud,
                                   const size_t gridSize, const float threshold)
{
    auto& triangles = model.getTriangleMeshes();
    for (const auto& point : pointCloud)
    {
        if (point.second.empty())
            continue;

        PLUGIN_INFO << "Material " << point.first
                    << ", number of balls: " << point.second.size()
                    << std::endl;

        model.createMaterial(point.first, std::to_string(point.first));

        MetaballsGenerator metaballsGenerator;
        metaballsGenerator.generateMesh(point.first, point.second, gridSize,
                                        threshold, triangles);
    }
    return !triangles.empty();
}
} // namespace bioexplorer
