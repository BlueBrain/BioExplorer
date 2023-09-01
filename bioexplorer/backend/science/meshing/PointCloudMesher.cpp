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

#include "PointCloudMesher.h"

#include <science/common/Logs.h>
#include <science/common/ThreadSafeContainer.h>

#include <platform/core/engineapi/Material.h>
#include <platform/core/engineapi/Model.h>

#ifdef USE_CGAL
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
#endif

using namespace core;

namespace bioexplorer
{
using namespace common;

namespace meshing
{
PointCloudMesher::PointCloudMesher() {}

bool PointCloudMesher::toConvexHull(ThreadSafeContainer& container, const PointCloud& pointCloud)
{
#ifdef USE_CGAL
    bool addModel{false};
    for (const auto& point : pointCloud)
    {
        const auto materialId = point.first;
        std::vector<Point_3> points;
        for (const auto& c : point.second)
            points.push_back({c.x, c.y, c.z});

        CGAL::Object obj;
        // compute convex hull of non-collinear points
        CGAL::convex_hull_3(points.begin(), points.end(), obj);
        if (const Polyhedron_3* poly = CGAL::object_cast<Polyhedron_3>(&obj))
        {
            for (auto eit = poly->edges_begin(); eit != poly->edges_end(); ++eit)
            {
                Point_3 a = eit->vertex()->point();
                Point_3 b = eit->opposite()->vertex()->point();
                const float radius = static_cast<float>(point.second[0].w);
                container.addCone(Vector3f(a.x(), a.y(), a.z()), radius, Vector3f(b.x(), b.y(), b.z()), radius,
                                  materialId, false);
                container.addSphere(Vector3f(a.x(), a.y(), a.z()), radius, materialId, false);
                addModel = true;
            }
        }
        else
            PLUGIN_ERROR("something else");
    }
    return addModel;
#else
    PLUGIN_THROW("The BioExplorer was not compiled with the CGAL library")
#endif
}
} // namespace meshing
} // namespace bioexplorer
