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

#include "SurfaceMesher.h"

#include <Defines.h>

#include <science/common/GeneralSettings.h>
#include <science/common/Logs.h>

#include <platform/core/engineapi/Material.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Scene.h>

#include <fstream>

#ifdef USE_ASSIMP
#include <platform/core/io/MeshLoader.h>
#endif

#ifdef USE_CGAL
#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Skin_surface_3.h>
#include <CGAL/Union_of_balls_3.h>
#include <CGAL/mesh_skin_surface_3.h>
#include <CGAL/mesh_union_of_balls_3.h>
#include <CGAL/subdivide_union_of_balls_mesh_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Skin_surface_traits_3<K> Traits;
typedef K::Point_3 Point_3;
typedef K::Weighted_point_3 Weighted_point;
typedef CGAL::Polyhedron_3<K> Polyhedron;
typedef CGAL::Skin_surface_traits_3<K> Traits;
typedef CGAL::Skin_surface_3<Traits> Skin_surface_3;
typedef CGAL::Union_of_balls_3<Traits> Union_of_balls_3;
#endif
#endif

using namespace core;

namespace bioexplorer
{
namespace meshing
{
using namespace common;

SurfaceMesher::SurfaceMesher(const uint32_t uuid)
    : _uuid(uuid)
{
}

ModelDescriptorPtr SurfaceMesher::generateSurface(core::Scene& scene, const std::string& name, const Vector4ds& points,
                                                  const double shrinkfactor)
{
#ifdef USE_CGAL
    ModelDescriptorPtr modelDescriptor{nullptr};
    MeshLoader meshLoader(scene);
    const std::string filename = GeneralSettings::getInstance()->getMeshFolder() + name + ".off";
    try
    {
        PLUGIN_INFO(3, "Trying to load surface from cache " << filename);
        modelDescriptor = meshLoader.importFromStorage(filename, LoaderProgress(), {});
        PLUGIN_INFO(3, "Surface loaded from cache " << filename);
        return modelDescriptor;
    }
    catch (const std::runtime_error& e)
    {
        PLUGIN_INFO(3, "Failed to load surface from cache (" << e.what() << "), constructing it...");
    }

    std::list<Weighted_point> l;
    for (const auto& point : points)
        l.push_front(Weighted_point(Point_3(point.x, point.y, point.z), point.w));

    PLUGIN_INFO(3, "Constructing skin surface from " << l.size() << " point");

    Polyhedron polyhedron;
    Skin_surface_3 skinSurface(l.begin(), l.end(), shrinkfactor);

    PLUGIN_INFO(3, "Meshing skin surface...");
    CGAL::mesh_skin_surface_3(skinSurface, polyhedron);
    CGAL::Polygon_mesh_processing::triangulate_faces(polyhedron);

    PLUGIN_INFO(3, "Adding mesh to model");
    std::ofstream out(filename);
    out << polyhedron;
    return meshLoader.importFromStorage(filename, LoaderProgress(), {});
#else
    PLUGIN_THROW("The BioExplorer was not compiled with the CGAL library")
#endif
}

ModelDescriptorPtr SurfaceMesher::generateUnionOfBalls(core::Scene& scene, const std::string& name,
                                                       const Vector4ds& points)
{
#ifdef USE_CGAL
    std::list<Weighted_point> l;
    for (const auto& point : points)
        l.push_front(Weighted_point(Point_3(point.x, point.y, point.z), point.w));

    ModelDescriptorPtr modelDescriptor{nullptr};
    MeshLoader meshLoader(scene);
    const std::string filename = GeneralSettings::getInstance()->getMeshFolder() + name + ".off";
    try
    {
        PLUGIN_INFO(3, "Trying to load union of balls from cache " << filename);
        modelDescriptor = meshLoader.importFromStorage(filename, LoaderProgress(), {});
        PLUGIN_INFO(3, "Surface loaded from cache " << filename);
        return modelDescriptor;
    }
    catch (const std::runtime_error& e)
    {
        PLUGIN_INFO(3, "Failed to load union of balls from cache (" << e.what() << "), constructing it...");
    }

    PLUGIN_INFO(3, "Constructing union of balls from " << l.size() << " points");

    Polyhedron polyhedron;
    Union_of_balls_3 union_of_balls(l.begin(), l.end());
    CGAL::mesh_union_of_balls_3(union_of_balls, polyhedron);

    PLUGIN_INFO(3, "Adding mesh to model");
    std::ofstream out(filename);
    out << polyhedron;
    return meshLoader.importFromStorage(filename, LoaderProgress(), {});
#else
    PLUGIN_THROW("The BioExplorer was not compiled with the CGAL library")
#endif
}
} // namespace meshing
} // namespace bioexplorer
