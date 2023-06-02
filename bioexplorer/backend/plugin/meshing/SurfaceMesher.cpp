/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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

#include "SurfaceMesher.h"

#include <Defines.h>

#include <plugin/common/GeneralSettings.h>
#include <plugin/common/Logs.h>

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

namespace bioexplorer
{
namespace meshing
{
using namespace common;

SurfaceMesher::SurfaceMesher(const uint32_t uuid)
    : _uuid(uuid)
{
}

ModelDescriptorPtr SurfaceMesher::generateSurface(core::Scene& scene, const std::string& pdbId,
                                                  const Vector4ds& atoms, const double shrinkfactor)
{
#ifdef USE_CGAL
    ModelDescriptorPtr modelDescriptor{nullptr};
    MeshLoader meshLoader(scene);
    const std::string filename = GeneralSettings::getInstance()->getMeshFolder() + pdbId + ".off";
    try
    {
        PLUGIN_INFO(3, "Trying to load surface from cache " << filename);
        modelDescriptor = meshLoader.importFromFile(filename, LoaderProgress(), {});
        PLUGIN_INFO(3, "Surface loaded from cache " << filename);
        return modelDescriptor;
    }
    catch (const std::runtime_error& e)
    {
        PLUGIN_INFO(3, "Failed to load surface from cache (" << e.what() << "), constructing it...");
    }

    std::list<Weighted_point> l;
    for (const auto& atom : atoms)
        l.push_front(Weighted_point(Point_3(atom.x, atom.y, atom.z), atom.w));

    PLUGIN_INFO(3, "Constructing skin surface from " << l.size() << " atoms");

    Polyhedron polyhedron;
    Skin_surface_3 skinSurface(l.begin(), l.end(), shrinkfactor);

    PLUGIN_INFO(3, "Meshing skin surface...");
    CGAL::mesh_skin_surface_3(skinSurface, polyhedron);
    CGAL::Polygon_mesh_processing::triangulate_faces(polyhedron);

    PLUGIN_INFO(3, "Adding mesh to model");
    std::ofstream out(filename);
    out << polyhedron;
    return meshLoader.importFromFile(filename, LoaderProgress(), {});
#else
    PLUGIN_THROW("The BioExplorer was not compiled with the CGAL library")
#endif
}

ModelDescriptorPtr SurfaceMesher::generateUnionOfBalls(core::Scene& scene, const std::string& pdbId,
                                                       const Vector4ds& atoms)
{
#ifdef USE_CGAL
    std::list<Weighted_point> l;
    for (const auto& atom : atoms)
        l.push_front(Weighted_point(Point_3(atom.x, atom.y, atom.z), atom.w));

    ModelDescriptorPtr modelDescriptor{nullptr};
    MeshLoader meshLoader(scene);
    const std::string filename = GeneralSettings::getInstance()->getMeshFolder() + pdbId + ".off";
    try
    {
        PLUGIN_INFO(3, "Trying to load union of balls from cache " << filename);
        modelDescriptor = meshLoader.importFromFile(filename, LoaderProgress(), {});
        PLUGIN_INFO(3, "Surface loaded from cache " << filename);
        return modelDescriptor;
    }
    catch (const std::runtime_error& e)
    {
        PLUGIN_INFO(3, "Failed to load union of balls from cache (" << e.what() << "), constructing it...");
    }

    PLUGIN_INFO(3, "Constructing union of balls from " << l.size() << " atoms");

    Polyhedron polyhedron;
    Union_of_balls_3 union_of_balls(l.begin(), l.end());
    CGAL::mesh_union_of_balls_3(union_of_balls, polyhedron);

    PLUGIN_INFO(3, "Adding mesh to model");
    std::ofstream out(filename);
    out << polyhedron;
    return meshLoader.importFromFile(filename, LoaderProgress(), {});
#else
    PLUGIN_THROW("The BioExplorer was not compiled with the CGAL library")
#endif
}
} // namespace meshing
} // namespace bioexplorer
