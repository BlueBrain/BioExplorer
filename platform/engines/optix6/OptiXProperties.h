/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#pragma once

#include <string>

namespace core
{
/*
Cuda functions
*/
static const char* OPTIX_CUDA_FUNCTION_CLOSEST_HIT_RADIANCE = "closest_hit_radiance";
static const char* OPTIX_CUDA_FUNCTION_CLOSEST_HIT_RADIANCE_TEXTURED = "closest_hit_radiance_textured";
static const char* OPTIX_CUDA_FUNCTION_ANY_HIT_SHADOW = "any_hit_shadow";
static const char* OPTIX_CUDA_FUNCTION_EXCEPTION = "exception";
static const char* OPTIX_CUDA_FUNCTION_CAMERA_ENVMAP_MISS = "envmap_miss";
static const char* OPTIX_CUDA_FUNCTION_BOUNDS = "bounds";
static const char* OPTIX_CUDA_FUNCTION_INTERSECTION = "intersect";
static const char* OPTIX_CUDA_FUNCTION_ROBUST_INTERSECTION = "robust_intersect";

/*
Acceleration strucure properties
*/
static const char* OPTIX_ACCELERATION_TYPE_SBVH = "Sbvh";
static const char* OPTIX_ACCELERATION_TYPE_TRBVH = "Trbvh";
static const char* DEFAULT_ACCELERATION_STRUCTURE = OPTIX_ACCELERATION_TYPE_TRBVH;

static const char* OPTIX_ACCELERATION_VERTEX_BUFFER_NAME = "vertex_buffer_name";
static const char* OPTIX_ACCELERATION_VERTEX_BUFFER_STRIDE = "vertex_buffer_stride";
static const char* OPTIX_ACCELERATION_INDEX_BUFFER_NAME = "index_buffer_name";
static const char* OPTIX_ACCELERATION_INDEX_BUFFER_STRIDE = "index_buffer_stride";

/*
Geometry properties
*/
static const char* OPTIX_GEOMETRY_PROPERTY_SPHERES = "spheres";
static const char* OPTIX_GEOMETRY_PROPERTY_CYLINDERS = "cylinders";
static const char* OPTIX_GEOMETRY_PROPERTY_CONES = "cones";
static const char* OPTIX_GEOMETRY_PROPERTY_VOLUMES = "volumes";

static const char* OPTIX_GEOMETRY_PROPERTY_TRIANGLE_MESH_VERTEX = "vertices_buffer";
static const char* OPTIX_GEOMETRY_PROPERTY_TRIANGLE_MESH_INDEX = "indices_buffer";
static const char* OPTIX_GEOMETRY_PROPERTY_TRIANGLE_MESH_NORMAL = "normal_buffer";
static const char* OPTIX_GEOMETRY_PROPERTY_TRIANGLE_MESH_TEXTURE_COORDINATES = "texcoord_buffer";

static const char* OPTIX_GEOMETRY_PROPERTY_STREAMLINE_VERTEX = "vertices_buffer";
static const char* OPTIX_GEOMETRY_PROPERTY_STREAMLINE_MESH_INDEX = "indices_buffer";

} // namespace core
