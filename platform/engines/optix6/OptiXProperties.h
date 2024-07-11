/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

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

#pragma once

#include <string>

namespace core
{
namespace engine
{
namespace optix
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
static const char* OPTIX_GEOMETRY_PROPERTY_SDF_GEOMETRIES = "sdf_geometries_buffer";
static const char* OPTIX_GEOMETRY_PROPERTY_SDF_GEOMETRIES_INDICES = "sdf_geometries_indices_buffer";
static const char* OPTIX_GEOMETRY_PROPERTY_SDF_GEOMETRIES_NEIGHBOURS = "sdf_geometries_neighbours_buffer";
static const char* OPTIX_GEOMETRY_PROPERTY_VOLUMES = "volumes";
static const char* OPTIX_GEOMETRY_PROPERTY_FIELDS = "fields";

static const char* OPTIX_GEOMETRY_PROPERTY_TRIANGLE_MESH_VERTEX = "vertices_buffer";
static const char* OPTIX_GEOMETRY_PROPERTY_TRIANGLE_MESH_INDEX = "indices_buffer";
static const char* OPTIX_GEOMETRY_PROPERTY_TRIANGLE_MESH_NORMAL = "normal_buffer";
static const char* OPTIX_GEOMETRY_PROPERTY_TRIANGLE_MESH_TEXTURE_COORDINATES = "texcoord_buffer";

static const char* OPTIX_GEOMETRY_PROPERTY_STREAMLINE_VERTEX = "vertices_buffer";
static const char* OPTIX_GEOMETRY_PROPERTY_STREAMLINE_MESH_INDEX = "indices_buffer";

} // namespace optix
} // namespace engine
} // namespace core