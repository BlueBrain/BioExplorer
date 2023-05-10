# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2020-2023 Blue BrainProject / EPFL
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <https://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# ==============================================================================
# Project
# ==============================================================================
set(NAME MEDIAMAKER)
set(LIBRARY_NAME MediaMaker)
project(${NAME} VERSION ${BIOEXPLORER_PACKAGE_VERSION})
set(${NAME}_VERSION_ABI 1)


# ==============================================================================
# Options
# ==============================================================================
option(BIOEXPLORER_MEDIA_MAKER_OSPRAY_ENABLED "Enable OSPRay module" ON)
find_package(OptiX6)
if(OPTIX_FOUND)
    option(BIOEXPLORER_MEDIA_MAKER_OPTIX6_ENABLED "Enable OptiX 6 module" ON)
endif()

# ==============================================================================
# Packages and settings
# ==============================================================================
set(BIOEXPLORER_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../..)
list(APPEND CMAKE_MODULE_PATH ${BIOEXPLORER_SOURCE_DIR}/CMake)
set(${NAME}_MODULES_DIR ${PROJECT_SOURCE_DIR})

find_package(TIFF REQUIRED)
find_package(exiv2 REQUIRED)

include_directories(
    ${BIOEXPLORER_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/generated/mediamaker
    ${PROJECT_SOURCE_DIR}
    ${BRAYNS_INCLUDE_DIRS}
    ${TIFF_INCLUDE_DIRS}
    ${${NAME}_MODULES_DIR}
)

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES
    plugin/api/Params.cpp
    plugin/MediaMakerPlugin.cpp
)

set(${NAME}_PUBLIC_HEADERS
    plugin/api/Params.h
    plugin/MediaMakerPlugin.h
)

set(${NAME}_PUBLIC_MODULE_LIBRARIES)
set(${NAME}_PRIVATE_MODULE_LIBRARIES)
# ==============================================================================
# OSPRay module
# ==============================================================================
set(OSPRAY_FOUND 0)
if(BIOEXPLORER_MEDIA_MAKER_OSPRAY_ENABLED)
    message(STATUS "[MediaMaker] Building OSPRay module")
    set(OSPRAY_FOUND 1)
    include(ispc)

    list(APPEND ${NAME}_SOURCES
        module/ispc/render/AmbientOcclusionRenderer.cpp
        module/ispc/render/DepthRenderer.cpp
        module/ispc/render/AlbedoRenderer.cpp
        module/ispc/render/ShadowRenderer.cpp
    )

    set(${NAME}_ISPC_SOURCES
        module/ispc/render/AmbientOcclusionRenderer.ispc
        module/ispc/render/AlbedoRenderer.ispc
        module/ispc/render/DepthRenderer.ispc
        module/ispc/render/ShadowRenderer.ispc
    )

    # ==============================================================================
    # Compile ispc code
    # ==============================================================================
    list(APPEND CMAKE_MODULE_PATH ${OSPRAY_CMAKE_ROOT})
    list(APPEND ALL_ISPC_INCLUDES ${${NAME}_MODULES_DIR})
    list(APPEND ALL_ISPC_INCLUDES ${OSPRAY_INCLUDE_DIRS})
    list(APPEND ALL_ISPC_INCLUDES ${GLM_INCLUDE_DIRS})
    include_directories_ispc(${ALL_ISPC_INCLUDES})
    ospray_ispc_compile(${${NAME}_ISPC_SOURCES})
    list(APPEND ${NAME}_SOURCES ${ISPC_OBJECTS})

    list(APPEND ${NAME}_PUBLIC_MODULE_LIBRARIES ${OSPRAY_LIBRARIES})
    list(APPEND ${NAME}_PRIVATE_MODULE_LIBRARIES braynsOSPRayEngine)
    include_directories(${OSPRAY_INCLUDE_DIRS})
endif()

# ==============================================================================
# OptiX 6 module
# ==============================================================================
set(OPTIX6_FOUND 0)
if(BIOEXPLORER_MEDIA_MAKER_OPTIX6_ENABLED)
    message(STATUS "[MediaMaker] Building OptiX 6 module")
    set(OPTIX6_FOUND 1)
    list(APPEND CMAKE_MODULE_PATH ${BIOEXPLORER_SOURCE_DIR}/CMake)

    find_package(CUDA REQUIRED)
    if(CUDA_FOUND)
      # This doesn't get called from common_find_package_post unless we export some
      # variables to the parent scope
      find_cuda_compatible_host_compiler()
    endif()
    message(STATUS "Using CUDA version ${CUDA_VERSION}")
    
    set(CUDA_NVCC_FLAGS "--use_fast_math")
    
    set(${NAME}_CU
        module/cuda/renderer/Albedo.cu
        module/cuda/renderer/ShadingNormal.cu
        module/cuda/renderer/GeometryNormal.cu
        module/cuda/renderer/AmbientOcclusion.cu
        module/cuda/renderer/Shadow.cu
        module/cuda/renderer/Depth.cu
    )
    
    CUDA_WRAP_SRCS(${LIBRARY_NAME} PTX ptx_generated_files ${${NAME}_CU})
    
    include(StringifyPtx)
    stringify_ptx(${ptx_generated_files})

    list(APPEND ${NAME}_SOURCES
        ${${NAME}_CU}
        ${ptx_generated_files}
        ${PTX_SOURCES}
    )

    list(APPEND ${NAME}_HEADERS ${PTX_HEADERS})
    include_directories(${OptiX6_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
endif()

# ==============================================================================
# Compile c++ code
# ==============================================================================
configure_file(
    plugin/common/Defines.h.in
    ${CMAKE_BINARY_DIR}/generated/mediamaker/Defines.h
)

set(${NAME}_LINK_LIBRARIES
    PUBLIC ${${NAME}_PUBLIC_MODULE_LIBRARIES} ${TIFF_LIBRARIES} exiv2
    PRIVATE braynsParameters braynsPluginAPI braynsCommon braynsEngine braynsIO ${${NAME}_PRIVATE_MODULE_LIBRARIES}
)

add_library(
    ${LIBRARY_NAME}
    SHARED ${${NAME}_SOURCES})

target_link_libraries(
    ${LIBRARY_NAME}
    ${${NAME}_LINK_LIBRARIES})

# ==============================================================================
# Install binaries
# ==============================================================================
INSTALL(TARGETS ${LIBRARY_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
