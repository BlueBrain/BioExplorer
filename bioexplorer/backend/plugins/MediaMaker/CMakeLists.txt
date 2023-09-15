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

# ==============================================================================
# Project
# ==============================================================================
set(NAME MEDIAMAKER)
set(LIBRARY_NAME MediaMaker)

# ==============================================================================
# Packages and settings
# ==============================================================================
list(APPEND CMAKE_MODULE_PATH ${BIOEXPLORER_SOURCE_DIRS}/CMake)
set(${NAME}_MODULES_DIR ${PROJECT_SOURCE_DIR})

find_package(TIFF REQUIRED)
find_package(exiv2 REQUIRED)

include_directories(
    ${BIOEXPLORER_SOURCE_DIRS}
    ${CMAKE_BINARY_DIR}/generated/${LIBRARY_NAME}
    ${PROJECT_SOURCE_DIR}
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
if(PLATFORM_OSPRAY_ENABLED)
    message(STATUS "[MediaMaker] Building OSPRay module")
    include(ispc)

    list(APPEND ${NAME}_SOURCES
        module/ispc/renderer/AmbientOcclusionRenderer.cpp
        module/ispc/renderer/DepthRenderer.cpp
        module/ispc/renderer/AlbedoRenderer.cpp
        module/ispc/renderer/ShadowRenderer.cpp
        module/ispc/renderer/RadianceRenderer.cpp
    )

    set(${NAME}_ISPC_SOURCES
        module/ispc/renderer/AmbientOcclusionRenderer.ispc
        module/ispc/renderer/AlbedoRenderer.ispc
        module/ispc/renderer/DepthRenderer.ispc
        module/ispc/renderer/ShadowRenderer.ispc
        module/ispc/renderer/RadianceRenderer.ispc
    )

    # ==============================================================================
    # Compile ispc code
    # ==============================================================================
    list(APPEND CMAKE_MODULE_PATH ${OSPRAY_CMAKE_ROOT})
    list(APPEND ALL_ISPC_INCLUDES ${BIOEXPLORER_SOURCE_DIRS})
    list(APPEND ALL_ISPC_INCLUDES ${${NAME}_MODULES_DIR})
    list(APPEND ALL_ISPC_INCLUDES ${OSPRAY_INCLUDE_DIRS})
    list(APPEND ALL_ISPC_INCLUDES ${GLM_INCLUDE_DIRS})
    include_directories_ispc(${ALL_ISPC_INCLUDES})
    ospray_ispc_compile(${${NAME}_ISPC_SOURCES})
    list(APPEND ${NAME}_SOURCES ${ISPC_OBJECTS})

    list(APPEND ${NAME}_PUBLIC_MODULE_LIBRARIES ${OSPRAY_LIBRARIES})
    list(APPEND ${NAME}_PRIVATE_MODULE_LIBRARIES OSPRayEngine)
    include_directories(${OSPRAY_INCLUDE_DIRS})
endif()

# ==============================================================================
# OptiX 6 module
# ==============================================================================
if(PLATFORM_OPTIX6_ENABLED)
    message(STATUS "[MediaMaker] Building OptiX 6 module")
    list(APPEND CMAKE_MODULE_PATH ${BIOEXPLORER_SOURCE_DIRS}/CMake)

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
        module/cuda/renderer/Radiance.cu
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
include_directories(
    SYSTEM
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${BIOEXPLORER_SOURCE_DIRS}
    ${CMAKE_BINARY_DIR}/extras/${LIBRARY_NAME}
)

set(${NAME}_LINK_LIBRARIES
    PUBLIC ${${NAME}_PUBLIC_MODULE_LIBRARIES} ${TIFF_LIBRARIES} exiv2
    PRIVATE CoreParameters CoreCommon CoreEngine CoreIO ${${NAME}_PRIVATE_MODULE_LIBRARIES}
)

add_library(
    ${LIBRARY_NAME}
    SHARED ${${NAME}_SOURCES})

link_directories(${CMAKE_BINARY_DIR})

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
