# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2020-2022 Blue BrainProject / EPFL
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
# Packages and settings
# ==============================================================================
set(BIOEXPLORER_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../..)
list(APPEND CMAKE_MODULE_PATH ${BIOEXPLORER_SOURCE_DIR}/CMake)
set(${NAME}_MODULES_DIR ${PROJECT_SOURCE_DIR})

find_package(TIFF REQUIRED)
find_package(exiv2 REQUIRED)

include(ispc)

include_directories(
    ${BIOEXPLORER_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/generated/bioexplorer
)

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES
    module/ispc/render/DefaultMaterial.cpp
    module/ispc/render/AmbientOcclusionRenderer.cpp
    module/ispc/render/DepthRenderer.cpp
    module/ispc/render/AlbedoRenderer.cpp
    module/ispc/render/ShadowRenderer.cpp
    plugin/api/Params.cpp
    plugin/MediaMakerPlugin.cpp
)

set(${NAME}_PUBLIC_HEADERS
    plugin/api/Params.h
    plugin/MediaMakerPlugin.h
)

set(${NAME}_ISPC_SOURCES
    module/ispc/render/DefaultMaterial.ispc
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

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${BRAYNS_INCLUDE_DIRS}
    ${OSPRAY_INCLUDE_DIRS}
    ${TIFF_INCLUDE_DIRS}
    ${${NAME}_MODULES_DIR}
)

set(${NAME}_LINK_LIBRARIES
    PUBLIC ${OSPRAY_LIBRARIES} ${TIFF_LIBRARIES} exiv2
    PRIVATE braynsParameters braynsPluginAPI braynsCommon braynsEngine braynsIO braynsOSPRayEngine
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
