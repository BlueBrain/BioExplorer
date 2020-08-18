# Copyright (c) 2020, Cyrille Favreau
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of https://github.com/favreau/BioExplorer

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# ==============================================================================
# Project
# ==============================================================================
set(NAME BIOEXPLORER)
set(LIBRARY_NAME BioExplorer)
project(${NAME} VERSION 0.6.0)
set(${NAME}_VERSION_ABI 1)

# ==============================================================================
# Packages and settings
# ==============================================================================
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake)
set(${NAME}_MODULES_DIR ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 14) # C++ 14

find_package(Brayns REQUIRED)
find_package(ospray REQUIRED SYSTEM)
find_package(Boost REQUIRED COMPONENTS program_options filesystem)
find_package(glm REQUIRED SYSTEM)
find_package(CGAL REQUIRED)
include(ispc)

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES
    module/ispc/camera/BioExplorerPerspectiveCamera.cpp
    module/ispc/render/BioExplorerMaterial.cpp
    module/ispc/render/BioExplorerRenderer.cpp
    module/ispc/render/BioExplorerFieldsRenderer.cpp
    plugin/common/Utils.cpp
    plugin/fields/FieldsHandler.cpp
    plugin/fields/Octree.cpp
    plugin/fields/OctreeNode.cpp
    plugin/io/BioExplorerLoader.cpp
    plugin/bioexplorer/Molecule.cpp
    plugin/bioexplorer/Assembly.cpp
    plugin/bioexplorer/Node.cpp
    plugin/bioexplorer/Membrane.cpp
    plugin/bioexplorer/Glycans.cpp
    plugin/bioexplorer/Mesh.cpp
    plugin/bioexplorer/Protein.cpp
    plugin/bioexplorer/RNASequence.cpp
    plugin/meshing/MetaballsGenerator.cpp
    plugin/meshing/PointCloudMesher.cpp
    plugin/api/Params.cpp
    plugin/BioExplorer.cpp
)

set(${NAME}_PUBLIC_HEADERS
    plugin/io/BioExplorerLoader.h
    plugin/bioexplorer/Molecule.h
    plugin/bioexplorer/Assembly.h
    plugin/bioexplorer/Node.h
    plugin/bioexplorer/Glycans.h
    plugin/bioexplorer/Mesh.h
    plugin/bioexplorer/Membrane.h
    plugin/bioexplorer/Protein.h
    plugin/bioexplorer/RNASequence.h
    plugin/common/CommonTypes.h
    plugin/meshing/MetaballsGenerator.h
    plugin/meshing/PointCloudMesher.h
    plugin/api/Params.h
    plugin/BioExplorer.h
)

set(${NAME}_ISPC_SOURCES
    module/ispc/camera/BioExplorerPerspectiveCamera.ispc
    module/ispc/render/utils/BioExplorerSkyBox.ispc
    module/ispc/render/utils/BioExplorerRandomGenerator.ispc
    module/ispc/render/BioExplorerFieldsRenderer.ispc
    module/ispc/render/BioExplorerRenderer.ispc
    module/ispc/render/BioExplorerMaterial.ispc
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
    ${OSPRAY_INCLUDE_DIRS}
    ${BRAYNS_INCLUDE_DIRS}
    ${GLM_INCLUDE_DIRS}
    ${${NAME}_MODULES_DIR}
)

set(${NAME}_LINK_LIBRARIES
    PUBLIC ${OSPRAY_LIBRARIES} ${CGAL_LIBRARIES}
           ${Boost_FILESYSTEM_LIBRARY} ${Boost_PROGRAM_OPTIONS_LIBRARY}
    PRIVATE braynsParameters braynsPluginAPI braynsCommon braynsEngine braynsIO
            braynsOSPRayEngine
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
