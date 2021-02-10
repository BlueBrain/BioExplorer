# Copyright (c) 2020-2021, Cyrille Favreau
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of https://github.com/BlueBrain/BioExplorer

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# ==============================================================================
# Project
# ==============================================================================
set(NAME BIOEXPLORER)
set(LIBRARY_NAME BioExplorer)
project(${NAME} VERSION 0.8.0)
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
find_package(FreeImage REQUIRED)
find_package(CGAL)

include(ispc)

configure_file( plugin/common/Defines.h.in ${CMAKE_BINARY_DIR}/generated/Defines.h )
include_directories( ${CMAKE_BINARY_DIR}/generated/ ) # Make sure it can be included...


# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES
    module/ispc/camera/PerspectiveStereoCamera.cpp
    module/ispc/render/AdvancedMaterial.cpp
    module/ispc/render/AdvancedRenderer.cpp
    module/ispc/render/FieldsRenderer.cpp
    plugin/common/Utils.cpp
    plugin/fields/FieldsHandler.cpp
    plugin/fields/Octree.cpp
    plugin/fields/OctreeNode.cpp
    plugin/io/CacheLoader.cpp
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
    plugin/meshing/SurfaceMesher.cpp
    plugin/api/Params.cpp
    plugin/BioExplorerPlugin.cpp
)

set(${NAME}_PUBLIC_HEADERS
    plugin/io/CacheLoader.h
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
    plugin/meshing/SurfaceMesher.h
    plugin/api/Params.h
    plugin/BioExplorerPlugin.h
)

set(${NAME}_ISPC_SOURCES
    module/ispc/camera/PerspectiveStereoCamera.ispc
    module/ispc/render/utils/BioExplorerSkyBox.ispc
    module/ispc/render/utils/BioExplorerRandomGenerator.ispc
    module/ispc/render/FieldsRenderer.ispc
    module/ispc/render/AdvancedRenderer.ispc
    module/ispc/render/AdvancedMaterial.ispc
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
    PUBLIC ${OSPRAY_LIBRARIES}
    PRIVATE braynsParameters braynsPluginAPI braynsCommon braynsEngine braynsIO
            braynsOSPRayEngine ${CGAL_LIBRARIES} ${FreeImage_LIBRARIES}
)

add_library(
    ${LIBRARY_NAME}
    SHARED ${${NAME}_SOURCES})

target_link_libraries(
    ${LIBRARY_NAME}
    ${${NAME}_LINK_LIBRARIES})


# ==============================================================================
# Unit testing
# ==============================================================================
option(${NAME}_UNIT_TESTING_ENABLED "Enable unit testing" ON)
if(${NAME}_UNIT_TESTING_ENABLED)
    add_subdirectory(tests)
endif()

# ==============================================================================
# Install binaries
# ==============================================================================
INSTALL(TARGETS ${LIBRARY_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
