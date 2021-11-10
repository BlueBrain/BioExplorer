# Copyright (c) 2020-2021, Cyrille Favreau
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of https://github.com/BlueBrain/BioExplorer

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# ==============================================================================
# Project
# ==============================================================================
set(LIBRARY_NAME BioExplorer)
set(${NAME}_VERSION_ABI 1)

# ==============================================================================
# Packages and settings
# ==============================================================================
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake)
set(${NAME}_MODULES_DIR ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 14) # C++ 14

# Mandatory packages
find_package(Brayns REQUIRED)
find_package(ospray REQUIRED SYSTEM)
find_package(Boost REQUIRED COMPONENTS program_options filesystem)
find_package(glm REQUIRED SYSTEM)
find_package(FreeImage REQUIRED)
find_package(OpenMP)

# Optional packages (managed by Defines.h.in)
set(CGAL_FOUND 0)
option(${NAME}_USE_CGAL "Use CGAL meshing features" ON)
if(${${NAME}_USE_CGAL})
    find_package(CGAL REQUIRED)
endif()

set(PQXX_FOUND 0)
option(${NAME}_USE_PQXX "Use PostgreSQL database" ON)
if(${${NAME}_USE_PQXX})
    unset(PQXX_INCLUDE_DIRECTORIES CACHE)
    unset(PQXX_LIBRARIES CACHE)
    find_package(PQXX REQUIRED)
endif()

include(ispc)

configure_file( plugin/common/Defines.h.in ${CMAKE_BINARY_DIR}/generated/bioexplorer/Defines.h )
include_directories( ${PROJECT_SOURCE_DIR}/bioexplorer/core ${CMAKE_BINARY_DIR}/generated/bioexplorer )


# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES
    module/ispc/camera/PerspectiveStereoCamera.cpp
    module/ispc/render/AdvancedMaterial.cpp
    module/ispc/render/AdvancedRenderer.cpp
    module/ispc/render/FieldsRenderer.cpp
    module/ispc/render/DensityRenderer.cpp
    module/ispc/render/PathTracingRenderer.cpp
    plugin/common/GeneralSettings.cpp
    plugin/common/Utils.cpp
    plugin/common/UniqueId.cpp
    plugin/common/Shapes.cpp
    plugin/fields/FieldsHandler.cpp
    plugin/fields/Octree.cpp
    plugin/fields/OctreeNode.cpp
    plugin/io/CacheLoader.cpp
    plugin/io/OOCManager.cpp
    plugin/biology/Molecule.cpp
    plugin/biology/Assembly.cpp
    plugin/biology/Node.cpp
    plugin/biology/Membrane.cpp
    plugin/biology/Glycans.cpp
    plugin/biology/ParametricMembrane.cpp
    plugin/biology/MeshBasedMembrane.cpp
    plugin/biology/Protein.cpp
    plugin/biology/RNASequence.cpp
    plugin/api/Params.cpp
    plugin/BioExplorerPlugin.cpp
)

set(${NAME}_PUBLIC_HEADERS
    plugin/io/OOCManager.h
    plugin/io/CacheLoader.h
    plugin/biology/Molecule.h
    plugin/biology/Assembly.h
    plugin/biology/Node.h
    plugin/biology/Glycans.h
    plugin/biology/ParametricMembrane.h
    plugin/biology/MeshBasedMembrane.h
    plugin/biology/Membrane.h
    plugin/biology/Protein.h
    plugin/biology/RNASequence.h
    plugin/common/UniqueId.h
    plugin/common/GeneralSettings.h
    plugin/common/CommonTypes.h
    plugin/common/Shapes.h
    plugin/api/Params.h
    plugin/BioExplorerPlugin.h
)

if(${CGAL_FOUND})
    list(APPEND ${NAME}_SOURCES
        plugin/meshing/PointCloudMesher.cpp
        plugin/meshing/SurfaceMesher.cpp
    )

    list(APPEND ${NAME}_PUBLIC_HEADERS 
        plugin/meshing/PointCloudMesher.h
        plugin/meshing/SurfaceMesher.h
    )
endif()

if(${PQXX_FOUND})
    list(APPEND ${NAME}_SOURCES plugin/io/db/DBConnector.cpp)
    list(APPEND ${NAME}_PUBLIC_HEADERS plugin/io/db/DBConnector.h)
endif()

if(${CGAL_FOUND})
    add_compile_options(-frounding-math)
endif()

set(${NAME}_ISPC_SOURCES
    module/ispc/camera/PerspectiveStereoCamera.ispc
    module/ispc/render/utils/BioExplorerSkyBox.ispc
    module/ispc/render/utils/BioExplorerRandomGenerator.ispc
    module/ispc/render/FieldsRenderer.ispc
    module/ispc/render/DensityRenderer.ispc
    module/ispc/render/AdvancedRenderer.ispc
    module/ispc/render/AdvancedMaterial.ispc
    module/ispc/render/PathTracingRenderer.ispc
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
            braynsOSPRayEngine ${FreeImage_LIBRARIES}
)

if(${CGAL_FOUND})
    list(APPEND ${NAME}_LINK_LIBRARIES ${CGAL_LIBRARIES})
endif()

if(${PQXX_FOUND})
    list(APPEND ${NAME}_LINK_LIBRARIES ${PQXX_LIBRARIES})
endif()


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
