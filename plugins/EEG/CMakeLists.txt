# Copyright (c) 2015-2020, EPFL/Blue Brain Project
# All rights reserved. Do not distribute without permission.
#
# This file is part of Brayns <https://github.com/BlueBrain/Brayns>

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# ==============================================================================
# Project
# ==============================================================================
set(NAME BRAYNSEEG)
set(LIBRARY_NAME braynsEEG)
project(${NAME} VERSION 0.1.0)
set(${NAME}_VERSION_ABI 1)

# ==============================================================================
# Packages and settings
# ==============================================================================
list(APPEND CMAKE_MODULE_PATH ${OSPRAY_CMAKE_ROOT})
set(${NAME}_MODULES_DIR ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 14)

find_package(ospray REQUIRED SYSTEM)
find_package(PQXX REQUIRED)

include(ispc)

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES
    module/ispc/render/FieldRenderer.cpp
    module/ispc/render/OctreeFieldRenderer.cpp
    plugin/common/Utils.cpp
    plugin/io/handlers/FieldHandler.cpp
    plugin/io/octree/OctreeNode.cpp
    plugin/io/octree/Octree.cpp
    plugin/io/handlers/OctreeFieldHandler.cpp
    plugin/io/db/DBConnector.cpp
    plugin/FieldRendererPlugin.cpp
    plugin/api/Params.cpp
)

set(${NAME}_PUBLIC_HEADERS
    plugin/FieldRendererPlugin.h
    plugin/common/Utils.h
    plugin/io/db/DBConnector.h
    plugin/io/handlers/FieldHandler.h
    plugin/io/octree/OctreeNode.h
    plugin/io/octree/Octree.h
    plugin/io/handlers/OctreeFieldHandler.h
    plugin/api/Params.h
)

set(${NAME}_ISPC_SOURCES
    module/ispc/render/FieldRenderer.ispc
    module/ispc/render/OctreeFieldRenderer.ispc
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
    ${OSPRAY_CMAKE_ROOT}    
    ${PROJECT_SOURCE_DIR}
    ${OSPRAY_INCLUDE_DIRS}
    ${BRAYNS_INCLUDE_DIRS}
    ${GLM_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${${NAME}_MODULES_DIR}
)

set(${NAME}_LINK_LIBRARIES
    PUBLIC ${OSPRAY_LIBRARIES} pqxx
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
