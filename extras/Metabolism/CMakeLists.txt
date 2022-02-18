# Copyright (c) 2020-2022, Cyrille Favreau
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of https://github.com/BlueBrain/BioExplorer

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# ==============================================================================
# Project
# ==============================================================================
set(NAME METABOLISM)
set(LIBRARY_NAME Metabolism)
project(${NAME} VERSION ${PACKAGE_VERSION})
set(${NAME}_VERSION_ABI 1)

# ==============================================================================
# Packages and settings
# ==============================================================================
set(BIOEXPLORER_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../..)
list(APPEND CMAKE_MODULE_PATH ${BIOEXPLORER_SOURCE_DIR}/CMake)
set(${NAME}_MODULES_DIR ${PROJECT_SOURCE_DIR})

find_package(PQXX REQUIRED)

include(ispc)

include_directories(
    ${BIOEXPLORER_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/generated/bioexplorer
)

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES
    module/ispc/render/MetabolismRenderer.cpp
    plugin/io/db/DBConnector.cpp
    plugin/common/Utils.cpp
    plugin/io/handlers/MetabolismHandler.cpp
    plugin/MetabolismPlugin.cpp
    plugin/api/Params.cpp
)

set(${NAME}_PUBLIC_HEADERS
    plugin/MetabolismPlugin.h
    plugin/common/Utils.h
    plugin/io/db/DBConnector.h
    plugin/io/handlers/MetabolismHandler.h
    plugin/api/Params.h
)

set(${NAME}_ISPC_SOURCES
    module/ispc/render/Glsl.ispc
    module/ispc/render/Cloud.ispc
    module/ispc/render/MetabolismRenderer.ispc
)

# ==============================================================================
# Compile ispc code
# ==============================================================================
list(APPEND CMAKE_MODULE_PATH ${OSPRAY_CMAKE_ROOT})
list(APPEND ALL_ISPC_INCLUDES ${${NAME}_MODULES_DIR})
list(APPEND ALL_ISPC_INCLUDES ${OSPRAY_INCLUDE_DIRS})
list(APPEND ALL_ISPC_INCLUDES ${GLM_INCLUDE_DIRS})
list(APPEND ALL_ISPC_INCLUDES ${BIOEXPLORER_SOURCE_DIR})
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
