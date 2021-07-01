# Copyright (c) 2020, Cyrille Favreau
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of https://github.com/favreau/Brayns-UC-MediaMaker

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# ==============================================================================
# Project
# ==============================================================================
set(NAME MEDIAMAKER)
set(LIBRARY_NAME mediaMaker)
project(${NAME} VERSION ${PACKAGE_VERSION})
set(${NAME}_VERSION_ABI 1)

# ==============================================================================
# Packages and settings
# ==============================================================================
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../CMake)
set(${NAME}_MODULES_DIR ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 14)

find_package(Brayns REQUIRED SYSTEM)
find_package(ospray REQUIRED SYSTEM)
find_package(TIFF REQUIRED SYSTEM)

include(ispc)

configure_file( plugin/Defines.h.in ${CMAKE_BINARY_DIR}/generated/mediamaker/Defines.h )
include_directories( ${CMAKE_BINARY_DIR}/generated/mediamaker ) # Make sure it can be included...

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
    PUBLIC ${OSPRAY_LIBRARIES} ${TIFF_LIBRARIES}
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
