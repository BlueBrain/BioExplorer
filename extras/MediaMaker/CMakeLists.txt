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
project(${NAME} VERSION 0.1.0)
set(${NAME}_VERSION_ABI 1)

# ==============================================================================
# Packages and settings
# ==============================================================================
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake)
set(${NAME}_MODULES_DIR ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 14)

find_package(Brayns REQUIRED SYSTEM)

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

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${BRAYNS_INCLUDE_DIRS}
    ${${NAME}_MODULES_DIR}
)

set(${NAME}_LINK_LIBRARIES
    PRIVATE braynsParameters braynsPluginAPI braynsCommon braynsEngine braynsIO
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
