# Copyright (c) 2020-2022, Cyrille Favreau
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of https://github.com/BlueBrain/BioExplorer

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(NAME BIOEXPLORER)
set(PACKAGE_VERSION 1.4.0)
project(${NAME} VERSION ${PACKAGE_VERSION})

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake)
set(${NAME}_MODULES_DIR ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 14)

find_package(Brayns REQUIRED)
find_package(ospray REQUIRED SYSTEM)
find_package(Boost REQUIRED COMPONENTS program_options filesystem)
find_package(glm REQUIRED SYSTEM)
find_package(FreeImage REQUIRED)

add_subdirectory(bioexplorer/core)

option(BIOEXPLORER_MEDIA_MAKER_ENABLED "Enable media maker plugin" ON)

if(BIOEXPLORER_MEDIA_MAKER_ENABLED)
    message(STATUS "Building MediaMaker plugin")
    add_subdirectory(extras/MediaMaker)
endif()

option(BIOEXPLORER_METABOLISM_ENABLED "Enable metabolism plugin" ON)

if(BIOEXPLORER_METABOLISM_ENABLED)
    message(STATUS "Building Metabolism plugin")
    add_subdirectory(extras/Metabolism)
endif()
