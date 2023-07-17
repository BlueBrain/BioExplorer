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

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(NAME BIOEXPLORER)

set(PACKAGE_VERSION_ABI 1)
set(PACKAGE_VERSION_MAJOR 1)
set(PACKAGE_VERSION_MINOR 6)
set(PACKAGE_VERSION_PATCH 0)

# Get the Git revision using the git command
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE PACKAGE_VERSION_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
project(${NAME} VERSION "${PACKAGE_VERSION_MAJOR}.${PACKAGE_VERSION_MINOR}.${PACKAGE_VERSION_PATCH}")
set(PACKAGE_REV_STRING "${PACKAGE_VERSION_MAJOR}.${PACKAGE_VERSION_MINOR}.${PACKAGE_VERSION_PATCH}.${PACKAGE_VERSION_REVISION}")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake)
set(${NAME}_MODULES_DIR ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 14)

find_package(Boost REQUIRED COMPONENTS program_options filesystem)
find_package(glm REQUIRED)
find_package(FreeImage REQUIRED)
find_package(ospray)
find_package(OptiX6)
find_package(CUDA)
find_package(assimp)
find_package(LibJpegTurbo)
find_package(Rockets)

set(BIOEXPLORER_SOURCE_DIRS ${PROJECT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/generated)

# ==============================================================================
# Options
# ==============================================================================

# Find the Intel C++ compiler
find_program(CMAKE_CXX_COMPILER NAMES icpc)
if(CMAKE_CXX_COMPILER)
    # If found, set the compiler and related flags
    message(STATUS "[Core] Building with Intel C++ compiler")
    # Set the compiler
    set(CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "Intel C++ compiler" FORCE)
endif()

set(OSPRAY_FOUND 0)
if(ospray_FOUND)
    include(ispc)
    option(PLATFORM_OSPRAY_ENABLED "Enable OSPRay 1 module" ON)
    if(PLATFORM_OSPRAY_ENABLED)
        set(OSPRAY_FOUND 1)
        message(STATUS "[Core] OSPRay module enabled")
    endif()
endif()

set(OPTIX6_FOUND 0)
if(OptiX6_FOUND AND CUDA_FOUND)
    option(PLATFORM_OPTIX6_ENABLED "Enable OptiX 6 module" OFF)
    if(PLATFORM_OPTIX6_ENABLED)
        set(OPTIX6_FOUND 1)
        message(STATUS "[Core] OptiX 6 module enabled")
    endif()
endif()

set(OPTIX7_FOUND 0)
if(OptiX7_FOUND AND CUDA_FOUND)
    option(PLATFORM_OPTIX7_ENABLED "Enable OptiX 7 module" OFF)
    if(PLATFORM_OPTIX7_ENABLED)
        set(OPTIX7_FOUND 1)
        message(STATUS "[Core] OptiX 7 module enabled")
    endif()
endif()

set(CGAL_FOUND 0)
if(${${NAME}_USE_CGAL})
    set(CGAL_DATA_DIR ".")
    find_package(CGAL REQUIRED)
endif()

set(ASSIMP_FOUND 0)
if(assimp_FOUND)
    option(PLATFORM_ASSIMP_ENABLED "Activate assimp library for standard 3D file format management" ${assimp_FOUND})
    if(PLATFORM_ASSIMP_ENABLED)
        set(ASSIMP_FOUND 1)
        link_directories(${ASSIMP_LIBRARY_DIRS})
        message(STATUS "[Core] Assimp module enabled")
    endif()
endif()

set(NETWORKING_FOUND 0)
if(LibLIBJPEGTURBO_FOUND AND Rockets_FOUND)
    if(ROCKETS_USE_LIBWEBSOCKETS)
        option(PLATFORM_NETWORKING_ENABLED "Activate networking interfaces" ON)
    else()
        option(PLATFORM_NETWORKING_ENABLED "Activate networking interfaces" OFF)
    endif()

    if(PLATFORM_NETWORKING_ENABLED)
        message(STATUS "[Core] Building networking module")
        set(NETWORKING_FOUND 1)
        # We require turbojpeg for networking to get image streaming working

        if(NOT ROCKETS_USE_LIBWEBSOCKETS)
            message(WARNING "Could not enable networking, missing Rockets with libwebsockets")
        endif()

        list(APPEND COMMON_FIND_PACKAGE_DEFINES PLATFORM_USE_NETWORKING)
    endif()
endif()

# ==============================================================================
# Update Defines.h.in and Version.h.in
# ==============================================================================
configure_file(
    common/Defines.h.in
    ${CMAKE_BINARY_DIR}/generated/Defines.h
)

configure_file(
    common/Version.h.in
    ${CMAKE_BINARY_DIR}/generated/Version.h
)

# ==============================================================================
# Core platform
# ==============================================================================
option(PLATFORM_BUILD_ENABLED "Activate platform build" ON)

if(PLATFORM_BUILD_ENABLED OR BIOEXPLORER_BUILD_ENABLED)
    add_subdirectory(common/deps)
endif()

if(PLATFORM_BUILD_ENABLED)
    message(STATUS "[Core] Platform build enabled")
    add_subdirectory(platform)
endif()

# ==============================================================================
# BioExplorer backend
# ==============================================================================
option(BIOEXPLORER_BUILD_ENABLED "Activate BioExplorer build" ON)

if(BIOEXPLORER_BUILD_ENABLED)
    message(STATUS "[Core] BioExplorer build enabled")
    add_subdirectory(bioexplorer/backend)
endif()

# ==============================================================================
# Space explorer
# ==============================================================================
option(SPACEEXPLORER_BUILD_ENABLED "Activate SpaceExplorer build" ON)

if(SPACEEXPLORER_BUILD_ENABLED)
    message(STATUS "[Core] SpaceExplorer build enabled")
    add_subdirectory(spaceexplorer)
endif()
