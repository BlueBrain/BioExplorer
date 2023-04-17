# Copyright (c) 2015-2023, EPFL/Blue Brain Project
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
# Grigori Chevtchenko <grigori.chevtchenko@epfl.ch>
#
# This file is part of Brayns <https://github.com/BlueBrain/Brayns>

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(Brayns VERSION 1.0.0)
set(Brayns_VERSION_ABI 1)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake
  ${CMAKE_SOURCE_DIR}/CMake/common)

if(NOT EXISTS ${CMAKE_SOURCE_DIR}/CMake/common/Common.cmake)
  message(FATAL_ERROR "CMake/common missing, run: git submodule update --init")
endif()

set(BRAYNS_LICENSE LGPL)
set(BRAYNS_DESCRIPTION "Hardware Agnostic Ray-Tracer")
set(BRAYNS_MAINTAINER_NAME "Blue Brain Project")
set(BRAYNS_MAINTAINER_EMAIL "<bbp-open-source@googlegroups.com>")
set(BRAYNS_MAINTAINER "${BRAYNS_MAINTAINER_NAME} ${BRAYNS_MAINTAINER_EMAIL}")

set(BRAYNS_PORT_DEPEND "assimp boost freeimage glew tbb")

# Unit testing
option(BRAYNS_UNIT_TESTING_ENABLED "Unit testing" ON)

# JSON message live debugging
option(BRAYNS_DEBUG_JSON_ENABLED "Catch JSON exceptions during runtime" OFF)

if(BRAYNS_DEBUG_JSON_ENABLED)
  add_definitions(-DBRAYNS_DEBUG_JSON_ENABLED)
endif(BRAYNS_DEBUG_JSON_ENABLED)

# Enable cloning of all optional subprojects for CI builds
if($ENV{TRAVIS})
  if(NOT DEFINED BRAYNS_NETWORKING_ENABLED)
    option(BRAYNS_NETWORKING_ENABLED "Activate networking interfaces" ON)
  endif()

  if(NOT DEFINED BRAYNS_DEFLECT_ENABLED)
    option(BRAYNS_DEFLECT_ENABLED "Activate streaming to display wall" OFF)
  endif()
endif()

set(CHOOSE_PYTHON_IGNORE_BOOST ON)
include(Common)

# Compiler options
if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function -Wno-class-memaccess -Wno-ignored-qualifiers")
endif()

if((CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0) OR(CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-copy")
endif()

# Core dependencies
common_find_package(Boost REQUIRED COMPONENTS program_options)
common_find_package(OpenMP)
common_find_package(libuv 1.8)
common_find_package(FreeImage)

# Archive loading
common_find_package(LibArchive)
option(BRAYNS_LIBARCHIVE_ENABLED "Activate libarchive for archive loading" ${LibArchive_FOUND})

if(BRAYNS_LIBARCHIVE_ENABLED)
  common_find_package(LibArchive REQUIRED)
else()
  common_find_package_disable(LibArchive)
endif()

# HTTP messaging
common_find_package(LibJpegTurbo)
common_find_package(Rockets)
common_find_package(FFMPEG 3.4 SYSTEM)

if(ROCKETS_FOUND AND ROCKETS_USE_LIBWEBSOCKETS)
  option(BRAYNS_NETWORKING_ENABLED "Activate networking interfaces" ON)
else()
  option(BRAYNS_NETWORKING_ENABLED "Activate networking interfaces" OFF)
endif()

if(BRAYNS_NETWORKING_ENABLED)
  # We require turbojpeg for networking to get image streaming working
  common_find_package(LibJpegTurbo REQUIRED)
  common_find_package(Rockets REQUIRED)

  if(NOT ROCKETS_USE_LIBWEBSOCKETS)
    message(WARNING "Could not enable networking, missing Rockets with libwebsockets")
  endif()

  list(APPEND COMMON_FIND_PACKAGE_DEFINES BRAYNS_USE_NETWORKING)
else()
  common_find_package_disable(LibJpegTurbo)
  common_find_package_disable(Rockets)
endif()

# OSPRay rendering engine
common_find_package(ospray 1.8 SYSTEM)
option(BRAYNS_OSPRAY_ENABLED "Activate OSPRay rendering engine" ${OSPRAY_FOUND})

if(BRAYNS_OSPRAY_ENABLED)
  common_find_package(ospray 1.8 SYSTEM REQUIRED)
else()
  common_find_package_disable(ospray)
endif()

# OptiX 6.x
option(BRAYNS_OPTIX6_ENABLED "Activate OptiX 6.x rendering engine" OFF)
option(BRAYNS_OPTIX6_TESTS_ENABLED "Activate OptiX tests" OFF)

# OptiX 7.x
option(BRAYNS_OPTIX7_ENABLED "Activate OptiX 7.x rendering engine" OFF)

if(BRAYNS_OPTIX6_ENABLED)
  add_subdirectory(engines/optix6)
endif()

if(BRAYNS_OPTIX7_ENABLED)
  add_subdirectory(engines/optix7)
endif()

# Mesh loading
common_find_package(assimp SYSTEM)
option(BRAYNS_ASSIMP_ENABLED "Activate assimp library for standard 3D file format management" ${assimp_FOUND})

if(BRAYNS_ASSIMP_ENABLED)
  common_find_package(assimp SYSTEM REQUIRED)
else()
  common_find_package_disable(assimp)
endif()

common_find_package_post()

list(APPEND BRAYNS_DEPENDENT_LIBRARIES Async++ glm)

add_subdirectory(deps)

if(libuv_FOUND)
  # The libuv version of Ubuntu 16.04 is too old to have this definition that is
  # required by uvw (in fact, it uses an old enough version as well to make
  # things work, but couldn't go further).
  message("LIBUV_VERSION " ${libuv_VERSION})

  if(libuv_VERSION VERSION_LESS 1.9.0)
    add_definitions(-DUV_DISCONNECT=4)
  endif()

  include_directories(SYSTEM deps/uvw/src)
endif()

# ------------------------------------------------------------------------------
# BRAYNS applications and libraries
# ------------------------------------------------------------------------------
if(libuv_FOUND)
  option(BRAYNS_SERVICE_ENABLED "Brayns Service" ON)

  if(BRAYNS_SERVICE_ENABLED)
    add_subdirectory(apps/BraynsService)
  endif()
else()
  set(BRAYNS_SERVICE_ENABLED OFF)
endif()

option(BRAYNS_BENCHMARK_ENABLED "Brayns Benchmark" ON)

if(BRAYNS_BENCHMARK_ENABLED)
  add_subdirectory(apps/BraynsBenchmark)
endif()

if(BRAYNS_OSPRAY_ENABLED)
  add_subdirectory(engines/ospray)
else()
  message(WARNING "OSPRay not found or enabled. Brayns cannot work without an engine")
endif()

add_subdirectory(plugins)
add_subdirectory(brayns)

if(BRAYNS_UNIT_TESTING_ENABLED)
  add_subdirectory(tests)
endif()

include(CommonCPack)

set(DOXYGEN_MAINPAGE_MD README.md)
set(DOXYGEN_EXTRA_INPUT "${PROJECT_SOURCE_DIR}/README.md ${PROJECT_SOURCE_DIR}/Changelog.md")
set(COMMON_PROJECT_DOMAIN ch.epfl.bluebrain)
include(DoxygenRule)
