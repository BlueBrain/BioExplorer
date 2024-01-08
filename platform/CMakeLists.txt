# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2020-2024 Blue BrainProject / EPFL
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

set(NAME Core)

# Define the package and version
set(PACKAGE_NAME platform)
set(PACKAGE_VERSION 1.0.0)

# Unit testing
option(${NAME}_UNIT_TESTING_ENABLED "Unit testing" ON)

# JSON message live debugging
option(${NAME}_DEBUG_JSON_ENABLED "Catch JSON exceptions during runtime" OFF)

if(${NAME}_DEBUG_JSON_ENABLED)
  add_definitions(-D${NAME}_DEBUG_JSON_ENABLED)
endif(${NAME}_DEBUG_JSON_ENABLED)

# Compiler options
if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function -Wno-class-memaccess -Wno-ignored-qualifiers")
endif()

if((CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0) OR(CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-copy")
endif()

# Core dependencies
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenMP)
find_package(Libuv)
find_package(FreeImage)

# Archive loading
find_package(LibArchive)
option(${NAME}_LIBARCHIVE_ENABLED "Activate libarchive for archive loading" ${LibArchive_FOUND})

# HTTP messaging
find_package(LibJpegTurbo)
find_package(Rockets)
find_package(FFMPEG COMPONENTS SWSCALE)

# OpenGL
find_package(GLEW REQUIRED)

add_subdirectory(deps)

if(Libuv_FOUND)
  include_directories(SYSTEM deps/uvw/src)
endif()


# ==============================================================================
# Engines
# ==============================================================================
if(PLATFORM_OSPRAY_ENABLED)
  message(STATUS "[Core] Building OSPRay 1 engine")
  add_subdirectory(engines/ospray)
endif()

if(PLATFORM_OPTIX6_ENABLED)
  message(STATUS "[Core] Building OptiX 6 engine")
  add_subdirectory(engines/optix6)
endif()

# ==============================================================================
# Applications and libraries
# ==============================================================================
if(Libuv_FOUND)
  option(PLATFORM_SERVICE_ENABLED "Service" ON)
  if(PLATFORM_SERVICE_ENABLED)
    message(STATUS "[Core] Building service application")
    add_subdirectory(apps/service)
  endif()
else()
  set(PLATFORM_SERVICE_ENABLED OFF)
endif()

if(GLEW_FOUND)
  option(PLATFORM_VIEWER_ENABLED "Viewer" ON)
  if(PLATFORM_VIEWER_ENABLED)
    message(STATUS "[Core] Building viewer application")
    add_subdirectory(apps/viewer/ui)
    add_subdirectory(apps/viewer)
  endif()
endif()

option(PLATFORM_BENCHMARK_ENABLED "Benchmark" ON)
if(PLATFORM_BENCHMARK_ENABLED)
  message(STATUS "[Core] Building benchmark application")
  add_subdirectory(apps/benchmark)
endif()

add_subdirectory(plugins)
add_subdirectory(core)

if(PLATFORM_UNIT_TESTING_ENABLED)
  add_subdirectory(tests)
endif()

# ==============================================================================
# Packaging and configuration
# ==============================================================================
list(APPEND PLATFORM_TARGETS Async++)

set_target_properties(${PLATFORM_TARGETS}
    PROPERTIES
    VERSION ${PACKAGE_VERSION_MAJOR}
    SOVERSION ${PACKAGE_VERSION_MAJOR}.${PACKAGE_VERSION_MINOR}
)

install(TARGETS ${PLATFORM_TARGETS}
    EXPORT ${PACKAGE_NAME}-targets
    LIBRARY DESTINATION lib
    INCLUDES DESTINATION include)

install(EXPORT ${PACKAGE_NAME}-targets
    DESTINATION share/${PACKAGE_NAME}/CMake
)

install(FILES ${PACKAGE_NAME}-config.cmake
    DESTINATION share/${PACKAGE_NAME}/CMake
)
