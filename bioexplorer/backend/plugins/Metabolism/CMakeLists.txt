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

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# ==============================================================================
# Project
# ==============================================================================
set(NAME METABOLISM)
set(LIBRARY_NAME Metabolism)

# ==============================================================================
# Packages and settings
# ==============================================================================
list(APPEND CMAKE_MODULE_PATH ${BIOEXPLORER_SOURCE_DIRS}/CMake)
set(${NAME}_MODULES_DIR ${PROJECT_SOURCE_DIR})

find_package(PQXX REQUIRED)

include(ispc)

include_directories(${BIOEXPLORER_SOURCE_DIRS})

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
list(APPEND ALL_ISPC_INCLUDES ${BIOEXPLORER_SOURCE_DIRS})
include_directories_ispc(${ALL_ISPC_INCLUDES})
ospray_ispc_compile(${${NAME}_ISPC_SOURCES})
list(APPEND ${NAME}_SOURCES ${ISPC_OBJECTS})

# ==============================================================================
# Compile c++ code
# ==============================================================================

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OSPRAY_INCLUDE_DIRS}
)

set(${NAME}_LINK_LIBRARIES
    PUBLIC ${OSPRAY_LIBRARIES} pqxx
    PRIVATE CoreParameters CoreCommon CoreEngine CoreIO
    OSPRayEngine
)

add_library(
    ${LIBRARY_NAME}
    SHARED ${${NAME}_SOURCES})

link_directories(${CMAKE_BINARY_DIR})
    
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
