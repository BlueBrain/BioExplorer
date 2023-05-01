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
set(BIOEXPLORER_PACKAGE_VERSION 1.6.0)
project(${NAME} VERSION ${BIOEXPLORER_PACKAGE_VERSION})

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

option(BIOEXPLORER_SONATA_ENABLED "Enable SONATA plugin" OFF)

if(BIOEXPLORER_SONATA_ENABLED)
    message(STATUS "Building SONATA plugin")
    add_subdirectory(extras/Sonata)
endif()
