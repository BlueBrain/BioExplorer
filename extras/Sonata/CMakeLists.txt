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

# ==============================================================================
# Project
# ==============================================================================
set(LIBRARY_NAME SonataExplorer)
set(${NAME}_VERSION_ABI 1)

# ==============================================================================
# Packages and settings
# ==============================================================================
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake)
set(${NAME}_MODULES_DIR ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 14) # C++ 14

# Packages
find_package(Brayns REQUIRED)
find_package(ospray REQUIRED SYSTEM)
find_package(Brion REQUIRED)

# OpenMP
find_package(OpenMP)

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Optional packages (managed by Defines.h.in)
set(CGAL_FOUND 0)
option(${NAME}_USE_CGAL "Use CGAL meshing features" ON)

if(${${NAME}_USE_CGAL})
    find_package(CGAL REQUIRED)
endif()

set(BIOEXPLORER_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../..)
configure_file(common/Defines.h.in ${CMAKE_BINARY_DIR}/generated/circuitexplorer/Defines.h)
include_directories(${PROJECT_SOURCE_DIR}/core ${CMAKE_BINARY_DIR}/generated/circuitexplorer)

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES
    common/Utils.cpp
    plugin/SonataExplorerPlugin.cpp
    plugin/api/SonataExplorerParams.cpp
    plugin/io/BrickLoader.cpp
    module/ispc/camera/SphereClippingPerspectiveCamera.cpp
)

set(${NAME}_PUBLIC_HEADERS
    common/Utils.h
    plugin/SonataExplorerPlugin.h
    plugin/api/SonataExplorerParams.h
    plugin/io/BrickLoader.h
)

list(APPEND ${NAME}_SOURCES
    module/ispc/render/CellGrowthRenderer.cpp
    module/ispc/render/ProximityDetectionRenderer.cpp
    module/ispc/render/CellGrowthRenderer.cpp
    plugin/neuroscience/common/MorphologyLoader.cpp
    plugin/neuroscience/common/ParallelModelContainer.cpp
    plugin/neuroscience/neuron/CellGrowthHandler.cpp
    plugin/neuroscience/neuron/VoltageSimulationHandler.cpp
    plugin/neuroscience/neuron/SpikeSimulationHandler.cpp
    plugin/neuroscience/astrocyte/AstrocyteLoader.cpp
    plugin/neuroscience/neuron/AbstractCircuitLoader.cpp
    plugin/neuroscience/neuron/PairSynapsesLoader.cpp
    plugin/neuroscience/neuron/MeshCircuitLoader.cpp
    plugin/neuroscience/neuron/MorphologyCollageLoader.cpp
    plugin/neuroscience/neuron/AdvancedCircuitLoader.cpp
    plugin/neuroscience/neuron/SynapseCircuitLoader.cpp
    plugin/meshing/MetaballsGenerator.cpp
)
list(APPEND ${NAME}_PUBLIC_HEADERS
    plugin/neuroscience/common/ParallelModelContainer.h
    plugin/neuroscience/common/MorphologyLoader.h
    plugin/neuroscience/neuron/CellGrowthHandler.h
    plugin/neuroscience/neuron/VoltageSimulationHandler.h
    plugin/neuroscience/neuron/SpikeSimulationHandler.h
    plugin/neuroscience/astrocyte/AstrocyteLoader.h
    plugin/neuroscience/neuron/AbstractCircuitLoader.h
    plugin/neuroscience/neuron/PairSynapsesLoader.h
    plugin/neuroscience/neuron/MeshCircuitLoader.h
    plugin/neuroscience/neuron/MorphologyCollageLoader.h
    plugin/neuroscience/neuron/AdvancedCircuitLoader.h
    plugin/neuroscience/neuron/SynapseCircuitLoader.h
    plugin/meshing/MetaballsGenerator.h
)

if(${CGAL_FOUND})
    list(APPEND ${NAME}_SOURCES plugin/meshing/PointCloudMesher.cpp)
    list(APPEND ${NAME}_PUBLIC_HEADERS plugin/meshing/PointCloudMesher.h)
endif()

set(${NAME}_ISPC_SOURCES
    module/ispc/camera/utils.ispc
    module/ispc/camera/SphereClippingPerspectiveCamera.ispc
)

list(APPEND ${NAME}_ISPC_SOURCES
    module/ispc/render/ProximityDetectionRenderer.ispc
    module/ispc/render/CellGrowthRenderer.ispc
)

# ==============================================================================
# Compile ispc code
# ==============================================================================
list(APPEND CMAKE_MODULE_PATH ${OSPRAY_CMAKE_ROOT})

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    set(OSPRAY_DEBUG_BUILD ON)
endif()

include(ispc)
list(APPEND CMAKE_MODULE_PATH ${OSPRAY_CMAKE_ROOT})
list(APPEND ALL_ISPC_INCLUDES ${${NAME}_MODULES_DIR})
list(APPEND ALL_ISPC_INCLUDES ${OSPRAY_INCLUDE_DIRS})
list(APPEND ALL_ISPC_INCLUDES ${GLM_INCLUDE_DIRS})
list(APPEND ALL_ISPC_INCLUDES ${BIOEXPLORER_SOURCE_DIR})

include_directories_ispc(${ALL_ISPC_INCLUDES})
ospray_ispc_compile(${${NAME}_ISPC_SOURCES})
list(APPEND ${NAME}_SOURCES ${ISPC_OBJECTS})

# ==============================================================================
# Compile C++ code
# ==============================================================================
include_directories(
    SYSTEM
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${BRAYNS_INCLUDE_DIRS}
    ${BIOEXPLORER_SOURCE_DIR}
)

set(${NAME}_LINK_LIBRARIES
    PUBLIC ${OSPRAY_LIBRARIES}
    PRIVATE braynsCommon braynsParameters braynsIO braynsEngine braynsPluginAPI
    braynsOSPRayEngine glm
    ${FREEIMAGE_LIBRARIES} ${HDF5_LIBRARIES}
)

list(APPEND ${NAME}_LINK_LIBRARIES Brion Brain)

if(${CGAL_FOUND})
    list(APPEND ${NAME}_LINK_LIBRARIES ${CGAL_LIBRARIES})
endif()

if(${PQXX_FOUND})
    list(APPEND ${NAME}_LINK_LIBRARIES ${PQXX_LIBRARIES})
endif()

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
