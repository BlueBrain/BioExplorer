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
set(LIBRARY_NAME BioExplorer)
set(${NAME}_VERSION_ABI 1)

# ==============================================================================
# Packages and settings
# ==============================================================================
option(${NAME}_USE_CGAL "Use CGAL meshing features" ON)

add_subdirectory(deps)

# Packages
find_package(OpenMP)
find_package(PQXX REQUIRED)

# Compiler flags
add_compile_options("-fopenmp")

# Optional packages (managed by Defines.h.in)
set(CGAL_FOUND 0)
if(${${NAME}_USE_CGAL})
    set(CGAL_DATA_DIR ".")
    find_package(CGAL REQUIRED)
endif()

configure_file(
    plugin/common/Defines.h.in
    ${CMAKE_BINARY_DIR}/generated/bioexplorer/Defines.h
)

include_directories(
    ${PROJECT_SOURCE_DIR}/bioexplorer/core
    ${CMAKE_BINARY_DIR}/generated/bioexplorer
)

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES
    plugin/common/GeneralSettings.cpp
    plugin/common/Utils.cpp
    plugin/common/UniqueId.cpp
    plugin/common/Node.cpp
    plugin/common/SDFGeometries.cpp
    plugin/common/shapes/Shape.cpp
    plugin/common/shapes/MeshShape.cpp
    plugin/common/shapes/SphereShape.cpp
    plugin/common/shapes/SphericalCellDiffusionShape.cpp
    plugin/common/shapes/HelixShape.cpp
    plugin/common/shapes/PlaneShape.cpp
    plugin/common/shapes/CubeShape.cpp
    plugin/common/shapes/BezierShape.cpp
    plugin/common/shapes/SinusoidShape.cpp
    plugin/common/shapes/FanShape.cpp
    plugin/common/shapes/RNAShape.cpp
    plugin/common/shapes/PointShape.cpp
    plugin/common/Assembly.cpp
    plugin/common/ThreadSafeContainer.cpp
    plugin/fields/FieldsHandler.cpp
    plugin/fields/Octree.cpp
    plugin/fields/OctreeNode.cpp
    plugin/io/CacheLoader.cpp
    plugin/io/OOCManager.cpp
    plugin/io/db/DBConnector.cpp
    plugin/molecularsystems/EnzymeReaction.cpp
    plugin/molecularsystems/Molecule.cpp
    plugin/molecularsystems/Membrane.cpp
    plugin/molecularsystems/Glycans.cpp
    plugin/molecularsystems/Protein.cpp
    plugin/molecularsystems/RNASequence.cpp
    plugin/morphologies/Morphologies.cpp
    plugin/atlas/Atlas.cpp
    plugin/morphologies/Synapses.cpp
    plugin/morphologies/Astrocytes.cpp
    plugin/morphologies/Neurons.cpp
    plugin/morphologies/SpikeSimulationHandler.cpp
    plugin/morphologies/SomaSimulationHandler.cpp
    plugin/morphologies/CompartmentSimulationHandler.cpp
    plugin/vasculature/VasculatureHandler.cpp
    plugin/vasculature/Vasculature.cpp
    plugin/connectomics/whitematter/WhiteMatter.cpp
    plugin/connectomics/synapses/SynapseEfficacy.cpp
    plugin/connectomics/synapses/SynapseEfficacySimulationHandler.cpp
    plugin/meshing/PointCloudMesher.cpp
    plugin/meshing/SurfaceMesher.cpp
    plugin/api/Params.cpp
    plugin/BioExplorerPlugin.cpp
)

set(${NAME}_PUBLIC_HEADERS
    plugin/io/OOCManager.h
    plugin/io/CacheLoader.h
    plugin/io/db/DBConnector.h
    plugin/common/Assembly.h
    plugin/common/Node.h
    plugin/common/SDFGeometries.h
    plugin/molecularsystems/EnzymeReaction.h
    plugin/molecularsystems/Molecule.h
    plugin/molecularsystems/Glycans.h
    plugin/molecularsystems/Membrane.h
    plugin/molecularsystems/Protein.h
    plugin/molecularsystems/RNASequence.h
    plugin/atlas/Atlas.h
    plugin/morphologies/Synapses.h
    plugin/morphologies/Morphologies.h
    plugin/morphologies/Astrocytes.h
    plugin/morphologies/Neurons.h
    plugin/morphologies/SpikeSimulationHandler.h
    plugin/morphologies/SomaSimulationHandler.h
    plugin/morphologies/CompartmentSimulationHandler.h
    plugin/vasculature/VasculatureHandler.h
    plugin/vasculature/Vasculature.h
    plugin/connectomics/whitematter/WhiteMatter.h
    plugin/connectomics/synapses/SynapseEfficacy.h
    plugin/connectomics/synapses/SynapseEfficacySimulationHandler.h
    plugin/common/UniqueId.h
    plugin/common/GeneralSettings.h
    plugin/common/CommonTypes.h
    plugin/common/ThreadSafeContainer.h
    plugin/common/shapes/Shape.h
    plugin/common/shapes/MeshShape.h
    plugin/common/shapes/SphereShape.h
    plugin/common/shapes/SphericalCellDiffusionShape.h
    plugin/common/shapes/HelixShape.h
    plugin/common/shapes/PlaneShape.h
    plugin/common/shapes/CubeShape.h
    plugin/common/shapes/BezierShape.h
    plugin/common/shapes/SinusoidShape.h
    plugin/common/shapes/FanShape.h
    plugin/common/shapes/RNAShape.h
    plugin/common/shapes/PointShape.h
    plugin/meshing/PointCloudMesher.h
    plugin/meshing/SurfaceMesher.h
    plugin/api/Params.h
    plugin/BioExplorerPlugin.h
)

if(${CGAL_FOUND} AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-frounding-math)
endif()

set(${NAME}_PUBLIC_MODULE_LIBRARIES)
set(${NAME}_PRIVATE_MODULE_LIBRARIES)
# ==============================================================================
# OSPRay module
# ==============================================================================
if(BIOEXPLORER_OSPRAY_ENABLED)
    list(APPEND ${NAME}_SOURCES
        module/ispc/renderer/artistic/GolgiStyleRenderer.cpp
        module/ispc/renderer/FieldsRenderer.cpp
        module/ispc/renderer/DensityRenderer.cpp
        module/ispc/renderer/PathTracingRenderer.cpp
        module/ispc/renderer/VoxelRenderer.cpp
    )

    set(${NAME}_ISPC_SOURCES
        module/ispc/renderer/artistic/GolgiStyleRenderer.ispc
        module/ispc/renderer/FieldsRenderer.ispc
        module/ispc/renderer/DensityRenderer.ispc
        module/ispc/renderer/PathTracingRenderer.ispc
        module/ispc/renderer/VoxelRenderer.ispc
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

    list(APPEND ${NAME}_PUBLIC_MODULE_LIBRARIES ${OSPRAY_LIBRARIES})
    list(APPEND ${NAME}_PRIVATE_MODULE_LIBRARIES braynsOSPRayEngine)
    include_directories(${OSPRAY_INCLUDE_DIRS})
endif()

# ==============================================================================
# OptiX 6 module
# ==============================================================================
if(BIOEXPLORER_OPTIX6_ENABLED)
    list(APPEND CMAKE_MODULE_PATH ${BIOEXPLORER_SOURCE_DIR}/CMake)

    find_package(CUDA REQUIRED)
    if(CUDA_FOUND)
      # This doesn't get called from common_find_package_post unless we export some
      # variables to the parent scope
      find_cuda_compatible_host_compiler()
    endif()
    message(STATUS "[BioExplorer] Using CUDA version ${CUDA_VERSION}")
    
    set(CUDA_NVCC_FLAGS "--use_fast_math")
    
    set(${NAME}_CU
        module/cuda/renderer/artistic/Golgi.cu
        module/cuda/renderer/Density.cu
        module/cuda/renderer/Fields.cu
        module/cuda/renderer/PathTracing.cu
        module/cuda/renderer/Voxel.cu
    )
    
    CUDA_WRAP_SRCS(${LIBRARY_NAME} PTX ptx_generated_files ${${NAME}_CU})
    
    include(StringifyPtx)
    stringify_ptx(${ptx_generated_files})

    list(APPEND ${NAME}_SOURCES
        ${${NAME}_CU}
        ${ptx_generated_files}
        ${PTX_SOURCES}
    )

    list(APPEND ${NAME}_HEADERS ${PTX_HEADERS})
    include_directories(${OptiX6_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
    list(APPEND ${NAME}_PUBLIC_MODULE_LIBRARIES ${OptiX6_LIBRARIES})
    list(APPEND ${NAME}_PRIVATE_MODULE_LIBRARIES braynsOptix6Engine)
endif()

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${OSPRAY_INCLUDE_DIRS}
    ${BRAYNS_INCLUDE_DIRS}
    ${morphio_INCLUDE_DIRS}
    ${GLM_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/bioexplorer/core/module/ispc
    ${${NAME}_MODULES_DIR}
)

set(${NAME}_LINK_LIBRARIES
    PUBLIC ${${NAME}_PUBLIC_MODULE_LIBRARIES}
    PRIVATE braynsParameters braynsPluginAPI braynsCommon braynsEngine braynsIO mpfr
    ${${NAME}_PRIVATE_MODULE_LIBRARIES} ${FreeImage_LIBRARIES} ${PQXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES}
)

if(${CGAL_FOUND})
list(APPEND ${NAME}_LINK_LIBRARIES ${CGAL_LIBRARIES} gmp)
endif()

add_library(
    ${LIBRARY_NAME}
    SHARED ${${NAME}_SOURCES})

target_link_libraries(
    ${LIBRARY_NAME}
    ${${NAME}_LINK_LIBRARIES})

# ==============================================================================
# Unit testing
# ==============================================================================
option(${NAME}_UNIT_TESTING_ENABLED "Enable unit testing" ON)

if(${NAME}_UNIT_TESTING_ENABLED)
    add_subdirectory(tests)
endif()

# ==============================================================================
# Install binaries
# ==============================================================================
INSTALL(TARGETS ${LIBRARY_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
