# Copyright (c) 2015-2023, EPFL/Blue Brain Project
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of Brayns <https://github.com/BlueBrain/Brayns>

# ==============================================================================
# Project
# ==============================================================================
set(NAME OPTIX7_ENGINE)
set(LIBRARY_NAME OptiX7Engine)

# Present the CUDA_64_BIT_DEVICE_CODE on the default set of options.
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

set(CUDA_MIN_SM_TARGET sm_50 CACHE STRING "Minimum CUDA SM architecture to use for compilation.")

# For FindOptix and CUDA patch
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

if(CUDA_FOUND)
  # This doesn't get called from common_find_package_post unless we export some
  # variables to the parent scope
  find_cuda_compatible_host_compiler()
endif()

message(STATUS "Using CUDA version ${CUDA_VERSION}")

set(CUDA_NVCC_FLAGS "-lineinfo --use_fast_math")

find_package(OptiX7 REQUIRED SYSTEM)

include_directories(
  ${OptiX7_INCLUDE}
  ${OptiX7_INSTALL_DIR}/SDK
  ${OptiX7_INSTALL_DIR}/SDK/sutil
  ${OptiX7_INSTALL_DIR}/SDK/support
  ${OptiX7_INSTALL_DIR}/SDK/build
)

set(BRAYNSOPTIX7ENGINE_OMIT_LIBRARY_HEADER ON)
set(BRAYNSOPTIX7ENGINE_OMIT_VERSION_HEADERS ON)

set_target_properties(
  ${target_name} PROPERTIES
  COMPILE_DEFINITIONS
  "OPTIX_SAMPLE_NAME_DEFINE=${target_name};OPTIX_SAMPLE_DIR_DEFINE=${target_name}")

set(BRAYNSOPTIX7ENGINE_CU
  cuda/camera/PerspectiveCamera.cu

  # cuda/camera/OpenDeckCamera.cu
  # cuda/Constantbg.cu
  cuda/geometry/Cones.cu
  cuda/geometry/Cylinders.cu
  cuda/geometry/Spheres.cu

  # cuda/geometry/TriangleMesh.cu
  # cuda/renderer/AdvancedSimulation.cu
  # cuda/renderer/BasicSimulation.cu
  cuda/renderer/Material.cu
)

CUDA_WRAP_SRCS(braynsOptix7Engine PTX ptx_generated_files ${BRAYNSOPTIX7ENGINE_CU})

include(StringifyPtx)
stringify_ptx(${ptx_generated_files})

set(BRAYNSOPTIX7ENGINE_SOURCES
  ${BRAYNSOPTIX7ENGINE_CU}
  ${ptx_generated_files}
  ${PTX_SOURCES}
  OptiXContext.cpp
  OptiXFrameBuffer.cpp
  OptiXScene.cpp
  OptiXCamera.cpp
  OptiXPerspectiveCamera.cpp

  # OptiXOpenDeckCamera.cpp
  OptiXRenderer.cpp
  OptiXEngine.cpp
  OptiXMaterial.cpp
  OptiXModel.cpp
)

set_source_files_properties(
  OptiXContext.cpp
  OptiXFrameBuffer.cpp
  OptiXScene.cpp
  OptiXCamera.cpp
  OptiXPerspectiveCamera.cpp

  # OptiXOpenDeckCamera.cpp
  OptiXRenderer.cpp
  OptiXEngine.cpp
  OptiXMaterial.cpp
  OptiXModel.cpp
  PROPERTIES COMPILE_FLAGS "-Wno-unused-parameter -Wno-missing-field-initializers"
)

set(BRAYNSOPTIX7ENGINE_HEADERS
  ${PTX_HEADERS}
  OptiXContext.h

  # OptiXFrameBuffer.h
  OptiXScene.h
  OptiXCamera.h

  # OptiXCameraProgram.h
  # OptiXOpenDeckCamera.h
  OptiXPerspectiveCamera.h
  OptiXRenderer.h
  OptiXEngine.h
  OptiXMaterial.h
  OptiXTypes.h
  OptiXUtils.h
)

# TODO: OptiX should be properly installed, and not used from its build folder
link_directories(${OptiX7_INSTALL_DIR}/SDK/build/lib)

set(BRAYNSOPTIX7ENGINE_LINK_LIBRARIES
  PRIVATE CoreEngine ${GLM_LIBRARIES} CoreCommon CoreParameters sutil_7_sdk ${CUDA_LIBRARIES})


# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(
  ${BIOEXPLORER_SOURCE_DIRS}
  ${CMAKE_BINARY_DIR}
  ${OptiX6_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

set(${NAME}_INCLUDE_NAME engines/optix6)

add_library(
    ${LIBRARY_NAME}
    SHARED ${${NAME}_SOURCES})

    target_link_libraries(
      ${LIBRARY_NAME}
      ${${NAME}_LINK_LIBRARIES}
  )
  
  # ==============================================================================
  # Install binaries
  # ==============================================================================
  install(
    TARGETS ${LIBRARY_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
  