# Copyright (c) 2015-2016, EPFL/Blue Brain Project
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of Brayns <https://github.com/BlueBrain/Brayns>

# Present the CUDA_64_BIT_DEVICE_CODE on the default set of options.
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

set(CUDA_MIN_SM_TARGET sm_50 CACHE STRING "Minimum CUDA SM architecture to use for compilation.")

# For FindOptix and CUDA patch
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

common_find_package(CUDA REQUIRED SYSTEM)

if(CUDA_FOUND)
  # This doesn't get called from common_find_package_post unless we export some
  # variables to the parent scope
  find_cuda_compatible_host_compiler()
endif()

message(STATUS "Using CUDA version ${CUDA_VERSION}")

set(CUDA_NVCC_FLAGS "-lineinfo --use_fast_math")

find_package(OptiX REQUIRED SYSTEM)

include_directories(
  ${OptiX_INCLUDE}
  ${OptiX_INSTALL_DIR}/SDK
  ${OptiX_INSTALL_DIR}/SDK/sutil
  ${OptiX_INSTALL_DIR}/SDK/support
  ${OptiX_INSTALL_DIR}/SDK/build
)

set(BRAYNSOPTIXENGINE_OMIT_LIBRARY_HEADER ON)
set(BRAYNSOPTIXENGINE_OMIT_VERSION_HEADERS ON)

set_target_properties(
  ${target_name} PROPERTIES
  COMPILE_DEFINITIONS
  "OPTIX_SAMPLE_NAME_DEFINE=${target_name};OPTIX_SAMPLE_DIR_DEFINE=${target_name}")

set(BRAYNSOPTIXENGINE_CU
  cuda/camera/PerspectiveCamera.cu

  # cuda/camera/OpenDeckCamera.cu
  # cuda/Constantbg.cu
  # cuda/geometry/Cones.cu
  # cuda/geometry/Cylinders.cu
  cuda/geometry/Spheres.cu

  # cuda/geometry/TriangleMesh.cu
  # cuda/renderer/AdvancedSimulation.cu
  # cuda/renderer/BasicSimulation.cu
  # cuda/renderer/PBR.cu
  # cuda/renderer/BBP.cu
  cuda/renderer/Material.cu
)

CUDA_WRAP_SRCS(braynsOptixEngine PTX ptx_generated_files ${BRAYNSOPTIXENGINE_CU})

include(StringifyPtx)
stringify_ptx(${ptx_generated_files})

set(BRAYNSOPTIXENGINE_SOURCES
  ${BRAYNSOPTIXENGINE_CU}
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

set(BRAYNSOPTIXENGINE_HEADERS
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
link_directories(${OptiX_INSTALL_DIR}/SDK/build/lib)

set(BRAYNSOPTIXENGINE_LINK_LIBRARIES
  PRIVATE braynsEngine glm braynsCommon braynsParameters sutil_7_sdk ${CUDA_LIBRARIES})

common_library(braynsOptixEngine)
