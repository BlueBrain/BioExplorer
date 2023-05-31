# Copyright (c) 2015-2023, EPFL/Blue Brain Project
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of Brayns <https://github.com/BlueBrain/Brayns>

# ==============================================================================
# Project
# ==============================================================================
set(NAME OPTIX6_ENGINE)
set(LIBRARY_NAME OptiX6Engine)

# For FindOptix and CUDA patch
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

if(CUDA_FOUND)
  # This doesn't get called from common_find_package_post unless we export some
  # variables to the parent scope
  find_cuda_compatible_host_compiler()
endif()
message(STATUS "Using CUDA version ${CUDA_VERSION}")

set(CUDA_NVCC_FLAGS "--use_fast_math")

# Patch for cmake 3.7->3.9
if(CMAKE_VERSION VERSION_GREATER "3.6" AND CMAKE_VERSION VERSION_LESS "3.10")
  set(CUDA_run_nvcc "${CMAKE_CURRENT_SOURCE_DIR}/CMake/run_nvcc.cmake")
endif()

set(${NAME}_CU
  cuda/camera/PerspectiveCamera.cu
  cuda/camera/OrthographicCamera.cu
  cuda/Constantbg.cu
  cuda/geometry/Cones.cu
  cuda/geometry/Cylinders.cu
  cuda/geometry/Spheres.cu
  cuda/geometry/TriangleMesh.cu
  cuda/renderer/Basic.cu
  cuda/renderer/Advanced.cu
)

CUDA_WRAP_SRCS(${LIBRARY_NAME} PTX ptx_generated_files ${${NAME}_CU})

include(StringifyPtx)
stringify_ptx(${ptx_generated_files})

set(${NAME}_SOURCES
  ${BRAYNSOPTIX6ENGINE_CU}
  ${ptx_generated_files}
  ${PTX_SOURCES}
  OptiXContext.cpp
  OptiXFrameBuffer.cpp
  OptiXScene.cpp
  OptiXCamera.cpp
  OptiXPerspectiveCamera.cpp
  OptiXOrthographicCamera.cpp
  OptiXRenderer.cpp
  OptiXEngine.cpp
  OptiXMaterial.cpp
  OptiXModel.cpp
  OptiXVolume.cpp
)

set_source_files_properties(
  OptiXContext.cpp
  OptiXFrameBuffer.cpp
  OptiXScene.cpp
  OptiXCamera.cpp
  OptiXPerspectiveCamera.cpp
  OptiXOrthographicCamera.cpp
  OptiXRenderer.cpp
  OptiXEngine.cpp
  OptiXMaterial.cpp
  OptiXModel.cpp
  OptiXVolume.cpp
  PROPERTIES COMPILE_FLAGS -Wno-shadow
)

set(${NAME}_HEADERS
  ${PTX_HEADERS}
  OptiXContext.h
  OptiXFrameBuffer.h
  OptiXScene.h
  OptiXCamera.h
  OptiXCameraProgram.h
  OptiXOrthographicCamera.h
  OptiXPerspectiveCamera.h
  OptiXRenderer.h
  OptiXEngine.h
  OptiXVolume.h
)

set(${NAME}_PUBLIC_HEADERS
  OptiXTypes.h
  OptiXContext.h
  OptiXCommonStructs.h
  cuda/renderer/TransferFunction.cuh
  cuda/Random.cuh
  cuda/Helpers.cuh
  cuda/Environment.cuh
)

set(
  ${NAME}_LINK_LIBRARIES
  PRIVATE CoreEngine ${GLM_LIBRARIES} CoreCommon CoreParameters ${CUDA_LIBRARIES} ${OptiX6_LIBRARIES}
)

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


if(CUDA_VERSION VERSION_GREATER 8)
  target_compile_options(${LIBRARY_NAME} PRIVATE "-Wno-error=cpp")
endif()

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
