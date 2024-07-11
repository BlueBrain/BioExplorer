# Copyright 2020 - 2024 Blue Brain Project / EPFL
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================
# Project
# ==============================================================================
set(NAME DICOM)
set(LIBRARY_NAME DICOM)

# ==============================================================================
# Packages and settings
# ==============================================================================
find_package(DCMTK REQUIRED)

# Compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_STANDARD 17)

include_directories(
  SYSTEM
  ${BIOEXPLORER_SOURCE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES
  plugin/DICOMPlugin.cpp
  plugin/io/DICOMLoader.cpp
)

set(${NAME}_PUBLIC_HEADERS
  plugin/DICOMPlugin.h
  plugin/io/DICOMLoader.h
)

set(${NAME}_PUBLIC_MODULE_LIBRARIES)
set(${NAME}_PRIVATE_MODULE_LIBRARIES)
# ==============================================================================
# OptiX 6 module
# ==============================================================================
if(PLATFORM_OPTIX6_ENABLED)
    message(STATUS "[DICOM] Building OptiX 6 module")
    list(APPEND CMAKE_MODULE_PATH ${BIOEXPLORER_SOURCE_DIRS}/CMake)

    find_package(CUDA REQUIRED)
    if(CUDA_FOUND)
      # This doesn't get called from common_find_package_post unless we export some
      # variables to the parent scope
      find_cuda_compatible_host_compiler()
    endif()
    message(STATUS "Using CUDA version ${CUDA_VERSION}")
    
    set(CUDA_NVCC_FLAGS "--use_fast_math")
    
    set(${NAME}_CU
        module/cuda/renderer/DICOM.cu
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
endif()

# ==============================================================================
# Compile c++ code
# ==============================================================================
set(${NAME}_LINK_LIBRARIES
  CoreParameters CoreCommon CoreEngine
  ${DCMTK_LIBRARIES})

add_library(
  ${LIBRARY_NAME}
  SHARED ${${NAME}_SOURCES})

target_link_libraries(
  ${LIBRARY_NAME}
  ${${NAME}_LINK_LIBRARIES})

# ================================================================================
# Install binaries
# ================================================================================
INSTALL(TARGETS ${LIBRARY_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

# ==============================================================================
# Install binaries
# ==============================================================================
INSTALL(TARGETS ${LIBRARY_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)