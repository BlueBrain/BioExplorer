# Copyright 2015 - 2024 Blue Brain Project / EPFL
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
#

# ==============================================================================
# Project
# ==============================================================================
set(NAME PLATFORM_PLUGIN_OPENDECK)
set(LIBRARY_NAME CorePluginOpenDeck)

set(
  ${NAME}_HEADERS
  plugin/common/OpenDeckParameters.h
  OpenDeckPlugin.h
)

set(
  ${NAME}_SOURCES
  plugin/common/OpenDeckParameters.cpp
  OpenDeckPlugin.cpp
)

set(${NAME}_LINK_LIBRARIES PRIVATE CoreCommon CoreEngine)

# ==============================================================================
# OSPRay module
# ==============================================================================
if(OSPRAY_FOUND)
  message(STATUS "[OpenDeck] OSPRay 1 module enabled")

  list(
    APPEND ${NAME}_SOURCES
    module/ispc/cylindric/CylindricCamera.cpp
    module/ispc/cylindricStereo/CylindricStereoCamera.cpp
    module/ispc/cylindricStereoTracked/CylindricStereoTrackedCamera.cpp)

  set(
    ${NAME}_ISPC_SOURCES
    module/ispc/cylindric/CylindricCamera.ispc
    module/ispc/cylindricStereo/CylindricStereoCamera.ispc
    module/ispc/cylindricStereoTracked/CylindricStereoTrackedCamera.ispc)

  list(APPEND ${NAME}_SOURCES ${${NAME}_ISPC_SOURCES})

  # reuse ispc setup and macros from ospray
  list(APPEND CMAKE_MODULE_PATH ${OSPRAY_CMAKE_ROOT})
  if(CMAKE_BUILD_TYPE STREQUAL Debug)
    set(OSPRAY_DEBUG_BUILD ON)
  endif()
  include(ispc)

  # Compile ispc code
  list(APPEND CMAKE_MODULE_PATH ${OSPRAY_CMAKE_ROOT})
  list(APPEND ALL_ISPC_INCLUDES ${BIOEXPLORER_SOURCE_DIRS})
  list(APPEND ALL_ISPC_INCLUDES ${${NAME}_MODULES_DIR})
  list(APPEND ALL_ISPC_INCLUDES ${OSPRAY_INCLUDE_DIRS})
  list(APPEND ALL_ISPC_INCLUDES ${GLM_INCLUDE_DIRS})
  include_directories_ispc(${ALL_ISPC_INCLUDES})
  ospray_ispc_compile(${${NAME}_ISPC_SOURCES})
  list(APPEND ${NAME}_SOURCES ${ISPC_OBJECTS})
  list(APPEND ${NAME}_LINK_LIBRARIES ${OSPRAY_LIBRARIES})
  include_directories(${OSPRAY_INCLUDE_DIRS})
endif()

# ==============================================================================
# OptiX 6 module
# ==============================================================================
if(OPTIX6_FOUND)
    list(APPEND CMAKE_MODULE_PATH ${BIOEXPLORER_SOURCE_DIRS}/CMake)

    if(CUDA_FOUND)
      # This doesn't get called from common_find_package_post unless we export some
      # variables to the parent scope
      find_cuda_compatible_host_compiler()
    endif()
    message(STATUS "[OpenDeck] OptiX 6 module enabled")
    
    set(CUDA_NVCC_FLAGS "--use_fast_math")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-shadow")
    
    set(${LIBRARY_NAME}_CU
        module/cuda/camera/cylindricStereo/CylindricStereoCamera.cu
    )

    list(APPEND ${NAME}_SOURCES plugin/optix6/OptiXCylindricStereoCamera.cpp)
    
    CUDA_WRAP_SRCS(${LIBRARY_NAME} PTX ptx_generated_files ${${LIBRARY_NAME}_CU})
    
    include(StringifyPtx)
    stringify_ptx(${ptx_generated_files})

    list(
      APPEND ${NAME}_SOURCES
      ${${LIBRARY_NAME}_CU}
      ${ptx_generated_files}
      ${PTX_SOURCES}
    )

    list(APPEND ${LIBRARY_NAME}_HEADERS ${PTX_HEADERS})

    include_directories(${OptiX6_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
    list(APPEND ${NAME}_LINK_LIBRARIES ${OptiX6_LIBRARIES} OptiX6Engine)
endif()

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR})

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
