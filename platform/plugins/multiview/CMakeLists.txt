# Copyright 2018 - 2024 Blue Brain Project / EPFL
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
# Responsible Author: Grigori Chevtchenko <grigori.chevtchenko@epfl.ch>

# ==============================================================================
# Project
# ==============================================================================
set(NAME PLATFORM_PLUGIN_MULTIVIEW)
set(LIBRARY_NAME CorePluginMultiview)

# ==============================================================================
# Packages and settings
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS})

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_HEADERS MultiviewPlugin.h)
set(${NAME}_SOURCES MultiviewPlugin.cpp)
set(${NAME}_LINK_LIBRARIES PRIVATE CoreCommon CoreEngine CoreParameters)

# ==============================================================================
# OSPRay module
# ==============================================================================
if(PLATFORM_OSPRAY_ENABLED)
  list(APPEND ${NAME}_SOURCES
    ispc/multiview/MultiviewCamera.cpp
  )

  set(${NAME}_ISPC_SOURCES
    ispc/multiview/MultiviewCamera.ispc
  )
  
  list(APPEND ${NAME}_SOURCES ${${NAME}_ISPC_SOURCES})

  # ==============================================================================
  # Compile ispc code
  # ==============================================================================
  list(APPEND CMAKE_MODULE_PATH ${OSPRAY_CMAKE_ROOT})
  list(APPEND ALL_ISPC_INCLUDES ${BIOEXPLORER_SOURCE_DIRS})
  list(APPEND ALL_ISPC_INCLUDES ${${NAME}_MODULES_DIR})
  list(APPEND ALL_ISPC_INCLUDES ${OSPRAY_INCLUDE_DIRS})
  list(APPEND ALL_ISPC_INCLUDES ${GLM_INCLUDE_DIRS})
  include_directories_ispc(${ALL_ISPC_INCLUDES})
  ospray_ispc_compile(${${NAME}_ISPC_SOURCES})
  list(APPEND ${NAME}_SOURCES ${ISPC_OBJECTS})

  list(APPEND ${NAME}_PUBLIC_MODULE_LIBRARIES ${OSPRAY_LIBRARIES})
  list(APPEND ${NAME}_PRIVATE_MODULE_LIBRARIES OSPRayEngine)
  include_directories(${OSPRAY_INCLUDE_DIRS})
endif()

# ==============================================================================
# Compile c++ code
# ==============================================================================
add_library(
  ${LIBRARY_NAME}
  SHARED ${${NAME}_SOURCES}
)

set(${NAME}_LINK_LIBRARIES
    PUBLIC ${${NAME}_PUBLIC_MODULE_LIBRARIES}
    PRIVATE CoreParameters CoreCommon CoreEngine CoreIO ${${NAME}_PRIVATE_MODULE_LIBRARIES}
)

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
