# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2015-2024 Blue BrainProject / EPFL
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
# Responsible Author: Daniel Nachbaur <daniel.nachbaur@epfl.ch>

# ==============================================================================
# Project
# ==============================================================================
set(NAME PLATFORM_PLUGIN_DEFLECT)
set(LIBRARY_NAME CorePluginDeflect)

find_package(Deflect REQUIRED)

set(${NAME}_SOURCES
  DeflectParameters.cpp
  DeflectPlugin.cpp
)
set(${NAME}_HEADERS
  DeflectParameters.h
  DeflectPlugin.h
  utils.h
)

set(${NAME}_LINK_LIBRARIES PRIVATE CoreCommon CoreEngine CoreManipulators CoreParameters Deflect)

# ==============================================================================
# OSPRay module
# ==============================================================================
if(USE_OSPRAY)
  list(APPEND ${NAME}_SOURCES DeflectPixelOp.cpp)
  list(APPEND ${NAME}_HEADERS DeflectPixelOp.h)
  list(APPEND ${NAME}_LINK_LIBRARIES PUBLIC ospray::ospray_common ospray::ospray
    PRIVATE OSPRayEngine)
endif()

if(Libuv_FOUND)
  list(APPEND ${NAME}_LINK_LIBRARIES PRIVATE ${Libuv_LIBRARIES})
endif()

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS})

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
