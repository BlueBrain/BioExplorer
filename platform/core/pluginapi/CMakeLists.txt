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

# ==============================================================================
# Project
# ==============================================================================
set(NAME PLATFORM_PLUGIN)
set(LIBRARY_NAME CorePlugin)

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_PUBLIC_HEADERS ExtensionPlugin.h Plugin.h)

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS})

add_library(${LIBRARY_NAME} INTERFACE)

target_sources(
    ${LIBRARY_NAME}
    INTERFACE ${${NAME}_PUBLIC_HEADERS})

target_include_directories(
  ${LIBRARY_NAME}
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
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

# ==============================================================================
# Packaging and configuration
# ==============================================================================
list(APPEND PLATFORM_TARGETS ${LIBRARY_NAME})  

install(FILES ${${NAME}_PUBLIC_HEADERS} DESTINATION include/${PACKAGE_NAME}/core/pluginapi)
