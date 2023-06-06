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
set(NAME PLATFORM_ENGINE)
set(LIBRARY_NAME CoreEngine)

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES
  Camera.cpp
  Engine.cpp
  FrameBuffer.cpp
  LightManager.cpp
  Material.cpp
  Model.cpp
  Renderer.cpp
  Scene.cpp
  SharedDataVolume.cpp
  Volume.cpp
)

set(${NAME}_PUBLIC_HEADERS
  BrickedVolume.h
  Camera.h
  Engine.h
  FrameBuffer.h
  LightManager.h
  Material.h
  Model.h
  Renderer.h
  Scene.h
  SharedDataVolume.h
  Volume.h
)

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
    PUBLIC Async++
    PRIVATE CoreParameters ${GLM_LIBRARIES} ${Boost_PROGRAM_OPTIONS_LIBRARY} ${FREEIMAGE_LIBRARIES}
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
