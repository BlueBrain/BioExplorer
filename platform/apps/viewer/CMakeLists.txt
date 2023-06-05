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

set(NAME viewer)

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES Viewer.cpp main.cpp)
set(${NAME}_HEADERS Viewer.h)

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS})

set(
  ${NAME}_LINK_LIBRARIES
  PUBLIC Core CoreCommon CoreIO CoreParameters CoreUI CorePluginRockets ${OPENGL_gl_LIBRARY}
)

add_executable(${NAME} ${${NAME}_SOURCES})

target_link_libraries(${NAME} ${${NAME}_LINK_LIBRARIES})

# ================================================================================
# Install binaries
# ================================================================================
install(
  TARGETS ${NAME}
  RUNTIME DESTINATION bin 
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
  