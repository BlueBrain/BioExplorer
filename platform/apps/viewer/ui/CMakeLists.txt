# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2020-2024 Blue BrainProject / EPFL
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
set(NAME PLATFORM_UI)
set(LIBRARY_NAME CoreUI)

set(${NAME}_SOURCES BaseWindow.cpp)
set(${NAME}_PUBLIC_HEADERS BaseWindow.h)
set(${NAME}_LINK_LIBRARIES
    PUBLIC Core CoreParameters 
    PRIVATE glut ${OPENGL_gl_LIBRARY} ${glew_LIBRARIES}
)

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS})

add_library(
    ${LIBRARY_NAME}
    SHARED ${${NAME}_SOURCES})

target_link_libraries(${LIBRARY_NAME} ${${NAME}_LINK_LIBRARIES})

if(APPLE)
  target_compile_options(CoreUI PRIVATE -Wno-deprecated-declarations)
  target_link_libraries(CoreUI PRIVATE "-framework OpenGL")
endif()

# ==============================================================================
# Install binaries
# ==============================================================================
install(
  TARGETS ${LIBRARY_NAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
