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
set(NAME DICOM)
set(LIBRARY_NAME DICOM)

# ==============================================================================
# Packages and settings
# ==============================================================================
find_package(DCMTK REQUIRED)

# Compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_STANDARD 17)

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

# ==============================================================================
# Compile c++ code
# ==============================================================================
set(${NAME}_LINK_LIBRARIES
  CoreParameters CoreCommon CoreEngine
  ${DCMTK_LIBRARIES})

include_directories(
  SYSTEM
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${BIOEXPLORER_SOURCE_DIRS}
)

add_library(
  ${LIBRARY_NAME}
  SHARED ${${NAME}_SOURCES})

target_link_libraries(
  ${LIBRARY_NAME}
  ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_FILESYSTEM_LIBRARY}
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