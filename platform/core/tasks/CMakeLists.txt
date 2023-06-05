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
set(NAME CoreTasks)
set(LIBRARY_NAME ${NAME})

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES
  AddModelFromBlobTask.cpp
  AddModelTask.cpp
  LoadModelFunctor.cpp
)

set(${NAME}_PUBLIC_HEADERS
  AddModelFromBlobTask.h
  AddModelTask.h
  LoadModelFunctor.h
  errors.h
)

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS})

set(${NAME}_LINK_LIBRARIES
    PRIVATE ${GLM_LIBRARIES} ${Boost_PROGRAM_OPTIONS_LIBRARY}
)

add_library(
    ${LIBRARY_NAME}
    SHARED ${${NAME}_SOURCES})
target_compile_options(${LIBRARY_NAME} PRIVATE "-Wno-attributes")

target_link_libraries(
    ${LIBRARY_NAME}
    ${${NAME}_LINK_LIBRARIES}
    PUBLIC CoreCommon CoreEngine
)

# ==============================================================================
# Install binaries
# ==============================================================================
INSTALL(TARGETS ${LIBRARY_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
