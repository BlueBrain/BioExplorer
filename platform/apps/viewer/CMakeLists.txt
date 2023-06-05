# Copyright (c) 2015-2016, EPFL/Blue Brain Project
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of Core <https://github.com/BlueBrain/Core>

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
  