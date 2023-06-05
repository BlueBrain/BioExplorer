# Copyright (c) 2015-2018, EPFL/Blue Brain Project
# All rights reserved. Do not distribute without permission.
# Responsible Author: Daniel Nachbaur <daniel.nachbaur@epfl.ch>
#
# This file is part of Brayns <https://github.com/BlueBrain/Brayns>

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

set(${NAME}_INCLUDE_NAME plugins/deflect)

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
