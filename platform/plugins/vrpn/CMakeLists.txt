# Copyright (c) 2015-2023, EPFL/Blue Brain Project
# All rights reserved. Do not distribute without permission.
#
# This file is part of Core <https://github.com/BlueBrain/Core>

# ==============================================================================
# Project
# ==============================================================================
set(NAME PLATFORM_PLUGIN_VRPN)
set(LIBRARY_NAME CorePluginVRPN)

find_package(VRPN REQUIRED)

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS} ${VRPN_INCLUDE_DIR})

set(${NAME}_HEADERS VRPNPlugin.h)
set(${NAME}_SOURCES VRPNPlugin.cpp)
set(${NAME}_LINK_LIBRARIES PRIVATE CoreCommon CoreEngine
  ${Libuv_LIBRARIES} ${VRPN_LIBRARIES})

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
