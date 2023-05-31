# Copyright (c) 2015-2023, EPFL/Blue Brain Project
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of Brayns <https://github.com/BlueBrain/Brayns>

# ==============================================================================
# Project
# ==============================================================================
set(NAME CORE_PLUGIN)
set(LIBRARY_NAME CorePlugin)

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_PUBLIC_HEADERS ExtensionPlugin.h Plugin.h)

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS})

set(${NAME}_INCLUDE_NAME brayns/pluginapi)

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
