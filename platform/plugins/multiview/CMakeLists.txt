# Copyright (c) 2018, EPFL/Blue Brain Project
# All rights reserved. Do not distribute without permission.
# Responsible Author: Grigori Chevtchenko <grigori.chevtchenko@epfl.ch>
#
# This file is part of Brayns <https://github.com/BlueBrain/Brayns>

# ==============================================================================
# Project
# ==============================================================================
set(NAME PLATFORM_PLUGIN_MULTIVIEW)
set(LIBRARY_NAME CorePluginMultiview)

set(${NAME}_HEADERS MultiviewPlugin.h)
set(${NAME}_SOURCES MultiviewPlugin.cpp)
set(${NAME}_LINK_LIBRARIES PRIVATE CoreCommon CoreEngine CoreParameters)

# ==============================================================================
# OSPRay module
# ==============================================================================
if(USE_OSPRAY)
  list(APPEND ${NAME}_SOURCES ispc/multiview/MultiviewCamera.cpp)
  set(${NAME}_ISPC_SOURCES ispc/multiview/MultiviewCamera.ispc)
  list(APPEND ${NAME}_SOURCES ${${NAME}_ISPC_SOURCES})

  # reuse ispc setup and macros from ospray
  list(APPEND CMAKE_MODULE_PATH ${OSPRAY_CMAKE_ROOT})
  if(CMAKE_BUILD_TYPE STREQUAL Debug)
    set(OSPRAY_DEBUG_BUILD ON)
  endif()
  include(ispc)

  # ==============================================================================
  # Compile ispc code
  # ==============================================================================
  include_directories_ispc(${PROJECT_SOURCE_DIR}/../../ ${OSPRAY_INCLUDE_DIRS})
  ospray_ispc_compile(${${NAME}_ISPC_SOURCES})
  list(APPEND ${NAME}_SOURCES ${ISPC_OBJECTS})
  list(APPEND ${NAME}_LINK_LIBRARIES OSPRayEngine ospray::ospray_common ospray::ospray ospray::ospray_module_ispc)
endif()

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS})

set(${NAME}_INCLUDE_NAME plugins/multiview)

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

