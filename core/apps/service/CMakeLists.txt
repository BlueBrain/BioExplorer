# Copyright (c) 2015-2023, EPFL/Blue Brain Project
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of Brayns <https://github.com/BlueBrain/Brayns>
set(NAME service)

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES main.cpp)

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS})

set(${NAME}_LINK_LIBRARIES PUBLIC Core ${LIBUV_LIBRARIES})

if(CORE_NETWORKING_ENABLED)
  list(APPEND ${NAME}_LINK_LIBRARIES PRIVATE CorePluginRockets)

  endif()

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
