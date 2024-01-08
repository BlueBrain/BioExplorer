# Copyright (c) 2015-2024, EPFL/Blue Brain Project
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>

# ==============================================================================
# Dependencies
# ==============================================================================
add_subdirectory(parameters)
add_subdirectory(common)
add_subdirectory(engineapi)
add_subdirectory(manipulators)
add_subdirectory(io)
add_subdirectory(tasks)
add_subdirectory(pluginapi)

# ==============================================================================
# Project
# ==============================================================================
set(NAME CORE)
set(LIBRARY_NAME Core)

unset(${NAME}_SOURCES)
unset(${NAME}_PUBLIC_HEADERS)
unset(${NAME}_HEADERS)

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES Core.cpp EngineFactory.cpp PluginManager.cpp)
set(${NAME}_HEADERS EngineFactory.h)
set(${NAME}_PUBLIC_HEADERS Core.h)

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS})

set(${NAME}_LINK_LIBRARIES PRIVATE ${Boost_PROGRAM_OPTIONS_LIBRARY})

if(NETWORKING_FOUND)
  list(APPEND ${NAME}_LINK_LIBRARIES CorePluginRockets)
endif()

add_library(${LIBRARY_NAME} SHARED ${${NAME}_SOURCES})

target_link_libraries(
    ${LIBRARY_NAME} ${${NAME}_LINK_LIBRARIES}
    PUBLIC CoreParameters CoreCommon CoreEngine
    PRIVATE CoreIO CoreManipulators CoreTasks
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

# ==============================================================================
# Packaging and configuration
# ==============================================================================
list(APPEND PLATFORM_TARGETS ${LIBRARY_NAME})  

set_target_properties(${LIBRARY_NAME}
    PROPERTIES
    VERSION ${PACKAGE_VERSION_MAJOR}
    SOVERSION ${PACKAGE_VERSION_MAJOR}.${PACKAGE_VERSION_MINOR}
)

install(TARGETS ${LIBRARY_NAME}
    EXPORT ${PACKAGE_NAME}-targets
    LIBRARY DESTINATION lib
    INCLUDES DESTINATION include)
