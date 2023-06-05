# Copyright (c) 2015-2023, EPFL/Blue Brain Project
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of Brayns <https://github.com/BlueBrain/Brayns>

# ==============================================================================
# Project
# ==============================================================================
set(NAME PLATFORM_MANIPULATORS)
set(LIBRARY_NAME CoreManipulators)

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES
  AbstractManipulator.cpp
  FlyingModeManipulator.cpp
  InspectCenterManipulator.cpp
)

set(${NAME}_PUBLIC_HEADERS
  AbstractManipulator.h
  FlyingModeManipulator.h
  InspectCenterManipulator.h
)

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS})

set(${NAME}_LINK_LIBRARIES
    PRIVATE ${GLM_LIBRARIES} ${Boost_PROGRAM_OPTIONS_LIBRARY}
)

set(${NAME}_INCLUDE_NAME brayns/manipulators)

add_library(
    ${LIBRARY_NAME}
    SHARED ${${NAME}_SOURCES})

target_link_libraries(
    ${LIBRARY_NAME}
    ${${NAME}_LINK_LIBRARIES}
    PUBLIC CoreCommon
    PRIVATE CoreEngine
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
