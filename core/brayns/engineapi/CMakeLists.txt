# Copyright (c) 2015-2023, EPFL/Blue Brain Project
# Responsible Author: Daniel Nachbaur <daniel.nachbaur@epfl.ch>
#
# This file is part of Brayns <https://github.com/BlueBrain/Brayns>

# ==============================================================================
# Project
# ==============================================================================
set(NAME CORE_ENGINE)
set(LIBRARY_NAME CoreEngine)

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES
  Camera.cpp
  Engine.cpp
  FrameBuffer.cpp
  LightManager.cpp
  Material.cpp
  Model.cpp
  Renderer.cpp
  Scene.cpp
  SharedDataVolume.cpp
  Volume.cpp
)

set(${NAME}_PUBLIC_HEADERS
  BrickedVolume.h
  Camera.h
  Engine.h
  FrameBuffer.h
  LightManager.h
  Material.h
  Model.h
  Renderer.h
  Scene.h
  SharedDataVolume.h
  Volume.h
)

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS})

set(${NAME}_INCLUDE_NAME brayns/engineapi)

add_library(
    ${LIBRARY_NAME}
    SHARED ${${NAME}_SOURCES})

target_link_libraries(
    ${LIBRARY_NAME}
    ${${NAME}_LINK_LIBRARIES}
    PUBLIC Async++
    PRIVATE CoreParameters ${GLM_LIBRARIES} ${Boost_PROGRAM_OPTIONS_LIBRARY} ${FREEIMAGE_LIBRARIES}
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
