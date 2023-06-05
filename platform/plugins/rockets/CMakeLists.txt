# Copyright (c) 2015-2018, EPFL/Blue Brain Project
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>

# ==============================================================================
# Project
# ==============================================================================
set(NAME PLATFORM_PLUGIN_ROCKETS)
set(LIBRARY_NAME CorePluginRockets)

set(${NAME}_HEADERS
  BinaryRequests.h
  ImageGenerator.h
  RocketsPlugin.h
  SnapshotTask.h
  Throttle.h
  Timeout.h
  jsonPropertyMap.h
  jsonSerialization.h
  jsonUtils.h
)

set(${NAME}_SOURCES
  ImageGenerator.cpp
  RocketsPlugin.cpp
  Throttle.cpp
  Timeout.cpp
  staticjson/staticjson.cpp
)

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

set_source_files_properties(staticjson/staticjson.cpp PROPERTIES COMPILE_FLAGS "-Wno-shadow")

set(
  ${NAME}_LINK_LIBRARIES
  PRIVATE Rockets CoreParameters CoreTasks ${FREEIMAGE_LIBRARIES}
)

if(LibLIBJPEGTURBO_FOUND)
  list(APPEND ${NAME}_LINK_LIBRARIES PRIVATE ${LIBJPEGTURBO_LIBRARIES})
endif()

if(FFMPEG_FOUND)
  list(APPEND ${NAME}_HEADERS encoder.h)
  list(APPEND ${NAME}_SOURCES encoder.cpp)
  list(APPEND ${NAME}_LINK_LIBRARIES PRIVATE ${FFMPEG_LIBRARIES})
endif()

if(Libuv_FOUND)
  list(APPEND ${NAME}_LINK_LIBRARIES PRIVATE ${Libuv_LIBRARIES})
endif()

add_library(
  ${LIBRARY_NAME}
  SHARED ${${NAME}_SOURCES})

target_compile_options(${LIBRARY_NAME} PRIVATE "-Wno-attributes")
target_link_libraries(
  ${LIBRARY_NAME}
  ${${NAME}_LINK_LIBRARIES}
  PUBLIC Async++
  PRIVATE CoreParameters ${GLM_LIBRARIES} ${Boost_PROGRAM_OPTIONS_LIBRARY} ${FREEIMAGE_LIBRARIES}
)

# needed for staticjson and rapidjson
target_include_directories(
  ${LIBRARY_NAME} SYSTEM PRIVATE
  ${FREEIMAGE_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/rapidjson/include)

if(FFMPEG_FOUND)
  target_include_directories(${LIBRARY_NAME} SYSTEM PRIVATE ${FFMPEG_INCLUDE_DIR})
endif()

# ==============================================================================
# Install binaries
# ==============================================================================
install(
  TARGETS ${LIBRARY_NAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
