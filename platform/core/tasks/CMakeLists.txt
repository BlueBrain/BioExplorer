# Copyright (c) 2015-2023, EPFL/Blue Brain Project
# Responsible Author: Daniel Nachbaur <daniel.nachbaur@epfl.ch>
#
# This file is part of Brayns <https://github.com/BlueBrain/Brayns>

# ==============================================================================
# Project
# ==============================================================================
set(NAME CoreTasks)
set(LIBRARY_NAME ${NAME})

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES
  AddModelFromBlobTask.cpp
  AddModelTask.cpp
  LoadModelFunctor.cpp
)

set(${NAME}_PUBLIC_HEADERS
  AddModelFromBlobTask.h
  AddModelTask.h
  LoadModelFunctor.h
  errors.h
)

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS})

set(${NAME}_LINK_LIBRARIES
    PRIVATE ${GLM_LIBRARIES} ${Boost_PROGRAM_OPTIONS_LIBRARY}
)

set(${NAME}_INCLUDE_NAME brayns/tasks)

add_library(
    ${LIBRARY_NAME}
    SHARED ${${NAME}_SOURCES})
target_compile_options(${LIBRARY_NAME} PRIVATE "-Wno-attributes")

target_link_libraries(
    ${LIBRARY_NAME}
    ${${NAME}_LINK_LIBRARIES}
    PUBLIC CoreCommon CoreEngine
)

# ==============================================================================
# Install binaries
# ==============================================================================
INSTALL(TARGETS ${LIBRARY_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
