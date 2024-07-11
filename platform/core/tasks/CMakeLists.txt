# Copyright 2020 - 2024 Blue Brain Project / EPFL
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
  Errors.h
)

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS})

set(${NAME}_LINK_LIBRARIES
    PRIVATE ${GLM_LIBRARIES}
)

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

install(FILES ${${NAME}_PUBLIC_HEADERS} DESTINATION include/${PACKAGE_NAME}/core/tasks)
