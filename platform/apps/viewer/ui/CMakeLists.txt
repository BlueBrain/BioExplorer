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
set(NAME PLATFORM_UI)
set(LIBRARY_NAME CoreUI)

set(${NAME}_SOURCES BaseWindow.cpp)
set(${NAME}_PUBLIC_HEADERS BaseWindow.h)
set(${NAME}_LINK_LIBRARIES
    PUBLIC Core CoreParameters 
    PRIVATE glut ${OPENGL_gl_LIBRARY} ${glew_LIBRARIES}
)

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS})

add_library(
    ${LIBRARY_NAME}
    SHARED ${${NAME}_SOURCES})

target_link_libraries(${LIBRARY_NAME} ${${NAME}_LINK_LIBRARIES})

if(APPLE)
  target_compile_options(CoreUI PRIVATE -Wno-deprecated-declarations)
  target_link_libraries(CoreUI PRIVATE "-framework OpenGL")
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
