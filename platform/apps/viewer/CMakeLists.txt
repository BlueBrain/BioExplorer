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

set(NAME viewer)

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES Viewer.cpp main.cpp)
set(${NAME}_HEADERS Viewer.h)

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS})

set(
  ${NAME}_LINK_LIBRARIES
  PUBLIC Core CoreCommon CoreIO CoreParameters CoreUI CorePluginRockets ${OPENGL_gl_LIBRARY}
)

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
  