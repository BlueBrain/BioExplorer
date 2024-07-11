# Copyright 2015 - 2024 Blue Brain Project / EPFL
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

#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Responsible Author: Daniel.Nachbaur@epfl.ch

set(TEST_LIBRARIES Core CoreIO CoreManipulators)

configure_file(paths.h.in ${PROJECT_BINARY_DIR}/tests/paths.h)

list(APPEND EXCLUDE_FROM_TESTS myPlugin.cpp)

if(TARGET pdiff)
  list(APPEND TEST_LIBRARIES pdiff ${FREEIMAGE_LIBRARIES})
else()
  list(APPEND EXCLUDE_FROM_TESTS
    clipPlaneRendering.cpp
    snapshot.cpp
    streamlines.cpp
  )
endif()

if(NOT OSPRAY_ENABLED)
  list(APPEND EXCLUDE_FROM_TESTS
    addModel.cpp
    addModelFromBlob.cpp
    background.cpp
    core.cpp
    clipPlaneRendering.cpp
    model.cpp
    plugin.cpp
    renderer.cpp
    shadows.cpp
    snapshot.cpp
    streamlines.cpp
    subsampling.cpp
    testImages.cpp
    transferFunction.cpp
    webAPI.cpp
    lights.cpp
  )
else()
  list(APPEND TEST_LIBRARIES OSPRayEngine)
endif()

if(NOT PLATFORM_ASSIMP_ENABLED)
  list(APPEND EXCLUDE_FROM_TESTS
    addModel.cpp
    addModelFromBlob.cpp
  )
endif()

list(APPEND EXCLUDE_FROM_TESTS shadows.cpp)

if(PLATFORM_NETWORKING_ENABLED AND OSPRAY_ENABLED)
  list(APPEND CMAKE_MODULE_PATH ${OSPRAY_CMAKE_ROOT})
  include(osprayUse)
  include_directories_ispc(${PROJECT_SOURCE_DIR})
  ospray_ispc_compile(${PROJECT_SOURCE_DIR}/engines/ospray/ispc/render/BasicRenderer.ispc)
  add_library(myPlugin SHARED
    myPlugin.cpp
    ${PROJECT_SOURCE_DIR}/plugins/Rockets/staticjson/staticjson.cpp
    ${ISPC_OBJECTS}
  )
  target_link_libraries(myPlugin
    Core
    OSPRayEngine
    ospray::ospray_common ospray::ospray
  )
  target_include_directories(myPlugin SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/plugins/Rockets
    ${PROJECT_SOURCE_DIR}/plugins/Rockets/rapidjson/include)
  list(APPEND TEST_LIBRARIES Rockets corePluginRockets myPlugin)
else()
  list(APPEND EXCLUDE_FROM_TESTS
    addModel.cpp
    addModelFromBlob.cpp
    background.cpp
    clipPlanes.cpp
    model.cpp
    plugin.cpp
    renderer.cpp
    snapshot.cpp
    throttle.cpp
    transferFunction.cpp
    webAPI.cpp
    json.cpp
  )
endif()

if(NOT OPTIX_ENABLED OR NOT PLATFORM_OPTIX_TESTS_ENABLED)
    list(APPEND EXCLUDE_FROM_TESTS demoOptiX.cpp)
endif()

include_directories(
  ${PROJECT_SOURCE_DIR}/plugins/Rockets
  ${PROJECT_SOURCE_DIR}/plugins/Rockets/rapidjson/include
)

list(APPEND LCOV_EXCLUDE '${PROJECT_SOURCE_DIR}/platform/common/utils/base64/*'
                         '${PROJECT_SOURCE_DIR}/plugins/Rockets/rapidjson/*'
                         '${PROJECT_SOURCE_DIR}/plugins/Rockets/staticjson/*')

foreach(FILE ${TESTS_SRC})
    get_filename_component(NAME ${FILE} NAME_WE)
    add_executable(${NAME} ${FILE})

    target_link_libraries(${NAME}
        PRIVATE
        Core CoreParameters CoreCommon CoreEngine CoreIO OSPRayEngine CorePluginRockets
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
        ${LIBRARY_NAME}
    )

    add_test(NAME ${NAME} COMMAND ${NAME})
endforeach()
