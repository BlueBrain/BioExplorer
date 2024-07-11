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
set(NAME PLATFORM_COMMON)
set(LIBRARY_NAME CoreCommon)

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES
  ImageManager.cpp
  PropertyMap.cpp
  input/KeyboardHandler.cpp
  light/Light.cpp
  loader/LoaderRegistry.cpp
  material/Texture2D.cpp
  scene/ClipPlane.cpp
  simulation/AbstractAnimationHandler.cpp
  transferFunction/TransferFunction.cpp
  utils/base64/base64.cpp
  utils/DynamicLib.cpp
  utils/ImageUtils.cpp
  utils/StringUtils.cpp
  utils/Utils.cpp
  Timer.cpp
  octree/PointOctree.cpp
  octree/PointOctreeNode.cpp
  octree/VectorOctree.cpp
  octree/VectorOctreeNode.cpp
)

set(${NAME}_PUBLIC_HEADERS
  Api.h
  Any.hpp
  ActionInterface.h
  BaseObject.h
  ImageManager.h
  Progress.h
  PropertyMap.h
  PropertyObject.h
  Statistics.h
  Timer.h
  Transformation.h
  Logs.h
  MathTypes.h
  Macros.h
  CommonTypes.h
  Types.h
  geometry/CommonDefines.h
  geometry/Cone.h
  geometry/Cylinder.h
  geometry/SDFGeometry.h
  geometry/Sphere.h
  geometry/Streamline.h
  geometry/TriangleMesh.h
  geometry/Curve.h
  input/KeyboardHandler.h
  light/Light.h
  loader/Loader.h
  loader/LoaderRegistry.h
  material/Texture2D.h
  scene/ClipPlane.h
  simulation/AbstractAnimationHandler.h
  tasks/Task.h
  tasks/TaskFunctor.h
  tasks/TaskRuntimeError.h
  transferFunction/TransferFunction.h
  utils/EnumUtils.h
  utils/ImageUtils.h
  utils/StringUtils.h
  utils/Utils.h
  octree/PointOctree.h
  octree/PointOctreeNode.h
  octree/VectorOctree.h
  octree/VectorOctreeNode.h
)

set(${NAME}_HEADERS
  utils/DynamicLib.h
  utils/filesystem.h
  utils/base64/base64.h
)

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS})

add_library(
    ${LIBRARY_NAME}
    SHARED ${${NAME}_SOURCES})

target_link_libraries(
    ${LIBRARY_NAME}
    ${${NAME}_LINK_LIBRARIES}
    PUBLIC ${GLM_LIBRARIES} Async++
    PRIVATE CoreParameters ${Boost_PROGRAM_OPTIONS_LIBRARY} ${OIIO}
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

foreach ( file ${${NAME}_PUBLIC_HEADERS} )
  get_filename_component( dir ${file} DIRECTORY )
  install( FILES ${file} DESTINATION include/${PACKAGE_NAME}/core/common/${dir} )
endforeach()
