# Copyright (c) 2015-2023, EPFL/Blue Brain Project
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of Core <https://github.com/BlueBrain/Core>

# Configure the output directories. To allow IMPI to do its magic we
# will put *executables* into the (same) build directory, but tag
# mic-executables with ".mic". *libraries* cannot use the
# ".mic"-suffix trick, so we'll put libraries into separate
# directories (names 'intel64' and 'mic', respectively)

# ==============================================================================
# Project
# ==============================================================================
set(NAME OSPRAY_ENGINE)
set(LIBRARY_NAME OSPRayEngine)

# compiler selection for the non-MIC part
if(APPLE)
  set(OSPRAY_COMPILER "CLANG" CACHE STRING "compiler to use (ICC,GCC, or CLANG)")
  set_property(CACHE OSPRAY_COMPILER PROPERTY STRINGS ICC GCC CLANG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-gnu-variable-sized-type-not-at-end")
else()
  set(OSPRAY_COMPILER "GCC" CACHE STRING "compiler to use (ICC or GCC)")
  set_property(CACHE OSPRAY_COMPILER PROPERTY STRINGS ICC GCC)
endif()


set(${NAME}_ISPC_SOURCES
  ispc/camera/utils/CameraUtils.ispc
  ispc/camera/OrthographicCamera.ispc
  ispc/camera/PanoramicCamera.ispc
  ispc/camera/PerspectiveStereoCamera.ispc
  ispc/camera/FishEyeCamera.ispc
  ispc/camera/PerspectiveParallaxCamera.ispc
  ispc/geometry/Cones.ispc
  ispc/geometry/SDFBeziers.ispc
  ispc/geometry/SDFGeometries.ispc
  ispc/geometry/RayMarching.isph
  ispc/render/BasicRenderer.ispc
  ispc/render/AdvancedRenderer.ispc
  ispc/render/utils/AdvancedMaterial.ispc
  ispc/render/utils/RandomGenerator.ispc
  ispc/render/utils/SkyBox.ispc
  ispc/render/utils/RendererUtils.ispc
  ispc/render/utils/SimulationRenderer.ispc
)

set(${NAME}_SOURCES
  OSPRayCamera.cpp
  OSPRayEngine.cpp
  OSPRayFrameBuffer.cpp
  OSPRayMaterial.cpp
  OSPRayModel.cpp
  OSPRayRenderer.cpp
  OSPRayScene.cpp
  OSPRayVolume.cpp
  utils.cpp
  ispc/camera/OrthographicCamera.cpp
  ispc/camera/PanoramicCamera.cpp
  ispc/camera/FishEyeCamera.cpp
  ispc/camera/PerspectiveStereoCamera.cpp
  ispc/camera/PerspectiveParallaxCamera.cpp
  ispc/geometry/Cones.cpp
  ispc/geometry/SDFBeziers.cpp
  ispc/geometry/SDFGeometries.cpp
  ispc/render/BasicRenderer.cpp
  ispc/render/AdvancedRenderer.cpp
  ispc/render/utils/AdvancedMaterial.cpp
  ispc/render/utils/AbstractRenderer.cpp
  ispc/render/utils/SimulationRenderer.cpp
)
list(APPEND ${NAME}_SOURCES ${${NAME}_ISPC_SOURCES})

set(${NAME}_HEADERS
  utils.h
  OSPRayCamera.h
  OSPRayEngine.h
  OSPRayFrameBuffer.h
  OSPRayMaterial.h
  OSPRayModel.h
  OSPRayRenderer.h
  OSPRayScene.h
  OSPRayVolume.h
  ispc/camera/OrthographicCamera.h
  ispc/camera/PanoramicCamera.h 
  ispc/camera/PerspectiveStereoCamera.h
  ispc/camera/FishEyeCamera.h
  ispc/camera/PerspectiveParallaxCamera.h
  ispc/render/BasicRenderer.h
  ispc/render/AdvancedRenderer.h
  ispc/render/utils/AdvancedMaterial.h
  ispc/render/utils/AbstractRenderer.h
  ispc/render/utils/SimulationRenderer.h
)

set(${NAME}_PUBLIC_HEADERS
  ispc/camera/utils/CameraUtils.ih
  ispc/geometry/Cones.h
  ispc/geometry/SDFBeziers.h
  ispc/geometry/SDFGeometries.h
  ispc/render/BasicRenderer.ih
  ispc/render/AdvancedRenderer.ih
  ispc/render/utils/SkyBox.ih
  ispc/render/utils/RendererUtils.ih
  ispc/render/utils/RandomGenerator.ih
  ispc/render/utils/AdvancedMaterial.h
  ispc/render/utils/AdvancedMaterial.ih
  ispc/render/utils/AbstractRenderer.h
  ispc/render/utils/AbstractRenderer.ih
  ispc/render/utils/SimulationRenderer.h
  ispc/render/utils/SimulationRenderer.ih
)

set_source_files_properties(
    OSPRayScene.cpp
    OSPRayMaterial.cpp
    PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)

set(${NAME}_LINK_LIBRARIES
    PRIVATE CoreEngine CoreParameters CoreCommon ospray::ospray_common ospray::ospray
    # Public linkage needed for transfer function test
    PUBLIC ospray::ospray_module_ispc
)

# reuse ispc setup and macros from ospray
list(APPEND CMAKE_MODULE_PATH ${OSPRAY_CMAKE_ROOT})
if(CMAKE_BUILD_TYPE STREQUAL Debug)
  set(OSPRAY_DEBUG_BUILD ON)
endif()
include(osprayUse)

# Compile ispc code
include_directories_ispc(
  ${PROJECT_SOURCE_DIR}
  ${BIOEXPLORER_SOURCE_DIRS}
)

ospray_ispc_compile(${${NAME}_ISPC_SOURCES})
list(APPEND ${NAME}_SOURCES ${ISPC_OBJECTS})

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
