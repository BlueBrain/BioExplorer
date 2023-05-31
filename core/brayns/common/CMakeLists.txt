# Copyright (c) 2015-2023, EPFL/Blue Brain Project
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of Brayns <https://github.com/BlueBrain/Brayns>

# ==============================================================================
# Project
# ==============================================================================
set(NAME CORE_COMMON)
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
  simulation/AbstractSimulationHandler.cpp
  transferFunction/TransferFunction.cpp
  utils/base64/base64.cpp
  utils/DynamicLib.cpp
  utils/imageUtils.cpp
  utils/stringUtils.cpp
  utils/utils.cpp
  Timer.cpp
)

set(${NAME}_PUBLIC_HEADERS
  any.hpp
  ActionInterface.h
  BaseObject.h
  ImageManager.h
  Progress.h
  PropertyMap.h
  PropertyObject.h
  Statistics.h
  Timer.h
  Transformation.h
  geometry/CommonDefines.h
  geometry/Cone.h
  geometry/Cylinder.h
  geometry/SDFGeometry.h
  geometry/SDFBezier.h
  geometry/Sphere.h
  geometry/Streamline.h
  geometry/TriangleMesh.h
  geometry/Curve.h
  input/KeyboardHandler.h
  light/Light.h
  loader/Loader.h
  loader/LoaderRegistry.h
  Logs.h
  material/Texture2D.h
  MathTypes.h
  Macros.h
  scene/ClipPlane.h
  simulation/AbstractSimulationHandler.h
  tasks/Task.h
  tasks/TaskFunctor.h
  tasks/TaskRuntimeError.h
  transferFunction/TransferFunction.h
  CommonTypes.h
  Types.h
  utils/enumUtils.h
  utils/imageUtils.h
  utils/stringUtils.h
  utils/utils.h
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

set(${NAME}_INCLUDE_NAME brayns/common)

add_library(
    ${LIBRARY_NAME}
    SHARED ${${NAME}_SOURCES})

target_link_libraries(
    ${LIBRARY_NAME}
    ${${NAME}_LINK_LIBRARIES}
    PUBLIC ${GLM_LIBRARIES} Async++
    PRIVATE CoreParameters ${Boost_PROGRAM_OPTIONS_LIBRARY} ${FreeImage_LIBRARIES}
)

# ==============================================================================
# Install binaries
# ==============================================================================
INSTALL(TARGETS ${LIBRARY_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
