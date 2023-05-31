# Copyright (c) 2015-2017, EPFL/Blue Brain Project
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of Brayns <https://github.com/BlueBrain/Brayns>

# ==============================================================================
# Project
# ==============================================================================
set(NAME CORE_UI)
set(LIBRARY_NAME CoreUI)

set(${NAME}_SOURCES BaseWindow.cpp)
set(${NAME}_PUBLIC_HEADERS BaseWindow.h)
set(${NAME}_LINK_LIBRARIES
    PUBLIC Core CoreParameters 
      ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY}
    PRIVATE glut ${OPENGL_gl_LIBRARY} ${glew_LIBRARIES}
)

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS})

set(${NAME}_INCLUDE_NAME brayns/engineapi)

add_library(
    ${LIBRARY_NAME}
    SHARED ${${NAME}_SOURCES})

target_link_libraries(${LIBRARY_NAME} ${${NAME}_LINK_LIBRARIES})

if(APPLE)
  target_compile_options(braynsUI PRIVATE -Wno-deprecated-declarations)
  target_link_libraries(braynsUI PRIVATE "-framework OpenGL")
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
