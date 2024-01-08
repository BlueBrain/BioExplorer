# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2020-2024 Blue BrainProject / EPFL
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

# ==============================================================================
# Project
# ==============================================================================
set(NAME PLATFORM_IO)
set(LIBRARY_NAME CoreIO)

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES
  VolumeLoader.cpp
  XYZBLoader.cpp
)

set(${NAME}_PUBLIC_HEADERS
  VolumeLoader.h
  XYZBLoader.h
)

if(BRAYNS_LIBARCHIVE_ENABLED)
  list(APPEND ${NAME}_SOURCES ArchiveLoader.cpp)
  list(APPEND ${NAME}_HEADERS ArchiveLoader.h)
  list(APPEND ${NAME}_LINK_LIBRARIES PRIVATE ${LibArchive_LIBRARIES})
endif()

if(PLATFORM_ASSIMP_ENABLED)
  list(
    APPEND ${NAME}_SOURCES
    MeshLoader.cpp
    assimpImporters/ObjFileImporter.cpp
    assimpImporters/ObjFileParser.cpp
    assimpImporters/ObjFileMtlImporter.cpp
  )
  
  if(assimp_VERSION VERSION_EQUAL 4.1.0)
    list(
      APPEND ${NAME}_SOURCES
      assimpImporters/PlyLoader.cpp assimpImporters/PlyParser.cpp
    )
    set_source_files_properties(assimpImporters/PlyLoader.cpp
      PROPERTIES COMPILE_FLAGS -Wno-shadow
    )
  endif()
  
  if(assimp_VERSION VERSION_LESS 4)
    list(APPEND ${NAME}_SOURCES assimp/DefaultIOStream.cpp)
  endif()
  
  list(APPEND ${NAME}_PUBLIC_HEADERS MeshLoader.h)
  list(APPEND ${NAME}_LINK_LIBRARIES PRIVATE ${ASSIMP_LIBRARIES})
endif()

# ==============================================================================
# Compile c++ code
# ==============================================================================
include_directories(${BIOEXPLORER_SOURCE_DIRS} ${ASSIMP_INCLUDE_DIRS})

add_library(${LIBRARY_NAME} SHARED ${${NAME}_SOURCES})

if(PLATFORM_ASSIMP_ENABLED)
  if(assimp_VERSION VERSION_EQUAL 4.1.0)
    target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_CUSTOM_PLY_IMPORTER)
  endif()
  if(assimp_VERSION VERSION_LESS 4)
    target_include_directories(${LIBRARY_NAME} PRIVATE .)
    target_compile_definitions(${LIBRARY_NAME} PRIVATE ASSIMP_VERSION_3)
  endif()
endif()

target_link_libraries(
  ${LIBRARY_NAME}
  PUBLIC ${${NAME}_LINK_LIBRARIES}
  PRIVATE CoreParameters CoreCommon CoreEngine
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

install(FILES ${${NAME}_PUBLIC_HEADERS} DESTINATION include/${PACKAGE_NAME}/core/io)
