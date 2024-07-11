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
