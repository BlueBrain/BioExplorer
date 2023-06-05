# Copyright (c) 2015-2023, EPFL/Blue Brain Project
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of Core <https://github.com/BlueBrain/Core>

# ==============================================================================
# Project
# ==============================================================================
set(NAME PLATFORM_IO)
set(LIBRARY_NAME CoreIO)

# ==============================================================================
# Sources
# ==============================================================================
set(${NAME}_SOURCES
  ProteinLoader.cpp
  VolumeLoader.cpp
  XYZBLoader.cpp
)

set(${NAME}_PUBLIC_HEADERS
  ProteinLoader.h
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
