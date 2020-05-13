cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(NAME BIOEXPLORER)
set(LIBRARY_NAME BioExplorer)
project(${NAME} VERSION 0.4.0)
set(${NAME}_VERSION_ABI 1)

# C++ 14
set(CMAKE_CXX_STANDARD 14)

find_package(Boost REQUIRED COMPONENTS program_options filesystem)
find_package(Brayns REQUIRED)
find_package(CGAL)

# Build type and options
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif(NOT CMAKE_BUILD_TYPE)

set(${NAME}_SOURCES
  common/Assembly.cpp
  common/Node.cpp
  common/Membrane.cpp
  common/Glycans.cpp
  common/Mesh.cpp
  common/Protein.cpp
  common/RNASequence.cpp
  meshing/MetaballsGenerator.cpp
  meshing/PointCloudMesher.cpp
  api/BioExplorerParams.cpp
  BioExplorer.cpp
)

set(${NAME}_PUBLIC_HEADERS
  common/Assembly.h
  common/Node.h
  common/Glycans.h
  common/Mesh.h
  common/Membrane.h
  common/Protein.h
  common/RNASequence.h
  meshing/MetaballsGenerator.h
  meshing/PointCloudMesher.h
  api/BioExplorerParams.h
  BioExplorer.h
)

set(${NAME}_LINK_LIBRARIES
    braynsParameters braynsPluginAPI braynsCommon braynsEngine braynsIO)

# Library
include_directories(
    ${BRAYNS_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR})
add_library(
    ${LIBRARY_NAME}
    SHARED ${${NAME}_SOURCES})
target_link_libraries(
    ${LIBRARY_NAME}
    ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_FILESYSTEM_LIBRARY}
    ${${NAME}_LINK_LIBRARIES})

# ================================================================================
# Install binaries
# ================================================================================
INSTALL(TARGETS ${LIBRARY_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

# Install headers
INSTALL( FILES ${${NAME}_PUBLIC_HEADERS} DESTINATION include/${LIBRARY_NAME})
