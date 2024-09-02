# - Try to find LASLIB
# Once done this will define
#
#  LASLIB_FOUND = LASLIB_FOUND - TRUE
#  LASLIB_INCLUDE_DIR - include directory for LASlib
#  LASLIB_LIBRARIES   - the libraries (as targets)

# first look in user defined locations
set(LASLIB_FOUND 0)
find_path(
      LASLIB_INCLUDE_DIR
      NAMES
            lasreader.hpp
      PATHS
            ${_LASLIB_DIR}/include/LASlib
            ${_LASLIB_DIR}
            ${CMAKE_INSTALL_PREFIX}/include/LASlib
            /usr/local/include/LASlib
            /usr/local/include
            /usr/include
      ENV
            LASLIB_INC_DIR
)

find_library(
      LASLIB_LIBRARIES
      NAMES
            LASlib
      PATHS
      ENV
            LD_LIBRARY_PATH
      ENV
            LIBRARY_PATH
            ${LASLIB_INCLUDE_DIR}/../../lib/LASlib
            ${_LASLIB_DIR}/lib/LASlib
            ${CMAKE_INSTALL_PREFIX}/lib/LASlib
            /usr/lib/x86_64-linux-gnu
            /usr/local/pgsql/lib
            /usr/local/lib
            /usr/lib
      ENV
            LASLIB_LIB_DIR
)

if(LASLIB_LIBRARIES AND LASLIB_INCLUDE_DIR)
  set(LASLIB_FOUND 1)
  set(LASLIB_USE_FILE "UseLASLIB")
endif()

