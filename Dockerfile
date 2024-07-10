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

# Image where BioExplorer plugin is built
FROM ubuntu:22.04 as builder
LABEL maintainer="cyrille.favreau@epfl.ch"
ARG DIST_PATH=/app/dist
ARG BUILD_TYPE=Release

# Install packages
RUN apt-get update \
   && apt-get -y --no-install-recommends install \
   build-essential \
   cmake \
   git \
   ninja-build \
   libarchive-dev \
   libboost-date-time-dev \
   libboost-filesystem-dev \
   libboost-iostreams-dev \
   libboost-program-options-dev \
   libboost-regex-dev \
   libboost-serialization-dev \
   libboost-system-dev \
   libboost-test-dev \
   libhdf5-serial-dev \
   libtbb2-dev \
   libturbojpeg0-dev \
   libuv1-dev \
   libpqxx-dev \
   libssl-dev \
   libcgal-dev \
   libexiv2-dev \
   libglm-dev \
   libtiff-dev \
   libmpfr-dev \
   libdcmtk-dev \
   libopenexr-dev \
   libopenimageio-dev \
   pkg-config \
   wget \
   ca-certificates \
   exiv2 \
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# --------------------------------------------------------------------------------
# Get CMake 3.21.1
# --------------------------------------------------------------------------------
RUN wget -O cmake-linux.sh https://cmake.org/files/v3.21/cmake-3.21.1-linux-x86_64.sh && \
   chmod +x cmake-linux.sh && \
   ./cmake-linux.sh --skip-license --exclude-subdir --prefix=/usr/local

# --------------------------------------------------------------------------------
# Install OpenImageIO
# https://github.com/AcademySoftwareFoundation/OpenImageIO.git
# --------------------------------------------------------------------------------
# ARG OIIO_TAG=v2.5.13.0
# ARG OIIO_SRC=/app/oiio

# RUN mkdir -p ${OIIO_SRC} \
#    && git clone https://github.com/AcademySoftwareFoundation/OpenImageIO.git ${OIIO_SRC} \
#    && cd ${OIIO_SRC} \
#    && git checkout ${OIIO_TAG} \
#    && git submodule update --init \
#    && mkdir -p build \
#    && cd build \
#    && CMAKE_PREFIX_PATH=${DIST_PATH} cmake .. -GNinja \
#    -DCMAKE_INSTALL_PREFIX=${DIST_PATH} \
#    -DUSE_OPENEXR=OFF \
#    && ninja install \
#    && ninja clean

# --------------------------------------------------------------------------------
# Install Brion
# https://github.com/BlueBrain/Brion
# --------------------------------------------------------------------------------
# ARG BRION_TAG=3.3.14
# ARG BRION_SRC=/app/brion

# RUN mkdir -p ${BRION_SRC} \
#    && git clone --recursive https://github.com/BlueBrain/Brion.git ${BRION_SRC} \
#    && cd ${BRION_SRC} \
#    && git checkout ${BRION_TAG} \
#    && git submodule update --init \
#    && mkdir -p build \
#    && cd build \
#    && CMAKE_PREFIX_PATH=${DIST_PATH} cmake .. -GNinja \
#    -DCMAKE_INSTALL_PREFIX=${DIST_PATH} \
#    -DBRION_SKIP_LIBSONATA_SUBMODULE=ON \
#    && ninja install \
#    && ninja clean

# --------------------------------------------------------------------------------
# Get ISPC
# https://github.com/ispc/ispc/releases/download/v1.12.0/ispc-v1.12.0b-linux.tar.gz
# --------------------------------------------------------------------------------
ARG ISPC_VERSION=1.12.0
ARG ISPC_DIR=ispc-v${ISPC_VERSION}b-linux
ARG ISPC_PATH=/app/$ISPC_DIR

RUN mkdir -p ${ISPC_PATH} \
   && wget --no-verbose https://github.com/ispc/ispc/releases/download/v${ISPC_VERSION}/${ISPC_DIR}.tar.gz \
   && tar zxvf ${ISPC_DIR}.tar.gz -C ${ISPC_PATH} --strip-components=1 \
   && rm -rf ${ISPC_PATH}/${ISPC_DIR}/examples

# Add ispc bin to the PATH
ENV PATH $PATH:${ISPC_PATH}/bin

# --------------------------------------------------------------------------------
# Install embree
# https://github.com/embree/embree/releases
# --------------------------------------------------------------------------------
ARG EMBREE_TAG=v3.8.0
ARG EMBREE_SRC=/app/embree

RUN mkdir -p ${EMBREE_SRC} \
   && git clone https://github.com/embree/embree.git ${EMBREE_SRC} \
   && cd ${EMBREE_SRC} \
   && git checkout ${EMBREE_TAG} \
   && mkdir -p build \
   && cd build \
   && CMAKE_PREFIX_PATH=${DIST_PATH} cmake .. -GNinja \
   -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
   -DEMBREE_ISPC_EXECUTABLE=${ISPC_PATH}/bin/ispc \
   -DEMBREE_TUTORIALS=OFF \
   -DEMBREE_IGNORE_INVALID_RAYS=ON \
   -DEMBREE_ISA_AVX512SKX=ON \
   -DCMAKE_INSTALL_PREFIX=${DIST_PATH} \
   && ninja install \
   && ninja clean

# --------------------------------------------------------------------------------
# Install OSPRay
# https://github.com/ospray/ospray/releases
# --------------------------------------------------------------------------------
ARG OSPRAY_TAG=v1.8.5
ARG OSPRAY_SRC=/app/ospray

RUN mkdir -p ${OSPRAY_SRC} \
   && git clone https://github.com/ospray/ospray.git ${OSPRAY_SRC} \
   && cd ${OSPRAY_SRC} \
   && git checkout ${OSPRAY_TAG} \
   && mkdir -p build \
   && cd build \
   && CMAKE_PREFIX_PATH=${DIST_PATH} cmake .. -GNinja \
   -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
   -DOSPRAY_ENABLE_TUTORIALS=OFF \
   -DOSPRAY_ENABLE_APPS=OFF \
   -DCMAKE_INSTALL_PREFIX=${DIST_PATH} \
   -DOSPRAY_AUTO_DOWNLOAD_TEST_IMAGES=OFF \
   -DOSPRAY_ENABLE_TUTORIALS=OFF \
   && ninja install \
   && ninja clean

# --------------------------------------------------------------------------------
# Install libwebsockets
# https://github.com/warmcat/libwebsockets/releases
# --------------------------------------------------------------------------------
ARG LWS_VERSION=2.3.0
ARG LWS_SRC=/app/libwebsockets
ARG LWS_FILE=v${LWS_VERSION}.tar.gz

RUN mkdir -p ${LWS_SRC} \
   && wget --no-verbose https://github.com/warmcat/libwebsockets/archive/${LWS_FILE} \
   && tar zxvf ${LWS_FILE} -C ${LWS_SRC} --strip-components=1 \
   && cd ${LWS_SRC} \
   && mkdir -p build \
   && cd build \
   && cmake .. -GNinja \
   -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
   -DLWS_STATIC_PIC=ON \
   -DLWS_WITH_SSL=OFF \
   -DLWS_WITH_ZLIB=OFF \
   -DLWS_WITH_ZIP_FOPS=OFF \
   -DLWS_WITHOUT_EXTENSIONS=ON \
   -DLWS_WITHOUT_TESTAPPS=ON \
   -DLWS_WITH_LIBUV=ON \
   -DCMAKE_INSTALL_PREFIX=${DIST_PATH} \
   && ninja install \
   && ninja clean

# --------------------------------------------------------------------------------
# Install Rockets
# https://github.com/BlueBrain/Rockets
# --------------------------------------------------------------------------------
ARG ROCKETS_TAG=1.0.1
ARG ROCKETS_SRC=/app/rockets

RUN mkdir -p ${ROCKETS_SRC} \
   && git clone https://github.com/favreau/Rockets.git ${ROCKETS_SRC} \
   && cd ${ROCKETS_SRC} \
   && git checkout ${ROCKETS_TAG} \
   && git submodule update --init \
   && mkdir -p build \
   && cd build \
   && CMAKE_PREFIX_PATH=${DIST_PATH} cmake .. -GNinja \
   -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
   -DCMAKE_INSTALL_PREFIX=${DIST_PATH} \
   && ninja install \
   && ninja clean

# --------------------------------------------------------------------------------
# Install Assimp
# https://github.com/assimp/assimp.git
# --------------------------------------------------------------------------------
ARG ASSIMP_TAG=v4.1.0
ARG ASSIMP_SRC=/app/assimp

RUN mkdir -p ${ASSIMP_SRC} \
   && git clone https://github.com/assimp/assimp.git ${ASSIMP_SRC} \
   && cd ${ASSIMP_SRC} \
   && git checkout ${ASSIMP_TAG} \
   && git submodule update --init \
   && mkdir -p build \
   && cd build \
   && CMAKE_PREFIX_PATH=${DIST_PATH} cmake .. -GNinja \
   -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
   -DCMAKE_INSTALL_PREFIX=${DIST_PATH} \
   && ninja install \
   && ninja clean

# --------------------------------------------------------------------------------
# Install BioExplorer
# --------------------------------------------------------------------------------
ARG BIOEXPLORER_SRC=/app
ADD . ${BIOEXPLORER_SRC}

WORKDIR /app

RUN cd ${BIOEXPLORER_SRC} \
   && rm -rf build \
   && mkdir build \
   && cd build \
   && git submodule update --init \
   && PATH=${ISPC_PATH}/bin:${PATH} \
   PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig \
   CMAKE_PREFIX_PATH=${DIST_PATH} \
   cmake .. -GNinja \
   -Wno-dev \
   -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
   -DCMAKE_INSTALL_PREFIX=${DIST_PATH} \
   -DBIOEXPLORER_USE_CGAL=OFF \
   -DPLATFORM_UNIT_TESTING_ENABLED=OFF \
   -DPLATFORM_OPTIX6_ENABLED=OFF \
   -DPLATFORM_NETWORKING_ENABLED=ON \
   -DPLATFORM_ASSIMP_ENABLED=ON \
   -DPLATFORM_VRPN_ENABLED=OFF \
   -DPLATFORM_MULTIVIEW_ENABLED=OFF \
   -DPLATFORM_OPENDECK_ENABLED=OFF \
   -DPLATFORM_MEMORY_ALIGNMENT=0 \
   -DPLATFORM_DEFLECT_ENABLED=OFF \
   -DBIOEXPLORER_SONATA_ENABLED=OFF \
   -DBIOEXPLORER_METABOLISM_ENABLED=OFF \
   -DBIOEXPLORER_MEDIA_MAKER_ENABLED=ON \
   -DMEDICALIMAGING_BUILD_ENABLED=ON \
   && ninja install \
   && ninja clean

# Final image, containing only BioExplorer and libraries required to run it
FROM ubuntu:22.04
ARG DIST_PATH=/app/dist

RUN apt-get update \
   && apt-get -y --no-install-recommends install \
   libarchive13 \
   libboost-date-time1.74.0 \
   libboost-filesystem1.74.0 \
   libboost-iostreams1.74.0 \
   libboost-program-options1.74.0 \
   libboost-regex1.74.0 \
   libboost-serialization1.74.0 \
   libboost-system1.74.0 \
   libboost-test1.74.0 \
   libhdf5-103 \
   libtbb2 \
   libturbojpeg \
   libuv1 \
   libpqxx-6.4 \
   libexiv2-27 \
   # libglm \
   libtiff5 \
   libmpfr6 \
   dcmtk \
   libopenexr25 \
   libopenimageio2.2 \
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# The COPY command below will:
# 1. create a container based on the `builder` image (but do not start it)
#    Equivalent to the `docker create` command
# 2. create a new image layer containing the
#    /app/dist directory of this new container
#    Equivalent to the `docker copy` command.
COPY --from=builder ${DIST_PATH} ${DIST_PATH}

# Add binaries from dist to the PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:${DIST_PATH}/lib
ENV PATH ${DIST_PATH}/bin:$PATH

# Expose a port from the container
# For more ports, use the `--expose` flag when running the container,
# see https://docs.docker.com/engine/reference/run/#expose-incoming-ports for docs.
EXPOSE 8200

# When running `docker run -ti --rm -p 8200:8200 bioexplorer`,
# this will be the cmd that will be executed (+ the CLI options from CMD).
# To ssh into the container (or override the default entry) use:
# `docker run -ti --rm --entrypoint bash -p 8200:8200 bioexplorer`
# See https://docs.docker.com/engine/reference/run/#entrypoint-default-command-to-execute-at-runtime
# for more docs
ENTRYPOINT ["service"]
CMD ["--http-server", ":8200", "--plugin", "MediaMaker", "--plugin", "DICOM", "--plugin", "BioExplorer"]
