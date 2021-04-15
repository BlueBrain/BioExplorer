
# Copyright (c) 2020, EPFL/Blue Brain Project
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of BioExplorer
# <https://github.com/BlueBrain/BioExplorer>
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License version 3.0 as published
# by the Free Software Foundation.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
# All rights reserved. Do not distribute without further notice.

FROM debian:buster-slim as builder
LABEL maintainer="cyrille.favreau@epfl.ch"

WORKDIR /app
ADD . /app/BioExplorer

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-setuptools python3-matplotlib && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


RUN cd /app/BioExplorer && \
    python3 -m pip install pip==9.0.2 && \
    python3 -m pip install brayns==1.0.0 && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install nglview && \
    python3 -m pip install -e . && \
    python3 -m pip list

ENV PATH /app/BioExplorer:$PATH

RUN chmod +x /app/BioExplorer/bioexplorer_python_sdk

ENTRYPOINT ["bioexplorer_python_sdk"]
