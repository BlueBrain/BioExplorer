
# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2020-2023 Blue BrainProject / EPFL
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

FROM debian:buster-slim as builder
LABEL maintainer="cyrille.favreau@epfl.ch"

WORKDIR /app
ADD . /app/BioExplorer

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-setuptools python3-matplotlib libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


RUN cd /app/BioExplorer && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install wheel && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install nglview && \
    python3 -m pip install -e . && \
    python3 -m pip list

ENV PATH /app/BioExplorer:$PATH

RUN chmod +x /app/BioExplorer/bioexplorer_python_sdk

# Expose a port from the container
# For more ports, use the `--expose` flag when running the container,
# see https://docs.docker.com/engine/reference/run/#expose-incoming-ports for docs.
EXPOSE 8888

ENTRYPOINT ["bioexplorer_python_sdk"]
