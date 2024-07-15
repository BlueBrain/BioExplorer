# Copyright 2020 - 2023 Blue Brain Project / EPFL
#
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
