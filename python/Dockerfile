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

FROM jupyter/scipy-notebook
LABEL maintainer="cyrille.favreau@epfl.ch"

WORKDIR /app
ADD . /app/BioExplorer

RUN python3 -m pip --version
RUN pip3 --version
RUN pip --version
RUN python -m pip --version


USER root
RUN chown -R $NB_UID /app/BioExplorer
RUN apt-get update && \
    apt-get install pkg-config -y --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


USER $NB_UID
RUN cd /app/BioExplorer && \
    pip install -r requirements.txt && \
    pip install nglview && \
    pip install "pip==9.0.2" && \
    pip install "brayns==1.0.0" && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER && \
    pip install -e . && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER && \
    pip list

ENV PATH /app/BioExplorer:$PATH

RUN chmod +x /app/BioExplorer/bioexplorer_python_sdk

ENTRYPOINT ["bioexplorer_python_sdk"]
