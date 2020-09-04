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
    rm -rf /var/lib/apt/lists/*

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