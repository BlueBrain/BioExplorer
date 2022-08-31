# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2020-2022 Blue BrainProject / EPFL
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

FROM node:10.18.0-jessie as builder
LABEL maintainer="cyrille.favreau@epfl.ch"

ARG DIST_PATH=/app/dist

WORKDIR /app
ADD . /app

RUN cd /app && \
    yarn install --pure-lockfile --non-interactive && \
    yarn build && \
    mkdir ${DIST_PATH} && \
    mv ./apps/viewer/build/* ${DIST_PATH}


FROM nginx:stable-alpine
ARG DIST_PATH=/app/dist

COPY --from=builder ${DIST_PATH} ${DIST_PATH}

# IMPORTANT: Check https://github.com/nginxinc/openshift-nginx/blob/master/Dockerfile for info on how to ensure OpenShift works with nginx.
ADD ./nginx/default.conf /etc/nginx/conf.d
ADD ./scripts/setup_nginx.sh /tmp
RUN /tmp/setup_nginx.sh && \
    rm -rf /var/cache/apk/* \
    /usr/share/man \
    /tmp

## Ports
EXPOSE 8080

STOPSIGNAL SIGTERM

## Entrypoint
# Run nginx in foreground
# See https://stackoverflow.com/a/40828098/1092007
# for more details
CMD ["nginx", "-g", "daemon off;"]
