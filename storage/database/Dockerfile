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

FROM postgres
ENV POSTGRES_PASSWORD bioexplorer
ENV POSTGRES_DB bioexplorer
COPY 01_bioexplorer_objects.sql /docker-entrypoint-initdb.d/
COPY 02_bioexplorer_neurons_schema.sql /docker-entrypoint-initdb.d/
COPY 03_bioexplorer_astrocytes_schema.sql /docker-entrypoint-initdb.d/
COPY 04_bioexplorer_vasculature_schema.sql /docker-entrypoint-initdb.d/
COPY 05_bioexplorer_connectome_schema.sql /docker-entrypoint-initdb.d/
COPY 06_bioexplorer_atlas_schema.sql /docker-entrypoint-initdb.d/
COPY 10_bioexplorer_data.sql /docker-entrypoint-initdb.d/
