-- The Blue Brain BioExplorer is a tool for scientists to extract and analyse
-- scientific data from visualization
--
-- Copyright 2020-2022 Blue BrainProject / EPFL
--
-- This program is free software: you can redistribute it and/or modify it under
-- the terms of the GNU General Public License as published by the Free Software
-- Foundation, either version 3 of the License, or (at your option) any later
-- version.
--
-- This program is distributed in the hope that it will be useful, but WITHOUT
-- ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
-- FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
-- details.
--
-- You should have received a copy of the GNU General Public License along with
-- this program.  If not, see <https://www.gnu.org/licenses/>.

create table if not exists atlas.etype
(
    guid        integer not null
        constraint etypes_pk
            primary key,
    description varchar
);

create table if not exists atlas.mesh
(
    guid     integer not null
        constraint mesh_pk
            primary key,
    vertices bytea   not null,
    indices  bytea   not null,
    normals  bytea,
    colors   bytea
);

create unique index if not exists mesh_guid_uindex
    on atlas.mesh (guid);

create table if not exists atlas.region
(
    guid              integer                                     not null
        constraint region_pk
            primary key,
    code              varchar                                     not null,
    description       varchar,
    parent_guid       integer default 0                           not null,
    level             integer default 0                           not null,
    atlas_guid        integer default 0                           not null,
    ontology_guid     integer default 0                           not null,
    color_hex_triplet varchar default 'FFFFFF'::character varying not null,
    graph_order       integer default 0                           not null,
    hemisphere_guid   integer default 0                           not null
);

create table if not exists atlas.cell
(
    guid                 integer                    not null
        constraint cells_pk
            primary key,
    cell_type_guid       integer          default 0 not null,
    region_guid          integer                    not null
        constraint cell_region_guid_fk
            references atlas.region
            on update cascade on delete cascade,
    electrical_type_guid integer          default 0 not null
        constraint cell_etype_guid_fk
            references atlas.etype
            on update cascade on delete cascade,
    x                    double precision           not null,
    y                    double precision           not null,
    z                    double precision           not null,
    rotation_x           double precision default 0 not null,
    rotation_y           double precision default 0 not null,
    rotation_z           double precision default 0 not null,
    rotation_w           double precision default 1 not null
);

create index if not exists cell_region_guid_index
    on atlas.cell (region_guid);

