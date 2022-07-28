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

create table if not exists connectome.glio_vascular
(
    guid                         integer          not null,
    astrocyte_population_guid    integer          not null,
    vasculature_population_guid  integer          not null,
    astrocyte_guid               integer          not null,
    astrocyte_section_guid       integer          not null,
    vasculature_node_guid        integer          not null,
    vasculature_section_guid     integer          not null,
    vasculature_segment_guid     integer          not null,
    endfoot_compartment_length   double precision not null,
    endfoot_compartment_diameter double precision not null,
    constraint glio_vascular_pk
        primary key (guid, astrocyte_population_guid, vasculature_population_guid)
);

create index if not exists glio_vascular_astrocyte_guid_index
    on connectome.glio_vascular (astrocyte_guid);

create unique index if not exists glio_vascular_astrocyte_section_guid_vasculature_section_guid_v
    on connectome.glio_vascular (astrocyte_section_guid, vasculature_section_guid, vasculature_segment_guid);

create index if not exists glio_vascular_vasculature_node_guid_astrocyte_guid_index
    on connectome.glio_vascular (vasculature_node_guid, astrocyte_guid);

create index if not exists glio_vascular_vasculature_node_guid_index
    on connectome.glio_vascular (vasculature_node_guid);

create table if not exists connectome.population
(
    guid        integer not null
        constraint population_pk
            primary key,
    schema      varchar,
    description varchar
);

create table if not exists connectome.structure
(
    guid        integer not null
        constraint structure_pk
            primary key,
    code        varchar not null,
    description varchar
);

create table if not exists connectome.streamline
(
    guid               integer not null,
    hemisphere_guid    integer not null,
    origin_region_guid integer not null
        constraint streamline_structure_guid_fk
            references connectome.structure
            on update cascade on delete cascade,
    target_region_guid integer not null
        constraint streamline_structure_guid_fk_2
            references connectome.structure
            on update cascade on delete cascade,
    points             bytea   not null
);

create unique index if not exists streamline_guid_origin_region_guid_target_region_guid_uindex
    on connectome.streamline (guid, origin_region_guid, target_region_guid);

create unique index if not exists structure_guid_uindex
    on connectome.structure (guid);
