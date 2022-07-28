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

create table if not exists astrocytes.configuration
(
    guid  varchar not null
        constraint configuration_pk
            primary key,
    value varchar not null
);

create unique index if not exists configuration_guid_uindex
    on astrocytes.configuration (guid);

create table if not exists astrocytes.end_foot
(
    guid                   integer not null,
    astrocyte_guid         integer not null,
    astrocyte_section_guid integer not null,
    vertices               bytea   not null,
    indices                bytea   not null
);

create table if not exists astrocytes.model_template
(
    guid        integer not null
        constraint model_template_pk
            primary key,
    code        varchar not null,
    description varchar
);

create unique index if not exists model_template_guid_uindex
    on astrocytes.model_template (guid);

create table if not exists astrocytes.model_type
(
    guid        integer not null
        constraint model_type_pk
            primary key,
    code        varchar not null,
    description integer
);

create unique index if not exists model_type_guid_uindex
    on astrocytes.model_type (guid);

create table if not exists astrocytes.morphology_type
(
    guid        integer not null
        constraint morphology_type_pk
            primary key,
    code        varchar not null,
    description varchar
);

create unique index if not exists morphology_type_guid_uindex
    on astrocytes.morphology_type (guid);

create table if not exists astrocytes.node
(
    guid                 integer          not null
        constraint node_pk
            primary key,
    population_guid      integer          not null,
    x                    double precision not null,
    y                    double precision not null,
    z                    double precision not null,
    radius               double precision not null,
    model_template_guid  varchar          not null,
    model_type_guid      varchar          not null,
    morphology           varchar          not null,
    morphology_type_guid varchar          not null
);

create index if not exists node_population_guid_index
    on astrocytes.node (population_guid);

create table if not exists astrocytes.node_type
(
    guid        integer not null
        constraint node_type_pk
            primary key,
    code        varchar not null,
    description varchar
);

create unique index if not exists node_type_guid_uindex
    on astrocytes.node_type (guid);

create table if not exists astrocytes.population
(
    guid        integer not null
        constraint population_pk
            primary key,
    name        varchar not null,
    description varchar
);

create unique index if not exists population_guid_uindex
    on astrocytes.population (guid);

create unique index if not exists population_name_uindex
    on astrocytes.population (name);

create table if not exists astrocytes.section
(
    morphology_guid     integer not null,
    section_guid        integer not null,
    section_parent_guid integer not null,
    section_type_guid   integer not null,
    points              bytea   not null,
    constraint section_pk
        primary key (morphology_guid, section_guid)
);

create index if not exists section_morphology_guid_index
    on astrocytes.section (morphology_guid);

create table if not exists astrocytes.section_type
(
    guid        integer not null
        constraint section_type_pk
            primary key,
    description varchar not null
);

create unique index if not exists section_type_description_uindex
    on astrocytes.section_type (description);

create unique index if not exists section_type_guid_uindex
    on astrocytes.section_type (guid);