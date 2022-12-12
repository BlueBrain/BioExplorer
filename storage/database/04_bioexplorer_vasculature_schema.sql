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

create table if not exists vasculature.node
(
    guid            integer           not null
        constraint node_pk
            primary key,
    x               double precision  not null,
    y               double precision  not null,
    z               double precision  not null,
    radius          double precision  not null,
    section_guid    integer default 0 not null,
    sub_graph_guid  integer default 0 not null,
    pair_guid       integer default 0 not null,
    entry_node_guid integer default 0 not null,
    region_guid     integer default 0 not null
);

create index if not exists node_section_guid_index
    on vasculature.node (section_guid);

create index if not exists node_region_guid_index
    on vasculature.node (region_guid);

create table if not exists vasculature.report
(
    guid integer                    not null
        constraint guid_pk
            primary key,
    description            varchar                    not null,
    start_time             double precision default 0 not null,
    end_time               double precision           not null,
    time_step              double precision           not null,
    time_units             varchar                    not null,
    data_units             varchar                    not null
);

create table if not exists vasculature.simulation_time_series
(
    report_guid integer not null,
    frame_guid  integer not null,
    values      bytea   not null,
    constraint simulation_time_series_pk
        primary key (report_guid, frame_guid)
);

create table if not exists vasculature.metadata
(
    guid  integer,
    name  varchar,
    value integer
);
