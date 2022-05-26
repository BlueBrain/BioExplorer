create table edge
(
    start_node_guid integer not null,
    end_node_guid   integer not null
)
    tablespace bioexplorer_ts;

alter table edge
    owner to bioexplorer;

create unique index edge_start_node_guid_end_node_guid_uindex
    on edge (start_node_guid, end_node_guid)
    tablespace bioexplorer_ts;

create table node
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
    entry_node_guid integer default 0 not null
)
    tablespace bioexplorer_ts;

alter table node
    owner to bioexplorer;

create index node_section_guid_index
    on node (section_guid)
    tablespace bioexplorer_ts;

create index node_radius_index
    on node (radius)
    tablespace bioexplorer_ts;

create table simulation_report
(
    simulation_report_guid integer                    not null
        constraint simulation_report_pk
            primary key,
    description            varchar                    not null,
    start_time             double precision default 0 not null,
    end_time               double precision           not null,
    time_step              double precision           not null,
    time_units             varchar                    not null,
    data_units             varchar                    not null
)
    tablespace bioexplorer_ts;

alter table simulation_report
    owner to bioexplorer;

create table simulation_time_series
(
    simulation_report_guid integer not null,
    frame_guid             integer not null,
    values                 bytea   not null,
    constraint simulation_time_series_pk
        primary key (simulation_report_guid, frame_guid)
)
    tablespace bioexplorer_ts;

alter table simulation_time_series
    owner to bioexplorer;

create table node_dries
(
    guid         integer          not null,
    x            smallint         not null,
    y            smallint         not null,
    z            smallint         not null,
    radius       double precision not null,
    section_guid integer          not null
);

alter table node_dries
    owner to bioexplorer;

create index node_section_guid_index_dries
    on node_dries (section_guid);

create table metadata
(
    guid  integer not null
        constraint metadata_pk
            primary key,
    name  varchar not null,
    value integer not null
)
    tablespace bioexplorer_ts;

alter table metadata
    owner to bioexplorer;

create unique index metadata_guid_uindex
    on metadata (guid)
    tablespace bioexplorer_ts;

create view section_count(count) as
SELECT count(DISTINCT node.section_guid) AS count
FROM vasculature.node;

alter table section_count
    owner to bioexplorer;

