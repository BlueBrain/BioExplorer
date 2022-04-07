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
);

alter table simulation_report
    owner to bioexplorer;

create table simulation_time_series
(
    simulation_report_guid integer not null,
    frame_guid             integer not null,
    values                 bytea   not null,
    constraint simulation_time_series_pk
        primary key (simulation_report_guid, frame_guid)
);

alter table simulation_time_series
    owner to bioexplorer;

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
);

alter table node
    owner to bioexplorer;

