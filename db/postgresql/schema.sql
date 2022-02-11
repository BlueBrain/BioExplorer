create table population
(
    guid        integer not null
        constraint population_pk
            primary key,
    description varchar not null
);

alter table population
    owner to bioexplorer;

create unique index population_guid_uindex
    on population (guid);

create table edge
(
    population_guid  integer not null,
    source_node_guid integer not null,
    target_node_guid integer not null,
    constraint egde_pk
        primary key (population_guid, source_node_guid, target_node_guid)
);

alter table edge
    owner to bioexplorer;

create table vasculature
(
    population_guid integer           not null,
    node_guid       integer           not null,
    section_guid    integer default 0 not null,
    type_guid       integer default 0 not null,
    subgraph_guid   integer default 0 not null,
    entry_node_guid integer default 0 not null,
    pair_guid       integer default 0 not null
);

alter table vasculature
    owner to bioexplorer;

create table node
(
    guid            integer          not null
        constraint node_pk
            primary key,
    population_guid integer          not null,
    node_guid       integer          not null,
    x               double precision not null,
    y               double precision not null,
    z               double precision not null,
    radius          double precision not null
);

alter table node
    owner to bioexplorer;

create table simulation_time_series
(
    simulation_report_guid integer not null,
    frame_guid             integer not null,
    values                 bytea   not null,
    constraint simulation_compartments_pk
        primary key (simulation_report_guid, frame_guid)
);

alter table simulation_time_series
    owner to bioexplorer;

create table simulation_report
(
    guid            integer                    not null,
    population_guid integer                    not null,
    description     varchar                    not null,
    start_time      double precision default 0 not null,
    end_time        double precision           not null,
    time_step       double precision           not null,
    time_units      varchar                    not null,
    data_units      varchar                    not null,
    constraint simulation_report_pk
        primary key (guid, population_guid)
);

alter table simulation_report
    owner to bioexplorer;
