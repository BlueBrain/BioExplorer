create table population
(
    guid        integer not null
        constraint population_pk
            primary key,
    name        varchar not null,
    description varchar
);

alter table population
    owner to bioexplorer;

create unique index population_guid_uindex
    on population (guid);

create unique index population_name_uindex
    on population (name);

create table simulation_report
(
    population_guid        integer                    not null
        constraint simulation_report_population_guid_fk
            references population
            on update cascade on delete cascade,
    simulation_report_guid integer                    not null,
    description            varchar                    not null,
    start_time             double precision default 0 not null,
    end_time               double precision           not null,
    time_step              double precision           not null,
    time_units             varchar                    not null,
    data_units             varchar                    not null,
    constraint simulation_report_pk
        primary key (population_guid, simulation_report_guid)
);

alter table simulation_report
    owner to bioexplorer;

create table simulation_time_series
(
    population_guid        integer not null,
    simulation_report_guid integer not null,
    frame_guid             integer not null,
    values                 bytea   not null,
    constraint simulation_time_series_pk
        primary key (simulation_report_guid, population_guid, frame_guid),
    constraint simulation_time_series_simulation_report_simulation_report_guid
        foreign key (simulation_report_guid, population_guid) references simulation_report (simulation_report_guid, population_guid)
            on update cascade on delete cascade
);

alter table simulation_time_series
    owner to bioexplorer;

create table node
(
    guid            integer           not null
        constraint node_pk
            primary key,
    population_guid integer           not null
        constraint node_population_guid_fk
            references population,
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

create index node_population_guid_index
    on node (population_guid);

